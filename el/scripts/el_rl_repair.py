"""el_rl_repair — substrate as RL policy with VERIFIED action signal.

NOT next-token prediction. The substrate proposes a code edit, the edit is
EXECUTED and verified by running its unit test, and reward-modulated local
plasticity updates the substrate. This is the SWE-bench kernel:

  buggy_code -> substrate.encode -> action (pos, char) -> apply_edit
              -> exec(code + test) -> reward in {0, 1}
              -> reward-modulated Hebbian on substrate weights

Toy task: 8 distinct one-char-bug Python snippets, each with a unit test.
Action space: (line_idx, col_idx, replacement_char). Substrate must learn
WHICH bug needs WHICH fix. Verified by actually running the patched code.

This is the substrate-RL primitive. Same loop scales to:
- multi-token edits (sequence of actions)
- real SWE-bench: action = full diff hunk, reward = test suite passes.
"""
from __future__ import annotations
import argparse, hashlib, sys, subprocess, tempfile, os, random, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from el.thermofield.field import FieldConfig
from el.thermofield.layered import LayeredField
from el.thermofield.crossbar import SparseCrossbar


# ---------- Toy SWE-bench-style tasks ----------
# Each task: (buggy_code, fix_position, fix_char, test_code)
# fix_position is (line_idx, col_idx) where buggy code differs from correct.
TASKS = [
    # 0: off-by-one in range
    ("def f(n):\n    return sum(range(n))",
     "def f(n):\n    return sum(range(n+1))",
     "assert f(4) == 10"),
    # 1: wrong operator + vs *
    ("def f(a, b):\n    return a + b",
     "def f(a, b):\n    return a * b",
     "assert f(3, 4) == 12"),
    # 2: wrong return value
    ("def f(x):\n    return x - 1",
     "def f(x):\n    return x + 1",
     "assert f(5) == 6"),
    # 3: wrong comparison
    ("def f(x):\n    return x < 0",
     "def f(x):\n    return x > 0",
     "assert f(5) == True and f(-1) == False"),
    # 4: missing abs
    ("def f(x):\n    return x",
     "def f(x):\n    return -x",
     "assert f(7) == -7"),
    # 5: wrong literal
    ("def f():\n    return 41",
     "def f():\n    return 42",
     "assert f() == 42"),
    # 6: wrong index
    ("def f(L):\n    return L[0]",
     "def f(L):\n    return L[-1]",
     "assert f([1,2,3]) == 3"),
    # 7: wrong string
    ("def f():\n    return 'no'",
     "def f():\n    return 'yes'",
     "assert f() == 'yes'"),
]


def diff_pos(buggy: str, fixed: str) -> tuple[int, int, str]:
    """Find single (line, col, char) where fixed differs from buggy.
    For multi-char replacements, returns the first diff position and the
    full replacement substring up to length-match."""
    bl, fl = buggy.split("\n"), fixed.split("\n")
    for li, (bline, fline) in enumerate(zip(bl, fl)):
        if bline != fline:
            for ci, (bc, fc) in enumerate(zip(bline, fline)):
                if bc != fc:
                    # replacement: take fixed substring from ci to end-of-diff
                    end_b, end_f = len(bline), len(fline)
                    return li, ci, fline[ci:end_f]
            # length differs at end
            return li, len(bline), fline[len(bline):]
    return -1, -1, ""


def apply_edit(buggy: str, line: int, col: int, replacement: str) -> str:
    lines = buggy.split("\n")
    if line < 0 or line >= len(lines):
        return buggy
    L = lines[line]
    new_L = L[:col] + replacement + L[col + len(replacement):]
    lines[line] = new_L
    return "\n".join(lines)


def verify(code: str, test: str, timeout: float = 3.0) -> bool:
    """ACTUALLY RUN the patched code + test in a subprocess. Returns True
    iff the test passes (no assertion error, no exception)."""
    full = code + "\n" + test + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=timeout, text=True)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try: os.unlink(path)
        except OSError: pass


# ---------- Encoding ----------
def code_to_cells(code: str, grid: int, cells_per_token: int = 8
                  ) -> list[tuple[int, int]]:
    """Encode buggy code into sparse L0 cells. Hashes BIGRAMS (ch, next_ch)
    with their position — bigrams give sharper discrimination than single
    chars on short Python snippets that share lots of boilerplate."""
    out: set[tuple[int, int]] = set()
    pad = code + "\x00"
    for i in range(len(code)):
        bg = pad[i] + pad[i + 1]
        for s in range(cells_per_token):
            h = hashlib.blake2b(f"L0|{i}|{bg}|{s}".encode(),
                                digest_size=4).digest()
            idx = int.from_bytes(h, "big") % (grid * grid)
            out.add((idx // grid, idx % grid))
    return sorted(out)


def action_to_cells(action_idx: int, n_actions: int, grid: int,
                    block: int = 6) -> list[tuple[int, int]]:
    """Each action gets a contiguous block in L2 (top layer) output region."""
    cols_per_row = grid // block
    br, bc = divmod(action_idx, cols_per_row)
    r0, c0 = br * block, bc * block
    if r0 + block > grid or c0 + block > grid:
        return []
    return [(r0 + dr, c0 + dc) for dr in range(block) for dc in range(block)]


# ---------- Substrate-RL agent ----------
class SubstrateRL:
    def __init__(self, *, n_actions: int, grid: int = 64, n_layers: int = 3,
                 crossbar_k: int = 8, n_relax: int = 8, hebb_lr: float = 0.06,
                 seed: int = 0, exploration: float = 0.15):
        cfg = FieldConfig(rows=grid, cols=grid, diffusion_rate=0.30,
                          decay=0.005, nonlinear_alpha=2.5)
        self.layered = LayeredField(n_layers=n_layers, cfg=cfg, seed=seed,
                                    vertical_rate=0.30,
                                    firing_threshold=0.25,
                                    spike_amplitude=0.25)
        self.crossbar = SparseCrossbar(n_cells=grid * grid, k=crossbar_k,
                                       seed=seed + 99, flux_rate=0.08)
        self.grid = grid
        self.n_relax = n_relax
        self.hebb_lr = hebb_lr
        self.n_actions = n_actions
        self.exploration = exploration
        self.rng = np.random.default_rng(seed + 7)

    def _step_full(self):
        self.layered.step()
        T0 = self.layered.layers[0].T
        flat = T0.reshape(-1)
        self.crossbar.step(flat)
        self.layered.layers[0].T = flat.reshape(self.grid, self.grid)
        if hasattr(self.layered.layers[0], "_clamp_positions"):
            self.layered.layers[0]._apply_clamps()

    def forward(self, code: str) -> np.ndarray:
        """Return per-action heat scores from L2 readout (no exploration)."""
        self.layered.reset()
        in_cells = code_to_cells(code, self.grid)
        self.layered.inject(0, in_cells, [1.0] * len(in_cells))
        for _ in range(self.n_relax):
            self._step_full()
        T_top = self.layered.layers[-1].T
        scores = np.zeros(self.n_actions, dtype=np.float32)
        for a in range(self.n_actions):
            cells = action_to_cells(a, self.n_actions, self.grid)
            if cells:
                scores[a] = float(np.mean([T_top[r, c] for r, c in cells]))
        return scores

    def select_action(self, scores: np.ndarray) -> int:
        """Epsilon-greedy over readout scores."""
        if self.rng.random() < self.exploration:
            return int(self.rng.integers(0, self.n_actions))
        # tie-break by tiny random noise
        noisy = scores + 1e-6 * self.rng.standard_normal(self.n_actions)
        return int(noisy.argmax())

    def reinforce(self, code: str, chosen_action: int, reward: float):
        """Reward-modulated Hebbian update.

        Re-runs forward with target action clamped on L2 to imprint the
        (input -> chosen action) association, scaled by (reward - baseline).
        Positive reward -> strengthen; negative (0 with baseline 0.5) -> weaken.
        """
        signed = reward - 0.5  # baseline subtraction
        if abs(signed) < 1e-6:
            return
        # Clamp both input AND chosen action; drive substrate to that joint state
        self.layered.reset()
        in_cells = code_to_cells(code, self.grid)
        out_cells = action_to_cells(chosen_action, self.n_actions, self.grid)
        self.layered.inject(0, in_cells, [1.0] * len(in_cells))
        if out_cells:
            self.layered.inject(self.layered.n_layers - 1, out_cells,
                                [1.0] * len(out_cells))
        for _ in range(self.n_relax):
            self._step_full()
        lr = self.hebb_lr * signed  # signed update
        # Intra-layer C update
        for layer in self.layered.layers:
            T = layer.T
            avg_h = 0.5 * (T[:, :-1] + T[:, 1:])
            avg_v = 0.5 * (T[:-1, :] + T[1:, :])
            layer.C_right += lr * avg_h
            layer.C_down += lr * avg_v
            np.clip(layer.C_right, 0.05, 1.5, out=layer.C_right)
            np.clip(layer.C_down, 0.05, 1.5, out=layer.C_down)
        # Vertical update
        for l in range(self.layered.n_layers - 1):
            co = 0.5 * (self.layered.layers[l].T + self.layered.layers[l + 1].T)
            self.layered.C_v[l] += lr * co
            np.clip(self.layered.C_v[l], 0.05, 1.5, out=self.layered.C_v[l])
        # Crossbar update
        T0_flat = self.layered.layers[0].T.reshape(-1)
        co_xb = T0_flat[self.crossbar.src] * T0_flat[self.crossbar.dst]
        new_C = self.crossbar.C + lr * 0.5 * co_xb
        np.clip(new_C, 0.05, 1.5, out=new_C)
        self.crossbar.C = new_C


# ---------- Training loop ----------
def run(args):
    # Build action catalog: (line, col, replacement) tuples per task
    actions: list[tuple[int, int, str]] = []
    for buggy, fixed, _ in TASKS:
        actions.append(diff_pos(buggy, fixed))
    n_actions = len(actions)
    print(f"[task] {len(TASKS)} bug-fix tasks, action space = {n_actions}")
    for i, (buggy, fixed, _) in enumerate(TASKS):
        l, c, r = actions[i]
        print(f"  task{i}: line {l} col {c} replace -> {r!r}")

    agent = SubstrateRL(n_actions=n_actions, grid=args.grid,
                        n_layers=args.n_layers, crossbar_k=args.crossbar_k,
                        n_relax=args.n_relax, hebb_lr=args.hebb_lr,
                        seed=args.seed, exploration=args.exploration)

    def eval_greedy() -> tuple[float, list[bool]]:
        results = []
        for ti, (buggy, _fixed, test) in enumerate(TASKS):
            scores = agent.forward(buggy)
            a = int(scores.argmax())
            line, col, repl = actions[a]
            patched = apply_edit(buggy, line, col, repl)
            ok = verify(patched, test)
            results.append(ok)
        return sum(results) / len(results), results

    rng = random.Random(args.seed)
    history = []
    t0 = time.time()
    for ep in range(args.episodes):
        ti = rng.randrange(len(TASKS))
        buggy, _fixed, test = TASKS[ti]
        scores = agent.forward(buggy)
        a = agent.select_action(scores)
        line, col, repl = actions[a]
        patched = apply_edit(buggy, line, col, repl)
        reward = 1.0 if verify(patched, test) else 0.0
        agent.reinforce(buggy, a, reward)
        history.append(reward)
        if (ep + 1) % args.eval_every == 0:
            rolling = sum(history[-args.eval_every:]) / args.eval_every
            acc, _ = eval_greedy()
            el = time.time() - t0
            print(f"[ep {ep+1:5d}] rolling_reward={rolling:.2f} "
                  f"greedy_pass_rate={acc:.2f} ({el:.1f}s)")

    print("\n[final eval]")
    acc, results = eval_greedy()
    for ti, ok in enumerate(results):
        print(f"  task{ti}: {'PASS' if ok else 'FAIL'}")
    print(f"  TOTAL: {sum(results)}/{len(results)} = {acc:.1%}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--crossbar-k", type=int, default=8)
    p.add_argument("--n-relax", type=int, default=8)
    p.add_argument("--hebb-lr", type=float, default=0.08)
    p.add_argument("--exploration", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
