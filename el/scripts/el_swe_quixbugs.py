"""el_swe_quixbugs — substrate-RL on REAL bug dataset (QuixBugs, 31 algos).

Real bugs, real algorithms (sorting, search, graph, math), real test data
loaded from JSON. Substrate is given the buggy file + 5 candidate fix
lines (1 correct + 4 distractors); it must pick which one to apply.
Verification: actually patch the file, import, and run ALL json testcases.

This is the SWE-bench primitive at small scale: localized one-line fix
with a real verifier. No mocks, no toy data, no demo.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, importlib, random, time, traceback, signal, os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from el.thermofield.field import FieldConfig
from el.thermofield.layered import LayeredField
from el.thermofield.crossbar import SparseCrossbar


QB = Path("/tmp/qb_work")
PROGS = QB / "python_programs"
CORRECT = QB / "correct_python_programs"
JSON = QB / "json_testcases"

# Insert QuixBugs path so `python_programs.X` imports work
sys.path.insert(0, str(QB))


# ---------- Build dataset ----------
def parse_single_line_diff(buggy_path: Path, correct_path: Path
                           ) -> tuple[int, str, str] | None:
    """Find the SINGLE buggy code line by matching each buggy line to its
    closest correct line (Levenshtein) and picking the first pair with a
    small (1-8 char) edit. Skips docstring/comment/blank lines.
    """
    import difflib
    bl = buggy_path.read_text().splitlines()
    cl = correct_path.read_text().splitlines()
    # Strip docstring + comment + blank from correct for matching
    code_only = []
    in_doc = False
    for line in cl:
        s = line.strip()
        if s.startswith(('"""', "'''")):
            in_doc = not in_doc
            if s.count('"""') == 2 or s.count("'''") == 2:
                in_doc = False
            continue
        if in_doc or s.startswith("#") or s == "":
            continue
        code_only.append(line)
    candidates = []
    for i, bline in enumerate(bl):
        s = bline.strip()
        if s == "" or s.startswith("#") or s.startswith(('"""', "'''")):
            continue
        # find closest correct line by ratio (high), but with edit > 0
        best_ratio, best_cline = 0.0, None
        for cline in code_only:
            if cline == bline:
                best_ratio = 1.0
                best_cline = cline
                break
            r = difflib.SequenceMatcher(None, bline, cline).ratio()
            if r > best_ratio:
                best_ratio, best_cline = r, cline
        if best_cline is not None and best_cline != bline and best_ratio >= 0.6:
            # edit distance estimate
            edit = abs(len(bline) - len(best_cline)) + sum(
                1 for a, b in zip(bline, best_cline) if a != b)
            candidates.append((edit, i, bline, best_cline, best_ratio))
    if not candidates:
        return None
    # prefer the smallest non-trivial edit
    candidates.sort(key=lambda x: (x[0], -x[4]))
    edit, i, bline, cline, _ = candidates[0]
    if edit < 1 or edit > 30:
        return None
    return (i, bline, cline)


def build_dataset() -> list[dict]:
    """One entry per algorithm with single-line bug + json testcases."""
    tasks = []
    for json_path in sorted(JSON.glob("*.json")):
        name = json_path.stem
        bp = PROGS / f"{name}.py"
        cp = CORRECT / f"{name}.py"
        if not (bp.exists() and cp.exists()):
            continue
        d = parse_single_line_diff(bp, cp)
        if d is None:
            continue
        line_no, buggy_line, correct_line = d
        try:
            cases = [json.loads(l) for l in json_path.read_text().splitlines()
                     if l.strip()]
        except Exception:
            continue
        tasks.append({
            "name": name,
            "line_no": line_no,
            "buggy_line": buggy_line,
            "correct_line": correct_line,
            "buggy_text": bp.read_text(),
            "cases": cases,
        })
    return tasks


# ---------- Verifier (actually run the algorithm) ----------
class Timeout(Exception): pass


def _alarm(signum, frame): raise Timeout()


def run_with_patched_line(name: str, line_no: int, new_line: str,
                          cases: list, per_case_timeout: float = 1.0) -> bool:
    """Patch python_programs/{name}.py with `new_line` at `line_no`,
    re-import, run every test case, return True iff all pass."""
    src_path = PROGS / f"{name}.py"
    original = src_path.read_text()
    lines = original.splitlines()
    if not (0 <= line_no < len(lines)):
        return False
    lines[line_no] = new_line
    patched = "\n".join(lines) + ("\n" if original.endswith("\n") else "")
    try:
        src_path.write_text(patched)
        # Force fresh import
        modname = f"python_programs.{name}"
        if modname in sys.modules:
            del sys.modules[modname]
        if "python_programs" in sys.modules:
            del sys.modules["python_programs"]
        try:
            mod = importlib.import_module(modname)
        except Exception:
            return False
        fx = getattr(mod, name, None)
        if fx is None:
            return False
        for inp, expected in cases:
            try:
                signal.signal(signal.SIGALRM, _alarm)
                signal.setitimer(signal.ITIMER_REAL, per_case_timeout)
                got = fx(*inp) if isinstance(inp, list) else fx(inp)
                signal.setitimer(signal.ITIMER_REAL, 0)
            except Exception:
                signal.setitimer(signal.ITIMER_REAL, 0)
                return False
            # Some tasks return generators
            try:
                if hasattr(got, "__iter__") and not isinstance(got, (str, list, tuple, dict)):
                    got = list(got)
            except Exception:
                return False
            if got != expected:
                return False
        return True
    finally:
        src_path.write_text(original)


# ---------- Build action candidates per task ----------
def build_candidates(tasks: list[dict], n_distract: int = 4, seed: int = 0
                     ) -> list[list[str]]:
    """For each task, list of (n_distract+1) candidate lines: the correct one
    PLUS n_distract distractors drawn from OTHER tasks' correct_lines."""
    rng = random.Random(seed)
    cands = []
    for i, t in enumerate(tasks):
        pool = [tasks[j]["correct_line"] for j in range(len(tasks)) if j != i]
        # also throw in the buggy line itself as a hard distractor
        distract = rng.sample(pool, min(n_distract - 1, len(pool)))
        distract.append(t["buggy_line"])
        opts = [t["correct_line"]] + distract
        rng.shuffle(opts)
        cands.append(opts)
    return cands


# ---------- Encoding ----------
def code_to_cells(code: str, grid: int, cells_per_token: int = 6
                  ) -> list[tuple[int, int]]:
    """Bigram-based positional encoding into L0."""
    out: set[tuple[int, int]] = set()
    pad = code + "\x00"
    for i in range(len(code)):
        bg = pad[i] + pad[i + 1]
        for s in range(cells_per_token):
            h = hashlib.blake2b(f"L0|{i % 1024}|{bg}|{s}".encode(),
                                digest_size=4).digest()
            idx = int.from_bytes(h, "big") % (grid * grid)
            out.add((idx // grid, idx % grid))
    return sorted(out)


def action_cells(action_idx: int, grid: int, block: int = 8
                 ) -> list[tuple[int, int]]:
    cols_per_row = grid // block
    br, bc = divmod(action_idx, cols_per_row)
    r0, c0 = br * block, bc * block
    if r0 + block > grid or c0 + block > grid:
        return []
    return [(r0 + dr, c0 + dc) for dr in range(block) for dc in range(block)]


# ---------- Substrate-RL agent ----------
class SubstrateRL:
    def __init__(self, *, n_actions: int, grid: int = 96, n_layers: int = 3,
                 crossbar_k: int = 8, n_relax: int = 6, hebb_lr: float = 0.10,
                 seed: int = 0, exploration: float = 0.30):
        cfg = FieldConfig(rows=grid, cols=grid, diffusion_rate=0.30,
                          decay=0.005, nonlinear_alpha=2.5)
        self.layered = LayeredField(n_layers=n_layers, cfg=cfg, seed=seed,
                                    vertical_rate=0.30, firing_threshold=0.25,
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
        self.layered.reset()
        in_cells = code_to_cells(code, self.grid)
        self.layered.inject(0, in_cells, [1.0] * len(in_cells))
        for _ in range(self.n_relax):
            self._step_full()
        T_top = self.layered.layers[-1].T
        scores = np.zeros(self.n_actions, dtype=np.float32)
        for a in range(self.n_actions):
            cells = action_cells(a, self.grid)
            if cells:
                scores[a] = float(np.mean([T_top[r, c] for r, c in cells]))
        return scores

    def select(self, scores: np.ndarray) -> int:
        if self.rng.random() < self.exploration:
            return int(self.rng.integers(0, self.n_actions))
        noisy = scores + 1e-6 * self.rng.standard_normal(self.n_actions)
        return int(noisy.argmax())

    def reinforce(self, code: str, action: int, reward: float):
        signed = reward - 0.5
        if abs(signed) < 1e-6:
            return
        self.layered.reset()
        in_cells = code_to_cells(code, self.grid)
        out_cells = action_cells(action, self.grid)
        self.layered.inject(0, in_cells, [1.0] * len(in_cells))
        if out_cells:
            self.layered.inject(self.layered.n_layers - 1, out_cells,
                                [1.0] * len(out_cells))
        for _ in range(self.n_relax):
            self._step_full()
        lr = self.hebb_lr * signed
        for layer in self.layered.layers:
            T = layer.T
            avg_h = 0.5 * (T[:, :-1] + T[:, 1:])
            avg_v = 0.5 * (T[:-1, :] + T[1:, :])
            layer.C_right += lr * avg_h
            layer.C_down += lr * avg_v
            np.clip(layer.C_right, 0.05, 1.5, out=layer.C_right)
            np.clip(layer.C_down, 0.05, 1.5, out=layer.C_down)
        for l in range(self.layered.n_layers - 1):
            co = 0.5 * (self.layered.layers[l].T + self.layered.layers[l + 1].T)
            self.layered.C_v[l] += lr * co
            np.clip(self.layered.C_v[l], 0.05, 1.5, out=self.layered.C_v[l])
        T0_flat = self.layered.layers[0].T.reshape(-1)
        co_xb = T0_flat[self.crossbar.src] * T0_flat[self.crossbar.dst]
        new_C = self.crossbar.C + lr * 0.5 * co_xb
        np.clip(new_C, 0.05, 1.5, out=new_C)
        self.crossbar.C = new_C


# ---------- Train + eval ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--n-candidates", type=int, default=5)
    p.add_argument("--grid", type=int, default=96)
    p.add_argument("--n-relax", type=int, default=6)
    p.add_argument("--hebb-lr", type=float, default=0.12)
    p.add_argument("--exploration", type=float, default=0.40)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--limit-tasks", type=int, default=0)
    args = p.parse_args()

    tasks = build_dataset()
    if args.limit_tasks > 0:
        tasks = tasks[: args.limit_tasks]
    print(f"[dataset] {len(tasks)} real bug-fix tasks (QuixBugs)")
    for i, t in enumerate(tasks):
        print(f"  t{i:2d}: {t['name']:30s} L{t['line_no']:2d} "
              f"buggy={t['buggy_line'].strip()[:40]!r:42s} "
              f"-> fix={t['correct_line'].strip()[:30]!r}")

    cands_per_task = build_candidates(tasks, n_distract=args.n_candidates - 1,
                                      seed=args.seed)
    correct_idx = [opts.index(t["correct_line"])
                   for opts, t in zip(cands_per_task, tasks)]

    # Sanity: verify correct fix actually passes
    print("\n[sanity] verifying ground-truth fixes pass...")
    n_sane = 0
    for ti, t in enumerate(tasks):
        ok = run_with_patched_line(t["name"], t["line_no"],
                                   t["correct_line"], t["cases"])
        if ok: n_sane += 1
        else:
            print(f"  WARN t{ti} {t['name']}: ground-truth fix FAILS verifier")
    print(f"[sanity] {n_sane}/{len(tasks)} ground-truth fixes pass\n")

    # Drop tasks where ground truth doesn't pass
    keep = []
    for ti, t in enumerate(tasks):
        ok = run_with_patched_line(t["name"], t["line_no"],
                                   t["correct_line"], t["cases"])
        if ok:
            keep.append(ti)
    tasks = [tasks[i] for i in keep]
    cands_per_task = [cands_per_task[i] for i in keep]
    correct_idx = [correct_idx[i] for i in keep]
    print(f"[dataset] using {len(tasks)} tasks where verifier confirms truth\n")

    n_actions = args.n_candidates

    # Random baseline
    rng_eval = random.Random(7)
    rand_passes = 0
    for ti, t in enumerate(tasks):
        a = rng_eval.randrange(n_actions)
        ok = run_with_patched_line(t["name"], t["line_no"],
                                   cands_per_task[ti][a], t["cases"])
        if ok: rand_passes += 1
    rand_rate = rand_passes / max(len(tasks), 1)
    print(f"[random baseline] {rand_passes}/{len(tasks)} = {rand_rate:.2%}")

    # Substrate
    agent = SubstrateRL(n_actions=n_actions, grid=args.grid,
                        n_relax=args.n_relax, hebb_lr=args.hebb_lr,
                        exploration=args.exploration, seed=args.seed)

    def eval_greedy() -> float:
        passes = 0
        for ti, t in enumerate(tasks):
            scores = agent.forward(t["buggy_text"])
            a = int(scores.argmax())
            ok = run_with_patched_line(t["name"], t["line_no"],
                                       cands_per_task[ti][a], t["cases"])
            if ok: passes += 1
        return passes / max(len(tasks), 1)

    rng_train = random.Random(args.seed)
    history = []
    t0 = time.time()
    for ep in range(args.episodes):
        ti = rng_train.randrange(len(tasks))
        t = tasks[ti]
        scores = agent.forward(t["buggy_text"])
        a = agent.select(scores)
        ok = run_with_patched_line(t["name"], t["line_no"],
                                   cands_per_task[ti][a], t["cases"])
        reward = 1.0 if ok else 0.0
        agent.reinforce(t["buggy_text"], a, reward)
        history.append(reward)
        if (ep + 1) % args.eval_every == 0:
            roll = sum(history[-args.eval_every:]) / args.eval_every
            ev = eval_greedy()
            el = time.time() - t0
            print(f"[ep {ep+1:5d}] roll_R={roll:.2f} greedy_pass={ev:.2%} "
                  f"({el:.0f}s)")

    final = eval_greedy()
    print(f"\n[FINAL] substrate greedy: {final:.2%}  vs  "
          f"random: {rand_rate:.2%}  ({len(tasks)} real bug-fix tasks)")


if __name__ == "__main__":
    main()
