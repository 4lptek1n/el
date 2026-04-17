"""el_memory — substrate'i kullanan persistent text memory CLI.

Substrate-backed associative memory. ChatGPT'nin yaptığını DEĞİL,
yapamadığını yapar: tamamen offline, backprop'suz, edge donanımda
çalışan, partial-cue ile hatırlayan kalıcı bellek.

Komutlar:
  remember "metin"      — metni substrate'e yaz, label'ı yaz
  recall "kısmi metin"  — partial cue ile en yakın metni bul
  list                  — yazılı tüm hatıralar
  forget LABEL          — bir hatırayı işaretten sil (substrate dokunulmaz)
  stats                 — substrate durumu (boyut, kapasite, kullanım)

Persistence: ~/.el_memory/ altında MultiModalSubstrate.save() formatında.
Her komut state'i diskten yükler, işlem yapar, kaydeder.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1] / "src"))

from el.thermofield.field import FieldConfig
from el.thermofield.multi_substrate import MultiModalSubstrate

GRID = 64                                 # 64x64 = 4096 cells
DENSITY = 40                              # ~1% sparse pattern per memory
DEFAULT_DIR = Path.home() / ".el_memory"
LABELS_FILE = "labels.json"


# ---------------------------------------------------------------- encoding
def text_to_pattern(text: str, grid: int = GRID, k: int = DENSITY) -> list[tuple[int, int]]:
    """Deterministic sparse pattern from text via tokenized hashing.

    Each token contributes K/n_tokens cells; collisions are kept (sparse
    overlap is the substrate's natural binding mechanism).
    """
    tokens = text.lower().split() or [text.lower()]
    cells_per_token = max(1, k // len(tokens))
    positions: set[tuple[int, int]] = set()
    for tok in tokens:
        for salt in range(cells_per_token):
            h = hashlib.blake2b(f"{tok}|{salt}".encode(), digest_size=4).digest()
            idx = int.from_bytes(h, "big") % (grid * grid)
            positions.add((idx // grid, idx % grid))
        if len(positions) >= k:
            break
    return sorted(positions)


# ---------------------------------------------------------------- persistence
def load_state(state_dir: Path) -> tuple[MultiModalSubstrate, dict]:
    """Load substrate and label map from disk, or build a fresh pair."""
    state_dir = Path(state_dir)
    meta_path = state_dir / "meta.npz"
    if meta_path.exists():
        sub = MultiModalSubstrate.load(state_dir)
    else:
        sub = MultiModalSubstrate(cfg=FieldConfig(rows=GRID, cols=GRID), seed=0)
    labels_path = state_dir / LABELS_FILE
    if labels_path.exists():
        labels = json.loads(labels_path.read_text())
    else:
        labels = {"entries": []}
    return sub, labels


def save_state(state_dir: Path, sub: MultiModalSubstrate, labels: dict) -> None:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    sub.save(state_dir)
    (state_dir / LABELS_FILE).write_text(json.dumps(labels, indent=2,
                                                    ensure_ascii=False))


# ---------------------------------------------------------------- commands
def cmd_remember(args, state_dir: Path) -> int:
    sub, labels = load_state(state_dir)
    text = " ".join(args.text)
    pattern = text_to_pattern(text)
    sub.store_pattern(pattern)
    labels["entries"].append({"text": text, "k": len(pattern)})
    save_state(state_dir, sub, labels)
    idx = len(labels["entries"]) - 1
    print(f"[remembered #{idx}] {text}")
    print(f"  encoded as {len(pattern)} cells in {GRID}×{GRID} substrate")
    return 0


def cmd_recall(args, state_dir: Path) -> int:
    sub, labels = load_state(state_dir)
    if not labels["entries"]:
        print("nothing remembered yet — use `remember` first.")
        return 1
    cue_text = " ".join(args.cue)
    cue = text_to_pattern(cue_text)
    if not cue:
        print("empty cue.")
        return 1
    pred_idx, score, _ = sub.recall(cue)
    if pred_idx < 0 or pred_idx >= len(labels["entries"]):
        print(f"no clean match (score={score:.3f}).")
        return 1
    matched = labels["entries"][pred_idx]
    print(f"[recall] cue: '{cue_text}'")
    print(f"  → match #{pred_idx}: '{matched['text']}'")
    print(f"  Jaccard score: {score:.3f}")
    if args.top:
        # Show all candidates ranked manually
        print("  candidates:")
        for i, e in enumerate(labels["entries"]):
            print(f"    #{i}: '{e['text']}'")
    return 0


def cmd_list(args, state_dir: Path) -> int:
    _, labels = load_state(state_dir)
    if not labels["entries"]:
        print("(empty)")
        return 0
    print(f"{len(labels['entries'])} memories:")
    for i, e in enumerate(labels["entries"]):
        print(f"  #{i}: {e['text']}")
    return 0


def cmd_forget(args, state_dir: Path) -> int:
    sub, labels = load_state(state_dir)
    idx = int(args.index)
    if not (0 <= idx < len(labels["entries"])):
        print(f"index {idx} out of range (have {len(labels['entries'])})")
        return 1
    removed = labels["entries"].pop(idx)
    # Substrate weights themselves are NOT erased (graceful fade);
    # only the label index is dropped, so recall can no longer return it.
    save_state(state_dir, sub, labels)
    print(f"[forgot #{idx}] '{removed['text']}'")
    print("  (substrate trace remains and may still attract neighbors)")
    return 0


def cmd_stats(args, state_dir: Path) -> int:
    sub, labels = load_state(state_dir)
    n = len(labels["entries"])
    n_edges = sub.bank.n_edges
    density = sub.bank.density()
    nonzero = int(np.count_nonzero(sub.bank.w))
    c_mean = float(np.mean(sub.field.C_right))
    print(f"el_memory state @ {state_dir}")
    print(f"  grid:           {GRID}×{GRID} ({GRID * GRID} cells)")
    print(f"  memories:       {n}")
    print(f"  skip edges:     {n_edges} ({density * 100:.2f}% density)")
    print(f"  active edges:   {nonzero} (have learned weight)")
    print(f"  C-channel mean: {c_mean:.4f}")
    return 0


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="el_memory",
        description="Substrate-backed persistent text memory (offline, no LLM).",
    )
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                   help=f"State directory (default: {DEFAULT_DIR})")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("remember", help="store a piece of text")
    pr.add_argument("text", nargs="+")

    pc = sub.add_parser("recall", help="retrieve nearest text by partial cue")
    pc.add_argument("cue", nargs="+")
    pc.add_argument("--top", action="store_true", help="show all candidates")

    sub.add_parser("list", help="list all stored memories")

    pf = sub.add_parser("forget", help="drop a memory by index (from `list`)")
    pf.add_argument("index", type=int)

    sub.add_parser("stats", help="show substrate state")

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    state_dir = Path(args.dir)
    handlers = {
        "remember": cmd_remember,
        "recall":   cmd_recall,
        "list":     cmd_list,
        "forget":   cmd_forget,
        "stats":    cmd_stats,
    }
    return handlers[args.cmd](args, state_dir)


if __name__ == "__main__":
    sys.exit(main())
