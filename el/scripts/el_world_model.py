"""el_world_model — substrate-as-world-model on REAL hourly weather data.

Data: Istanbul 2024 hourly 2m-temperature, 8784 timesteps, pulled live
from Open-Meteo Archive (no API key, public source). Cached to
el/data/istanbul_2024_hourly.json.

Task: given the last W=24 hours of binned temperatures, predict the
bin of hour W+1. Pure dynamics learning, no labels other than the
next hour's own value.

Substrate model: PatternMemory stores (window_pattern, next_bin) pairs.
At test time, recall the closest stored window, return its next_bin.
This is the substrate doing what it's built for: associative recall on
real-world dynamics.

Baselines (must be beaten or honestly lost to):
  - persistence: predict next_bin = last_bin   (strong baseline on smooth dynamics)
  - daily-cycle: predict next_bin = bin at hour-24h ago  (captures diurnal)
  - AR(1) regression on continuous temps, then bin
  - global mode: predict the most frequent bin

Honest report: top-1 acc, MAE in bins, log-loss-equivalent (acc within
±1 bin), all on a held-out chronological test split (no leakage).
"""
from __future__ import annotations
import sys, json, time, urllib.request, hashlib
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from el.thermofield.field import FieldConfig
from el.thermofield.pattern_memory import PatternMemory


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "istanbul_2024_hourly.json"
URL = ("https://archive-api.open-meteo.com/v1/archive"
       "?latitude=41.0&longitude=29.0"
       "&start_date=2024-01-01&end_date=2024-12-31"
       "&hourly=temperature_2m,wind_speed_10m")


def _ffill(arr: np.ndarray) -> np.ndarray:
    if np.any(np.isnan(arr)):
        last = 0.0
        for i in range(len(arr)):
            if np.isnan(arr[i]): arr[i] = last
            else: last = arr[i]
    return arr


def load_data() -> tuple[np.ndarray, np.ndarray]:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[data] fetching {URL}")
        with urllib.request.urlopen(URL, timeout=30) as r:
            DATA_PATH.write_bytes(r.read())
    d = json.loads(DATA_PATH.read_text())
    temps = _ffill(np.array(d["hourly"]["temperature_2m"], dtype=np.float32))
    wind = _ffill(np.array(d["hourly"]["wind_speed_10m"], dtype=np.float32))
    return temps, wind


def bin_temps(temps: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Discretize into equal-frequency bins computed on the full series."""
    edges = np.quantile(temps, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-3; edges[-1] += 1e-3
    bins = np.searchsorted(edges, temps, side="right") - 1
    bins = np.clip(bins, 0, n_bins - 1).astype(np.int32)
    return bins, edges


# ---- substrate encoder: window of W bins -> sparse pattern on grid ----
def window_to_pattern(window: np.ndarray, n_bins: int, grid: int
                      ) -> list[tuple[int, int]]:
    """Each timestep occupies a row band; its bin places a column.
    Plus position-aware redundant cells for robustness."""
    W = len(window)
    cells = set()
    for t in range(W):
        row_band = min(grid - 1, (t * grid) // W)
        col_band = min(grid - 1, (int(window[t]) * grid) // n_bins)
        for dr in range(2):
            r = min(grid - 1, row_band + dr)
            c = min(grid - 1, col_band)
            cells.add((r, c))
            h = hashlib.blake2b(f"{t}|{int(window[t])}|{dr}".encode(),
                                digest_size=4).digest()
            idx = int.from_bytes(h, "big") % (grid * grid)
            cells.add((min(grid - 1, idx // grid), min(grid - 1, idx % grid)))
    return sorted(cells)


# ---- substrate world model ----
class SubstrateWorldModel:
    def __init__(self, n_bins: int, window: int, grid: int = 64,
                 max_store: int = 1500, seed: int = 0):
        self.n_bins, self.window, self.grid = n_bins, window, grid
        self.max_store = max_store
        self.cfg = FieldConfig(rows=grid, cols=grid)
        self.pm = PatternMemory(
            cfg=self.cfg, seed=seed,
            write_lr=0.07, write_steps=10, write_decay=0.005,
            recall_steps=6,
        )
        self.labels: list[int] = []  # parallel to pm.patterns

    def fit(self, bins: np.ndarray) -> None:
        """Store (window -> next_bin) pairs from training bins.
        Subsample to max_store to keep recall O(N) tractable."""
        n_pairs = len(bins) - self.window
        idx = np.linspace(0, n_pairs - 1, min(n_pairs, self.max_store)).astype(int)
        for i in idx:
            win = bins[i:i + self.window]
            nxt = int(bins[i + self.window])
            self.pm.store(window_to_pattern(win, self.n_bins, self.grid))
            self.labels.append(nxt)

    def predict_one(self, win: np.ndarray) -> int:
        cue = window_to_pattern(win, self.n_bins, self.grid)
        idx, _, _ = self.pm.recall(cue)
        return self.labels[idx] if 0 <= idx < len(self.labels) else 0

    def predict(self, bins: np.ndarray) -> np.ndarray:
        n = len(bins) - self.window
        out = np.empty(n, dtype=np.int32)
        for i in range(n):
            out[i] = self.predict_one(bins[i:i + self.window])
        return out


# ---- baselines ----
def persistence_pred(bins: np.ndarray, window: int) -> np.ndarray:
    return bins[window - 1: -1].copy()  # next = last


def daily_cycle_pred(bins: np.ndarray, window: int) -> np.ndarray:
    """Predict next = bin from 24h ago (same hour yesterday)."""
    n = len(bins) - window
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        # window covers indices [i, i+window); next is i+window.
        # 24h-prior of next is i+window-24, clipped to in-window.
        ref = max(0, i + window - 24)
        out[i] = bins[ref]
    return out


def ar1_pred(temps_train: np.ndarray, temps_test_full: np.ndarray,
             window: int, edges: np.ndarray, n_bins: int) -> np.ndarray:
    """Fit AR(1): t' = a*t + b on training residuals, apply, then bin."""
    x = temps_train[:-1]; y = temps_train[1:]
    a, b = np.polyfit(x, y, 1)
    n = len(temps_test_full) - window
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        last_t = temps_test_full[i + window - 1]
        pred_t = a * last_t + b
        bn = int(np.searchsorted(edges, pred_t, side="right") - 1)
        out[i] = max(0, min(n_bins - 1, bn))
    return out


def global_mode_pred(bins_train: np.ndarray, n: int) -> np.ndarray:
    mode = int(np.bincount(bins_train).argmax())
    return np.full(n, mode, dtype=np.int32)


# ---- eval ----
def report(name: str, pred: np.ndarray, true: np.ndarray) -> dict:
    acc1 = float((pred == true).mean())
    mae = float(np.abs(pred - true).mean())
    acc_pm1 = float((np.abs(pred - true) <= 1).mean())
    print(f"  {name:<28} top1={acc1:.3f}  ±1bin={acc_pm1:.3f}  MAE={mae:.2f} bins")
    return dict(name=name, top1=acc1, pm1=acc_pm1, mae=mae)


def shifted_target(bins: np.ndarray, window: int, horizon: int
                   ) -> tuple[np.ndarray, int]:
    """Returns (true_te, n) where the prediction at position i is for
    bin at i+window+horizon-1. horizon=1 = next hour, horizon=6 = 6h ahead."""
    end = len(bins) - window - (horizon - 1)
    return bins[window + horizon - 1:window + horizon - 1 + end].copy(), end


def run_task(name: str, signal_train: np.ndarray, signal_test_full: np.ndarray,
             signal_train_bins: np.ndarray, signal_test_bins: np.ndarray,
             n_bins: int, window: int, horizon: int,
             edges: np.ndarray) -> list[dict]:
    print(f"\n{'=' * 78}\nTASK: {name}  (window={window}h, horizon=+{horizon}h, "
          f"n_bins={n_bins})\n{'=' * 78}")
    bins_tr = signal_train_bins
    bins_te = signal_test_bins
    true_te, n_pred = shifted_target(bins_te, window, horizon)
    if n_pred <= 0: return []

    # substrate
    print(f"[substrate] storing up to 1500 (window→{horizon}h-ahead) pairs ...")
    t0 = time.time()
    swm = SubstrateWorldModel(n_bins=n_bins, window=window, grid=64,
                              max_store=1500, seed=0)
    # build (window, label) pairs for training
    n_train_pairs = len(bins_tr) - window - (horizon - 1)
    sub_idx = np.linspace(0, n_train_pairs - 1,
                          min(n_train_pairs, swm.max_store)).astype(int)
    for i in sub_idx:
        win = bins_tr[i:i + window]
        nxt = int(bins_tr[i + window + horizon - 1])
        swm.pm.store(window_to_pattern(win, n_bins, swm.grid))
        swm.labels.append(nxt)
    t_fit = time.time() - t0
    t0 = time.time()
    pred_swm = np.empty(n_pred, dtype=np.int32)
    for i in range(n_pred):
        pred_swm[i] = swm.predict_one(bins_te[i:i + window])
    t_pred = time.time() - t0
    print(f"[substrate] fit={t_fit:.1f}s  predict={t_pred:.1f}s")

    # baselines
    pred_pers = bins_te[window - 1: window - 1 + n_pred].copy()  # next = last
    pred_daily = np.array([
        bins_te[max(0, i + window + horizon - 1 - 24)] for i in range(n_pred)
    ], dtype=np.int32)
    # AR(horizon): apply AR(1) k times
    x = signal_train[:-1]; y = signal_train[1:]
    a, b = np.polyfit(x, y, 1)
    pred_ar = np.empty(n_pred, dtype=np.int32)
    for i in range(n_pred):
        v = signal_test_full[i + window - 1]
        for _ in range(horizon): v = a * v + b
        bn = int(np.searchsorted(edges, v, side="right") - 1)
        pred_ar[i] = max(0, min(n_bins - 1, bn))
    pred_mode = np.full(n_pred, int(np.bincount(bins_tr).argmax()),
                        dtype=np.int32)

    print(f"[results] n_test={n_pred}")
    rs = [
        report("global mode", pred_mode, true_te),
        report("persistence (next=last)", pred_pers, true_te),
        report("daily-cycle (24h ago)", pred_daily, true_te),
        report(f"AR(1)^{horizon} +bin", pred_ar, true_te),
        report("substrate world model", pred_swm, true_te),
    ]
    bb = max((r for r in rs if r["name"] != "substrate world model"),
             key=lambda x: x["top1"])
    sub = next(r for r in rs if r["name"] == "substrate world model")
    delta = sub["top1"] - bb["top1"]
    verdict = ("*** SUBSTRATE WINS ***" if delta > 0 else
               f"substrate loses by {-delta:.3f}")
    print(f"[verdict {name}] substrate {sub['top1']:.3f}  vs best "
          f"baseline ({bb['name']}) {bb['top1']:.3f}  Δ={delta:+.3f}  {verdict}")
    return rs


def main():
    print("=" * 78)
    print("EL WORLD MODEL — REAL hourly weather (Istanbul 2024)")
    print("=" * 78)
    temps, wind = load_data()
    print(f"[data] {len(temps)} hourly samples")
    print(f"[temp] range {temps.min():.1f}°..{temps.max():.1f}°  "
          f"mean {temps.mean():.1f}°  std {temps.std():.1f}°")
    print(f"[wind] range {wind.min():.1f}..{wind.max():.1f} km/h  "
          f"mean {wind.mean():.1f}  std {wind.std():.1f}")

    n_bins = 12
    window = 24
    split = int(0.80 * len(temps))

    temp_bins, temp_edges = bin_temps(temps, n_bins)
    wind_bins, wind_edges = bin_temps(wind, n_bins)

    summary = []
    for name, sig, bins, horizon, edges in [
        ("temperature +1h",  temps, temp_bins, 1,  temp_edges),
        ("temperature +6h",  temps, temp_bins, 6,  temp_edges),
        ("temperature +24h", temps, temp_bins, 24, temp_edges),
        ("wind +1h",         wind,  wind_bins, 1,  wind_edges),
        ("wind +6h",         wind,  wind_bins, 6,  wind_edges),
    ]:
        rs = run_task(name,
                      sig[:split], sig[split - window:],
                      bins[:split], bins[split - window:],
                      n_bins, window, horizon, edges)
        if rs:
            sub = next(r for r in rs if r["name"] == "substrate world model")
            bb = max((r for r in rs if r["name"] != "substrate world model"),
                     key=lambda x: x["top1"])
            summary.append((name, sub, bb))

    print("\n" + "=" * 78)
    print("FINAL SUMMARY  (substrate vs best baseline, top-1 acc on real test)")
    print("=" * 78)
    print(f"{'task':<20} {'substrate':>10} {'baseline':>10} {'baseline name':<24} {'Δ':>7}  verdict")
    for name, sub, bb in summary:
        d = sub["top1"] - bb["top1"]
        v = "WIN" if d > 0 else ("TIE" if abs(d) < 0.005 else "loss")
        print(f"{name:<20} {sub['top1']:>10.3f} {bb['top1']:>10.3f} "
              f"{bb['name']:<24} {d:>+7.3f}  {v}")


if __name__ == "__main__":
    main()
