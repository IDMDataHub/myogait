# -*- coding: utf-8 -*-
"""
finalise_quality.py -- synchronisation quality flags and robust L/R side mapping.

Called at the end of build_validation_dataset.py; can also be run alone on an
existing 01_data_prepared/ (fast, no re-analysis):

    python 02_python/finalise_quality.py

Why: the video's left/right labels can be mirrored w.r.t. the Vicon (walking
direction). The frame-level cross-correlation sync detects that for most runs
(r ~ 0.98). When the sync is poor (r < SYNC_R_OK, e.g. patient 0114 whose pose
labels flip L/R within the trial), the sync's side decision is not trusted: the
side mapping is then chosen from the cycle-normalised curves (the assignment
with the lowest centred RMSE over hip/knee/ankle), and flagged `side_method =
"curves"`. Frame-level analyses must use only pairs with `sync_ok == True`.

Rewrites: sync.csv (+sync_ok), side_map.csv (+side_method, costs),
curves_run_mean_side_matched_long.csv, agreement_curves.csv,
agreement_synced_timeseries.csv (+sync_ok).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

SYNC_R_OK = 0.90
COMMON_JOINTS = ("hip", "knee", "ankle")


def _curve_cost(means: pd.DataFrame, pair_id: str, swap: bool) -> float:
    g = means[(means.pair_id == pair_id) & means.joint.isin(COMMON_JOINTS)]
    v, c = g[g.system == "video"], g[g.system == "vicon"]
    costs = []
    for j in COMMON_JOINTS:
        for s in ("L", "R"):
            sv = ("R" if s == "L" else "L") if swap else s
            a = v[(v.joint == j) & (v.side == sv)].sort_values("pct")["mean"].to_numpy()
            b = c[(c.joint == j) & (c.side == s)].sort_values("pct")["mean"].to_numpy()
            if len(a) == 101 and len(b) == 101:
                costs.append(np.sqrt(np.mean(((a - a.mean()) - (b - b.mean())) ** 2)))
    return float(np.mean(costs)) if costs else float("inf")


def finalise(out_dir: str) -> None:
    from myogait_app.agreement import curve_metrics

    rd = lambda f: pd.read_csv(os.path.join(out_dir, f), dtype={"patient": str})  # noqa: E731
    sync = rd("sync.csv")
    META = ["dataset", "view", "setting"]
    runs = rd("runs.csv")
    means = rd("curves_run_mean_long.csv").drop(columns=META, errors="ignore")
    meta = runs[["pair_id"] + [c for c in META if c in runs.columns]]

    def with_meta(df):
        df = df.drop(columns=META, errors="ignore").merge(meta, on="pair_id", how="left")
        return df[["pair_id"] + [c for c in META if c in df] +
                  [c for c in df.columns if c not in ["pair_id"] + META]]

    sync["sync_ok"] = sync["r"] >= SYNC_R_OK
    rows = []
    for pair_id in runs.pair_id:
        s = sync[sync.pair_id == pair_id]
        c0, c1 = _curve_cost(means, pair_id, False), _curve_cost(means, pair_id, True)
        if len(s) and bool(s.sync_ok.iloc[0]):
            swap, method = bool(s.side_swap.iloc[0]), "sync"
        else:
            swap, method = c1 < c0, "curves"
        rows.append({"pair_id": pair_id, "side_swap": swap, "side_method": method,
                     "sync_r": float(s.r.iloc[0]) if len(s) else None,
                     "curve_cost_no_swap": round(c0, 3), "curve_cost_swap": round(c1, 3),
                     "sync_and_curves_agree": (c1 < c0) == swap,
                     "video_side_for_vicon_L": "R" if swap else "L",
                     "video_side_for_vicon_R": "L" if swap else "R"})
    side_map = pd.DataFrame(rows)
    swap_by = dict(zip(side_map.pair_id, side_map.side_swap))

    mv = means[means.system == "video"].copy()
    mv["side"] = [("R" if s_ == "L" else "L") if swap_by.get(p_, False) else s_
                  for p_, s_ in zip(mv.pair_id, mv.side)]
    mc = means[means.system == "vicon"]
    matched = pd.concat([mc, mv]).assign(side_ref=lambda d: d.side)

    both = mc.merge(mv, on=["pair_id", "patient", "group", "side", "joint", "pct"],
                    suffixes=("_vicon", "_video"))
    agree = []
    for (pair_id, pid, grp, side, joint), g in both.groupby(["pair_id", "patient", "group", "side", "joint"]):
        if joint not in COMMON_JOINTS:
            continue
        g = g.sort_values("pct")
        if len(g) < 50 or g["mean_video"].isna().any() or g["mean_vicon"].isna().any():
            continue
        m = curve_metrics(g["mean_video"].to_numpy(), g["mean_vicon"].to_numpy())
        if m:
            agree.append({"pair_id": pair_id, "patient": pid, "group": grp, "side": side,
                          "joint": joint, "side_swap": swap_by.get(pair_id, False),
                          "side_method": side_map.set_index("pair_id").side_method.get(pair_id),
                          "n_cycles_video": int(g["n_cycles_video"].iloc[0]),
                          "n_cycles_vicon": int(g["n_cycles_vicon"].iloc[0]), **m})

    ts = rd("agreement_synced_timeseries.csv")
    ts = ts.drop(columns=[c for c in ("sync_ok",) if c in ts]).merge(
        sync[["pair_id", "sync_ok"]], on="pair_id", how="left")

    sync.to_csv(os.path.join(out_dir, "sync.csv"), index=False)
    with_meta(side_map).to_csv(os.path.join(out_dir, "side_map.csv"), index=False)
    with_meta(matched).to_csv(os.path.join(out_dir, "curves_run_mean_side_matched_long.csv"), index=False)
    with_meta(pd.DataFrame(agree)).to_csv(os.path.join(out_dir, "agreement_curves.csv"), index=False)
    ts.to_csv(os.path.join(out_dir, "agreement_synced_timeseries.csv"), index=False)
    print(f"finalise: sync_ok {int(sync.sync_ok.sum())}/{len(sync)} | side by curves: "
          f"{int((side_map.side_method == 'curves').sum())} | sync & curves agree on side: "
          f"{int(side_map.loc[side_map.side_method == 'sync', 'sync_and_curves_agree'].sum())}/"
          f"{int((side_map.side_method == 'sync').sum())}")


if __name__ == "__main__":
    pkg = os.environ.get("MYOKIN_PKG") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    finalise(os.path.join(pkg, "01_data_prepared"))
