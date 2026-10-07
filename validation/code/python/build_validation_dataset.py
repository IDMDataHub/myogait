# -*- coding: utf-8 -*-
"""
build_validation_dataset.py
===========================

Builds EVERY prepared table of the myogait validation package
(01_data_prepared/) from the ORIGINAL data (00_original/), for two datasets:

  myokinesis : iPhone held by hand and FOLLOWING the subject (clinic, 15 patients,
               10 healthy + 5 NMD) <-> Vicon 200 Hz. 1 video run <-> 1 Vicon trial,
               paired by timestamp (00_original/myokinesis/pairing/).
  bath       : Bath BioCV lab dataset (healthy adults), FIXED synchronised cameras,
               4 views (cam01 lateral-L, cam05 lateral-R, cam03 frontal, cam07 rear)
               <-> Vicon (markers.c3d). Same trial name = same walk.

Markerless  : pose = Sapiens2-quick (mediapipe33) + myogait pipeline
Reference   : Vicon markers through the SAME myogait pipeline (mg.load_c3d)

Requirements (tested): Python 3.12, myogait==0.8.9, myogait-app==0.9.0,
                       numpy, pandas.   pip install myogait==0.8.9 myogait-app==0.9.0

Usage:   python 02_python/build_validation_dataset.py          (from the package root)
Runtime: ~5-10 min (128 recordings through the full pipeline).

Every table is long/"tidy" format, keyed by `pair_id` = "<patient>_<IMG_xxxx>".
Column meanings are in 01_data_prepared/data_dictionary.csv.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import myogait as mg                                   # noqa: E402
from myogait_app.agreement import curve_metrics         # noqa: E402
from myogait_app.autoconfig import run_auto             # noqa: E402
from myogait_app.pipeline import PipelineConfig         # noqa: E402
from myogait_app.pooling import _apply_study_subject    # noqa: E402
from myogait_app.reliability import accelerometric_scalars  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths (relative to the package root; override with env var MYOKIN_PKG)
# --------------------------------------------------------------------------- #
PKG = os.environ.get("MYOKIN_PKG") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MK_PIV = os.path.join(PKG, "00_original", "myokinesis", "pivots_json")
PAIRING = os.path.join(PKG, "00_original", "myokinesis", "pairing", "run_trial_map.json")
BATH_SJ = os.path.join(PKG, "00_original", "bath", "sapiens2_json")
BATH_C3D = os.path.join(PKG, "00_original", "bath", "vicon_c3d")
BATH_PARTICIPANTS = os.path.join(PKG, "00_original", "bath", "participantData.csv")
META = ["dataset", "view", "setting"]
BATH_VIEWS = {"cam01": ("lateral_L", "Bath - lateral left (cam01)"),
              "cam05": ("lateral_R", "Bath - lateral right (cam05)"),
              "cam03": ("frontal", "Bath - frontal (cam03)"),
              "cam07": ("rear", "Bath - rear (cam07)")}
OUT = os.path.join(PKG, "01_data_prepared")
TS_DIR = os.path.join(OUT, "timeseries")
SY_DIR = os.path.join(OUT, "synced")
for d in (OUT, TS_DIR, SY_DIR):
    os.makedirs(d, exist_ok=True)

# Clinical study groups (SAIN = healthy, NMD = neuromuscular disease) are not
# distributed: they are read from a local, non-public file when present.
GROUPS_FILE = os.path.join(PKG, "00_original", "myokinesis", "groups.csv")
GROUPS = {}
if os.path.exists(GROUPS_FILE):
    with open(GROUPS_FILE, encoding="utf-8") as fh:
        for line in fh.read().splitlines()[1:]:
            pid, grp = line.split(",")[:2]
            GROUPS[pid.strip()] = grp.strip()

SIDES = ("left", "right")
SIDE_TAG = {"left": "L", "right": "R"}
# per-frame angle columns exported in timeseries/ (present on both systems)
FRAME_ANGLES = ["hip_L", "knee_L", "ankle_L", "hip_R", "knee_R", "ankle_R",
                "trunk_angle", "pelvis_tilt", "pelvis_obliquity"]
# joints compared between systems (present in both systems' normalised cycles)
COMMON_JOINTS = ("hip", "knee", "ankle")
SYNC_HZ = 100.0          # common rate for the time-synchronised series
SYNC_JOINTS = ("knee", "hip", "ankle")


def log(*a):
    print(*a, flush=True)


def num(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def resample101(c):
    c = np.asarray(c, float)
    if c.size == 101:
        return c
    return np.interp(np.linspace(0, 1, 101), np.linspace(0, 1, c.size), c)


def bath_participants():
    """Bath BioCV participant sheet -> {P03: {height_m, sex, age, mass_kg}}."""
    if not os.path.exists(BATH_PARTICIPANTS):
        return {}
    df = pd.read_csv(BATH_PARTICIPANTS).dropna(subset=["Participant Code"])
    df = df[df["Participant Code"].astype(str).str.match(r"^P\d+$")]
    return {r["Participant Code"]: {"height_m": num(r["Stature (m)"]), "sex": r["Sex"],
                                    "age": num(r["Age"]), "mass_kg": num(r["Mass (Kg)"])}
            for _, r in df.iterrows()}


def analyse(path, subject=None):
    """Full myogait pipeline, same auto-recipe as the app (Analysis/Cohort).

    `subject` (Bath) injects the participant's stature, which calibrates the
    video step/stride/speed to metres exactly as in the original Bath report
    (the Bath pose JSONs carry no anthropometry of their own).
    """
    data = mg.load_c3d(path) if path.lower().endswith(".c3d") else mg.load_json(path)
    if subject and subject.get("height_m"):
        for block in ("subject", "study"):
            if not isinstance(data.get(block), dict):
                data[block] = {}
            data[block]["height_m"] = subject["height_m"]
    study = dict(data.get("study") or {})
    base = _apply_study_subject(PipelineConfig(), study)
    res, _used, reasons = run_auto(data, path, base)
    if not res.ok:
        raise RuntimeError(f"pipeline failed on {os.path.basename(path)}")
    return data, res, "; ".join(reasons)


def frame_series(D):
    """DataFrame frame/time + per-frame angles (+ Vicon 3-D DOFs when present)."""
    fps = float((D.get("events") or {}).get("fps") or D["meta"].get("fps"))
    rows = []
    for fr in (D.get("angles") or {}).get("frames") or []:
        r = {"frame": fr.get("frame_idx"), "time_s": None}
        for k, v in fr.items():
            if k in ("frame_idx", "landmark_positions"):
                continue
            if isinstance(v, (int, float)) or v is None:
                r[k] = num(v)
        rows.append(r)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["time_s"] = df["frame"] / fps
    return df, fps


def events_rows(pair_id, system, D):
    ev = D.get("events") or {}
    fps = float(ev.get("fps") or D["meta"].get("fps"))
    out = []
    for key, side, kind in (("left_hs", "L", "HS"), ("left_to", "L", "TO"),
                            ("right_hs", "R", "HS"), ("right_to", "R", "TO")):
        for e in ev.get(key) or []:
            f = e.get("frame")
            out.append({"pair_id": pair_id, "system": system, "side": side, "event": kind,
                        "frame": f, "time_s": num(e.get("time")) if e.get("time") is not None
                        else (f / fps if f is not None else None),
                        "confidence": num(e.get("confidence")),
                        "method": ev.get("method")})
    return out


def cycles_tables(pair_id, pid, grp, system, C, fps):
    meta, params, curves = [], [], []
    for c in C.get("cycles", []):
        side = SIDE_TAG.get(str(c.get("side")).lower(), str(c.get("side")))
        uid = f"{pair_id}|{system}|{side}{c.get('cycle_id')}"
        sf, ef = c.get("start_frame"), c.get("end_frame")
        meta.append({"pair_id": pair_id, "patient": pid, "group": grp, "system": system,
                     "cycle_uid": uid, "cycle_id": c.get("cycle_id"), "side": side,
                     "start_frame": sf, "end_frame": ef, "toe_off_frame": c.get("toe_off_frame"),
                     "start_time_s": sf / fps if sf is not None else None,
                     "end_time_s": ef / fps if ef is not None else None,
                     "duration_s": num(c.get("duration")), "stance_pct": num(c.get("stance_pct")),
                     "swing_pct": num(c.get("swing_pct"))})
        for joint, arr in (c.get("angles_normalized") or {}).items():
            if arr is None:
                continue
            a = np.asarray(arr, float)
            if a.size < 10 or not np.isfinite(a).all():
                continue
            a = resample101(a)
            params.append({"pair_id": pair_id, "patient": pid, "group": grp, "system": system,
                           "cycle_uid": uid, "side": side, "joint": joint,
                           "rom": float(a.max() - a.min()), "max": float(a.max()),
                           "min": float(a.min()), "mean": float(a.mean()),
                           "pct_at_max": int(np.argmax(a)), "pct_at_min": int(np.argmin(a))})
            for pct, v in enumerate(a):
                curves.append((pair_id, pid, grp, system, uid, side, joint, pct, round(float(v), 4)))
    return meta, params, curves


def run_means(curves_df):
    """mean/sd curve per pair x system x side x joint x pct."""
    g = curves_df.groupby(["pair_id", "patient", "group", "system", "side", "joint", "pct"])["angle"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out.columns = ["pair_id", "patient", "group", "system", "side", "joint", "pct",
                   "mean", "sd", "n_cycles"]
    return out


def st_row(pair_id, pid, grp, system, S):
    st = S.get("spatiotemporal") or {}
    sl = S.get("step_length") or {}
    ws = S.get("walking_speed") or {}
    return {"pair_id": pair_id, "patient": pid, "group": grp, "system": system,
            "cadence_spm": num(st.get("cadence_steps_per_min")),
            "stride_time_s": num(st.get("stride_time_mean_s")),
            "stride_time_sd_s": num(st.get("stride_time_std_s")),
            "step_time_s": num(st.get("step_time_mean_s")),
            "stance_pct_L": num(st.get("stance_pct_left")), "stance_pct_R": num(st.get("stance_pct_right")),
            "swing_pct_L": num(st.get("swing_pct_left")), "swing_pct_R": num(st.get("swing_pct_right")),
            "double_support_pct": num(st.get("double_support_pct")),
            "step_length_L_m": num(sl.get("step_length_left")),
            "step_length_R_m": num(sl.get("step_length_right")),
            "stride_length_m": num(sl.get("stride_length_left")),
            "speed_mps": num(ws.get("speed_mean")),
            "length_unit": sl.get("unit"), "calibrated": sl.get("calibrated"),
            "length_source": sl.get("source", "pixel_scale" if system == "video" else None),
            "camera_motion": sl.get("camera_motion", "static")}


def bio_row(pair_id, pid, grp, system, data, S):
    a = accelerometric_scalars(data) or {}
    hr = S.get("harmonic_ratio") or {}
    return {"pair_id": pair_id, "patient": pid, "group": grp, "system": system,
            "IH_ap": num(a.get("index_of_harmonicity_ap")), "RMS_ap": num(a.get("rms_accel_ap")),
            "RMS_vert": num(a.get("rms_accel_vertical")), "LF_HF_ap": num(a.get("lf_hf_ratio_ap")),
            "HR_ap": num(hr.get("hr_ap")), "HR_vert": num(hr.get("hr_vertical"))}


# ---------------------------- time synchronisation -------------------------- #
def _uniform(df, col, t0, t1):
    t = np.arange(t0, t1, 1.0 / SYNC_HZ)
    s = df[["time_s", col]].dropna()
    if len(s) < 5:
        return t, np.full(t.size, np.nan)
    return t, np.interp(t, s["time_s"].values, s[col].values, left=np.nan, right=np.nan)


def synchronise(video_df, vicon_df):
    """Time offset between the Vicon trial and the video, by normalised
    cross-correlation of the concatenated sagittal hip/knee/ankle signals (L and R),
    resampled at SYNC_HZ. Allows PARTIAL overlap (>= 60 % of the shorter signal):
    in Myokinesis the short Vicon trial sits inside a long video, in Bath both
    cover roughly the same walk. Tries the natural side assignment and the L<->R
    swap. Returns dict(lag_s, r, side_swap, overlap_s) with t_video = t_vicon + lag_s.
    """
    vt0, vt1 = vicon_df["time_s"].min(), vicon_df["time_s"].max()
    wt0, wt1 = video_df["time_s"].min(), video_df["time_s"].max()
    if not all(map(np.isfinite, (vt0, vt1, wt0, wt1))):
        return None
    best = None
    for swap in (False, True):
        vic, vid = [], []
        for j in SYNC_JOINTS:
            for s_ in ("L", "R"):
                s_vid = ("R" if s_ == "L" else "L") if swap else s_
                vic.append(_uniform(vicon_df, f"{j}_{s_}", vt0, vt1)[1])
                vid.append(_uniform(video_df, f"{j}_{s_vid}", wt0, wt1)[1])
        n, m = len(vic[0]), len(vid[0])
        minov = int(0.6 * min(n, m))

        def corr_at(k):
            i0, i1 = max(0, -k), min(n, m - k)
            if i1 - i0 < minov:
                return None
            xs, ys = [], []
            for a_, b_ in zip(vic, vid):
                x_, y_ = a_[i0:i1], b_[i0 + k:i1 + k]
                ok = np.isfinite(x_) & np.isfinite(y_)
                if ok.sum() < 0.6 * (i1 - i0):
                    return None
                xs.append(x_[ok] - x_[ok].mean())
                ys.append(y_[ok] - y_[ok].mean())
            x, y = np.concatenate(xs), np.concatenate(ys)
            den = np.sqrt((x ** 2).sum() * (y ** 2).sum())
            return float((x * y).sum() / den) if den > 0 else None

        # coarse search every COARSE samples, then refine +-COARSE around the best
        # (gait signals are smooth at 100 Hz: 50 ms steps cannot miss the peak)
        COARSE = 5
        lo, hi = -(n - minov), m - minov
        cand = [(corr_at(k), k) for k in range(lo, hi + 1, COARSE)]
        cand = [c for c in cand if c[0] is not None]
        if not cand:
            continue
        _, k0 = max(cand)
        fine = [(corr_at(k), k) for k in range(max(lo, k0 - COARSE), min(hi, k0 + COARSE) + 1)]
        r, k = max(c for c in fine if c[0] is not None)
        if best is None or r > best["r"]:
            i0, i1 = max(0, -k), min(n, m - k)
            best = {"lag_s": float(wt0 + k / SYNC_HZ - vt0), "r": r,
                    "side_swap": swap, "overlap_s": float((i1 - i0) / SYNC_HZ)}
    return best


def discover_pairs():
    """Every (video, Vicon) pair of both datasets, as dicts."""
    pairs = []
    pairing = json.load(open(PAIRING, encoding="utf-8")) if os.path.exists(PAIRING) else {}
    for pid in sorted(GROUPS):
        for img, info in sorted(((pairing.get(pid) or {}).get("processed") or {}).items()):
            trial = info["trial"]
            fc = glob.glob(os.path.join(MK_PIV, pid, f"{img}_vicon_{trial}.json"))
            pairs.append({"dataset": "myokinesis", "view": "iphone_following",
                          "setting": "Myokinesis - iPhone following the subject",
                          "pair_id": f"{pid}_{img}", "patient": pid, "group": GROUPS[pid],
                          "video_run": img, "vicon_trial": trial,
                          "pairing_residual_s": num(info.get("residual_s")),
                          "video_path": os.path.join(MK_PIV, pid, f"{img}_sapiens2.json"),
                          "vicon_path": fc[0] if fc else ""})
    for f in sorted(glob.glob(os.path.join(BATH_SJ, "P*", "*.myogait.json"))):
        m = re.match(r"(P\d+)_(WALK_\d+)_(cam\d+)_", os.path.basename(f))
        if not m or m.group(3) not in BATH_VIEWS:
            continue
        P, T, cam = m.groups()
        view, setting = BATH_VIEWS[cam]
        pairs.append({"dataset": "bath", "view": view, "setting": setting,
                      "pair_id": f"{P}_{T}_{cam}", "patient": P, "group": "HEALTHY",
                      "video_run": f"{T}_{cam}", "vicon_trial": f"{P}_{T}",
                      "pairing_residual_s": 0.0,
                      "video_path": f, "vicon_path": os.path.join(BATH_C3D, P, f"{P}_{T}.c3d")})
    return pairs


def synced_frame(video_df, vicon_df, sync):
    vt0, vt1 = vicon_df["time_s"].min(), vicon_df["time_s"].max()
    t = np.arange(vt0, vt1, 1.0 / SYNC_HZ)
    out = {"t_vicon_s": np.round(t - vt0, 3), "t_video_s": np.round(t + sync["lag_s"], 3)}
    for j in ("hip", "knee", "ankle"):
        for s in ("L", "R"):
            s_vid = ("R" if s == "L" else "L") if sync["side_swap"] else s
            vc = vicon_df[["time_s", f"{j}_{s}"]].dropna()
            vd = video_df[["time_s", f"{j}_{s_vid}"]].dropna()
            out[f"vicon_{j}_{s}"] = np.interp(t, vc["time_s"], vc[f"{j}_{s}"], left=np.nan, right=np.nan)
            out[f"video_{j}_{s}"] = np.interp(t + sync["lag_s"], vd["time_s"], vd[f"{j}_{s_vid}"],
                                              left=np.nan, right=np.nan)
    return pd.DataFrame(out).round(4)


# --------------------------------------------------------------------------- #
def main():
    runs, subjects = [], {}
    ev_all, cyc_meta, cyc_params, curve_rows = [], [], [], []
    st_rows, bio_rows, agree_rows, sync_rows, sync_agree = [], [], [], [], []
    swap_by_pair = {}
    vicon_cache = {}          # one Vicon trial can be paired with several camera views
    bath_subj = bath_participants()

    only = [p for p in os.environ.get("MYOKIN_ONLY", "").split(",") if p]
    for P in discover_pairs():
        pid, grp, pair_id = P["patient"], P["group"], P["pair_id"]
        img, trial = P["video_run"], P["vicon_trial"]
        if only and pid not in only:
            continue
        fv, fc = P["video_path"], P["vicon_path"]
        if not (os.path.exists(fv) and fc and os.path.exists(fc)):
            log(f"  [skip] {pair_id}: missing file")
            continue
        try:
            dv, rv, why_v = analyse(fv, bath_subj.get(pid) if P["dataset"] == "bath" else None)
            if fc not in vicon_cache:
                vicon_cache[fc] = analyse(fc)
            dc, rc, why_c = vicon_cache[fc]
        except Exception as exc:  # noqa: BLE001
            log(f"  [skip] {pair_id}: {exc}")
            continue
        info = {"residual_s": P["pairing_residual_s"]}

        sub = dv.get("subject") or {}
        bs = bath_subj.get(pid, {})
        subjects.setdefault(pid, {"patient": pid, "group": grp, "sex": bs.get("sex"),
                                  "age": bs.get("age"), "mass_kg": bs.get("mass_kg"),
                                  "height_m": num(sub.get("height_m")) or bs.get("height_m"),
                                  "femur_mm": num(sub.get("femur_length_mm")),
                                  "tibia_mm": num(sub.get("tibia_length_mm")),
                                  "foot_mm": num(sub.get("foot_length_mm"))})

        # per-frame series
        ts_v, fps_v = frame_series(rv.data)
        ts_c, fps_c = frame_series(rc.data)
        ts_v.to_csv(os.path.join(TS_DIR, f"{pair_id}_video.csv"), index=False)
        ts_c.to_csv(os.path.join(TS_DIR, f"{pair_id}_vicon.csv"), index=False)

        # events + cycles
        ev_all += events_rows(pair_id, "video", rv.data) + events_rows(pair_id, "vicon", rc.data)
        for system, res, fps in (("video", rv, fps_v), ("vicon", rc, fps_c)):
            m, p, cu = cycles_tables(pair_id, pid, grp, system, res.cycles, fps)
            cyc_meta += m
            cyc_params += p
            curve_rows += cu

        # spatio-temporal + biomarkers
        st_rows += [st_row(pair_id, pid, grp, "video", rv.stats),
                    st_row(pair_id, pid, grp, "vicon", rc.stats)]
        bio_rows += [bio_row(pair_id, pid, grp, "video", dv, rv.stats),
                     bio_row(pair_id, pid, grp, "vicon", dc, rc.stats)]

        # time synchronisation (whole-trial, frame level)
        sync = synchronise(ts_v, ts_c)
        swap_by_pair[pair_id] = bool(sync["side_swap"]) if sync else False
        if sync:
            sdf = synced_frame(ts_v, ts_c, sync)
            sdf.to_csv(os.path.join(SY_DIR, f"{pair_id}_synced.csv"), index=False)
            sync_rows.append({"pair_id": pair_id, "patient": pid, "group": grp, **sync})
            for j in ("hip", "knee", "ankle"):
                for s in ("L", "R"):
                    a, b = sdf[f"video_{j}_{s}"].values, sdf[f"vicon_{j}_{s}"].values
                    ok = np.isfinite(a) & np.isfinite(b)
                    if ok.sum() < 20:
                        continue
                    d = a[ok] - b[ok]
                    sync_agree.append({"pair_id": pair_id, "patient": pid, "group": grp,
                                       "joint": j, "side": s, "n_samples": int(ok.sum()),
                                       "rmse": float(np.sqrt(np.mean(d ** 2))),
                                       "bias": float(d.mean()),
                                       "rmse_centered": float(np.sqrt(np.mean((d - d.mean()) ** 2))),
                                       "r": float(np.corrcoef(a[ok], b[ok])[0, 1])})

        runs.append({"pair_id": pair_id, "dataset": P["dataset"], "view": P["view"],
                     "setting": P["setting"], "patient": pid, "group": grp,
                     "video_run": img, "vicon_trial": trial,
                     "pairing_residual_s": num(info.get("residual_s")),
                     "video_fps": round(fps_v, 3), "vicon_fps": round(fps_c, 3),
                     "video_n_frames": len(ts_v), "vicon_n_frames": len(ts_c),
                     "video_duration_s": round(len(ts_v) / fps_v, 2),
                     "vicon_duration_s": round(len(ts_c) / fps_c, 2),
                     "video_n_cycles": len(rv.cycles.get("cycles", [])),
                     "vicon_n_cycles": len(rc.cycles.get("cycles", [])),
                     "video_recipe": why_v, "vicon_recipe": why_c,
                     "video_pivot": os.path.relpath(fv, PKG).replace("\\", "/"),
                     "vicon_pivot": os.path.relpath(fc, PKG).replace("\\", "/")})
        log(f"  {pair_id} <-> {trial}  cyc v/c={runs[-1]['video_n_cycles']}/"
            f"{runs[-1]['vicon_n_cycles']}  sync r={sync['r']:.2f}" if sync else
            f"  {pair_id} <-> {trial}  (no sync)")

    # ---- write tables -------------------------------------------------------
    curves = pd.DataFrame(curve_rows, columns=["pair_id", "patient", "group", "system",
                                               "cycle_uid", "side", "joint", "pct", "angle"])
    means = run_means(curves)
    # side mapping: the pose model's "left" is the anatomical left only when the
    # subject walks in the expected direction; the frame-level sync tells us when
    # the video L/R must be swapped to match the Vicon (reference) sides.
    side_map = pd.DataFrame([{"pair_id": k, "side_swap": v,
                              "video_side_for_vicon_L": "R" if v else "L",
                              "video_side_for_vicon_R": "L" if v else "R"}
                             for k, v in swap_by_pair.items()])
    mv = means[means.system == "video"].copy()
    mv["side"] = [("R" if s_ == "L" else "L") if swap_by_pair.get(p_, False) else s_
                  for p_, s_ in zip(mv.pair_id, mv.side)]      # now expressed in Vicon sides
    mc = means[means.system == "vicon"]
    both = mc.merge(mv, on=["pair_id", "patient", "group", "side", "joint", "pct"],
                    suffixes=("_vicon", "_video"))
    for (pair_id, pid, grp, side, joint), g in both.groupby(["pair_id", "patient", "group", "side", "joint"]):
        if joint not in COMMON_JOINTS:
            continue
        g = g.sort_values("pct")
        if len(g) < 50 or g["mean_video"].isna().any() or g["mean_vicon"].isna().any():
            continue
        m = curve_metrics(g["mean_video"].values, g["mean_vicon"].values)
        if m:
            agree_rows.append({"pair_id": pair_id, "patient": pid, "group": grp,
                               "side": side, "joint": joint, "side_swap": swap_by_pair.get(pair_id, False),
                               "n_cycles_video": int(g["n_cycles_video"].iloc[0]),
                               "n_cycles_vicon": int(g["n_cycles_vicon"].iloc[0]), **m})
    # side-matched per-run mean curves, ready for plotting (vicon side labels)
    matched = pd.concat([mc.assign(side_ref=mc.side), mv.assign(side_ref=mv.side)])

    tables = {
        "subjects.csv": pd.DataFrame(list(subjects.values())),
        "runs.csv": pd.DataFrame(runs),
        "events.csv": pd.DataFrame(ev_all),
        "cycles.csv": pd.DataFrame(cyc_meta),
        "cycle_params_long.csv": pd.DataFrame(cyc_params),
        "curves_cycles_long.csv": curves,
        "curves_run_mean_long.csv": means,
        "curves_run_mean_side_matched_long.csv": matched,
        "side_map.csv": side_map,
        "agreement_curves.csv": pd.DataFrame(agree_rows),
        "spatiotemporal.csv": pd.DataFrame(st_rows),
        "biomarkers.csv": pd.DataFrame(bio_rows),
        "sync.csv": pd.DataFrame(sync_rows),
        "agreement_synced_timeseries.csv": pd.DataFrame(sync_agree),
    }
    meta = pd.DataFrame(runs)[["pair_id"] + META]
    pmeta = pd.DataFrame(runs)[["patient", "dataset"]].drop_duplicates("patient")
    for name, df in tables.items():
        if "pair_id" in df.columns and name != "runs.csv":
            df = df.merge(meta, on="pair_id", how="left")
            df = df[["pair_id"] + META + [c for c in df.columns if c not in ["pair_id"] + META]]
        elif name == "subjects.csv" and len(df):
            df = df.merge(pmeta, on="patient", how="left")
        df.to_csv(os.path.join(OUT, name), index=False)
        log(f"wrote {name:34s} {len(df):>8d} rows")
    # sync quality flags + robust L/R mapping (rewrites the side-matched tables)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from finalise_quality import finalise
    finalise(OUT)
    log(f"\nDONE: {len(runs)} paired runs, myogait {mg.__version__}")


if __name__ == "__main__":
    main()
