# Provenance script for the myogait 0.9.3 kinematic results in validation/README.md.
# It ran on local copies of BioCV (paths below are the authors' machine) and on the
# pairing tables of the full validation package; it is kept for traceability, not
# as a turnkey script.
"""Proof figure: marker reference (computed WITHOUT myogait) vs myogait video angles
with current defaults (Butterworth 4 Hz, correction_factor 0.8) and proposed
(8 Hz, 1.0), BioCV cam01, 9 healthy subjects. Hip / knee / ankle."""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "D:/projets/myogait_hip_pelvis/scripts")
sys.path.insert(0, "//wsl.localhost/Ubuntu/home/ffer/myogait")
import myogait as mg
from common import load_markers, marker_angles, curve101, c3d_path, sapiens_path, RES, SUBJECTS
from a_common import marker_canon, incl, pitch, rolling_sign

meta = pd.read_csv(RES / "cycles_meta_cam01.csv")
trials = pd.read_csv(RES / "trials_cam01.csv")
for c in ("mk_dir_ok", "vd_ok", "id_swap", "vd_dir_ok"):
    meta[c] = meta[c].astype(str).str.lower().eq("true")
meta = meta[meta.mk_dir_ok & meta.vd_ok & ~meta.id_swap & meta.vd_dir_ok]
trials["swap"] = trials["swap"].astype(str).str.lower().eq("true")

VARIANTS = {"proposed": (8.0, 1.0)}


def video_angles_myogait(path, fc, cf):
    data = mg.load_json(str(path))
    data = mg.normalize(data, filters=["butterworth"], butterworth_cutoff=fc)
    data = mg.compute_angles(data, calibrate=False, correction_factor=cf)
    data = mg.canonicalize_angle_signs(data)
    tmap = {f["frame_idx"]: f.get("time_s") for f in data["frames"]}
    fps = float(data["meta"]["fps"])
    af = data["angles"]["frames"]
    t = np.array([tmap.get(a["frame_idx"], a["frame_idx"] / fps) for a in af], float)
    out = {"t": t}
    for k in ("hip_L", "hip_R", "knee_L", "knee_R", "ankle_L", "ankle_R"):
        out[k] = np.array([np.nan if a.get(k) is None else a[k] for a in af], float)
    return out


rows = []  # subject, joint, system, curve
for (subj, trial), cyc in meta.groupby(["subject", "trial"]):
    tr = trials[(trials.subject == subj) & (trials.trial == trial)].iloc[0]
    cp = c3d_path(subj, trial)
    vp = sapiens_path(subj, trial, "cam01")
    if cp is None or vp is None or not vp.exists():
        continue
    m, fs = load_markers(cp, fc=15.0)
    ma = marker_angles(m, fs)
    mc = marker_canon(m)
    tm = np.arange(len(ma["dir"])) / fs
    d = ma["dir"]
    mk = {}
    fw = lambda p: p[:, 1]
    up = lambda p: p[:, 2]
    for s in "LR":
        mk[f"hip_{s}"] = ma[f"hip3d_{s}"]
        mk[f"knee_{s}"] = ma[f"knee_{s}"]
        k, a, h, to = (mc[f"{s}_{q}"] for q in ("KJC", "AJC", "HEEL", "TOE"))
        dd = rolling_sign(fw(to) - fw(h), int(fs))
        mk[f"ankle_{s}"] = incl(dd * (fw(k) - fw(a)), up(k) - up(a)) + pitch(dd * (fw(to) - fw(h)), up(to) - up(h))
    vids = {v: video_angles_myogait(vp, *p) for v, p in VARIANTS.items()}
    lag = float(tr.lag) if np.isfinite(tr.lag) else 0.0
    for _, c in cyc.iterrows():
        s = c.side
        vs = ({"L": "R", "R": "L"}[s]) if bool(tr.swap) else s
        for j in ("hip", "knee", "ankle"):
            rows.append((subj, j, "marker", curve101(tm, mk[f"{j}_{s}"], c.t0, c.t1)))
            for v, va in vids.items():
                rows.append((subj, j, v, curve101(va["t"] - lag, va[f"{j}_{vs}"], c.t0 + 0, c.t1)))
    print(subj, trial, len(cyc), flush=True)

df = pd.DataFrame(rows, columns=["subject", "joint", "system", "curve"])
# subject means
S = {}
for (subj, j, sy), g in df.groupby(["subject", "joint", "system"]):
    S[(subj, j, sy)] = np.nanmean(np.vstack(g.curve.values), 0)
subs = sorted({k[0] for k in S})

# ankle sign check vs marker (video convention: dorsiflexion +)
for j in ("hip", "knee", "ankle"):
    r = np.nanmean([np.corrcoef(S[(s, j, "marker")], S[(s, j, "proposed")])[0, 1] for s in subs])
    if r < 0:
        for s in subs:
            S[(s, j, "marker")] = -S[(s, j, "marker")]
        print("flipped marker sign for", j)

met = []
for j in ("hip", "knee", "ankle"):
    for v in VARIANTS:
        for s in subs:
            a, b = S[(s, j, v)], S[(s, j, "marker")]
            ok = np.isfinite(a) & np.isfinite(b)
            a0, b0 = a - np.nanmean(a[ok]), b - np.nanmean(b[ok])
            seg = slice(50, 76)
            met.append(dict(joint=j, variant=v, subject=s,
                            r=np.corrcoef(a[ok], b[ok])[0, 1],
                            crmse=np.sqrt(np.nanmean((a0 - b0)[ok] ** 2)),
                            crmse_50_75=np.sqrt(np.nanmean((a0[seg] - b0[seg]) ** 2)),
                            rom_err=(np.nanmax(a) - np.nanmin(a)) - (np.nanmax(b) - np.nanmin(b)),
                            offset=np.nanmean(a[ok] - b[ok]),
                            min_err=np.nanmin(a0) - np.nanmin(b0),
                            max_err=np.nanmax(a0) - np.nanmax(b0)))
met = pd.DataFrame(met)
summ = met.groupby(["joint", "variant"]).mean(numeric_only=True).round(2)
print(summ)
summ.to_csv("D:/projets/myogait_hip_pelvis/results/doc_metrics.csv")
met.to_csv("D:/projets/myogait_hip_pelvis/results/doc_metrics_per_subject.csv", index=False)
for j in ("hip", "knee", "ankle"):
    absr = []
    for s_ in subs:
        a, b = S[(s_, j, "proposed")], S[(s_, j, "marker")]
        absr.append(np.sqrt(np.nanmean((a - b) ** 2)))
    print(j, "absolute RMSE", round(float(np.mean(absr)), 2), "n subjects", len(subs))

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
x = np.arange(101)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
tit = {"hip": "Hip flexion", "knee": "Knee flexion", "ankle": "Ankle dorsiflexion"}
for ax, j in zip(axes, ("hip", "knee", "ankle")):
    for sy, col, lab in (("marker", "#222222", "Marker-based capture (Qualisys)"),
                         ("proposed", "#1f6fb4", "myogait, single video")):
        C = np.vstack([S[(s_, j, sy)] for s_ in subs])
        mu, sd = np.nanmean(C, 0), np.nanstd(C, 0)
        ax.plot(x, mu, color=col, lw=2, label=lab)
        ax.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
    r = summ.loc[(j, "proposed")]
    ax.text(0.02, 0.03, f"r {r.r:.2f} · shape RMSE {r.crmse:.1f}° · offset {r.offset:+.1f}°",
            transform=ax.transAxes, fontsize=8, color="#333333")
    ax.set_title(tit[j], fontsize=10)
    ax.set_xlabel("Gait cycle (%)", fontsize=9)
    ax.axhline(0, color="#999999", lw=0.5)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Angle (°)", fontsize=9)
axes[0].legend(fontsize=8, frameon=False, loc="upper right")
fig.suptitle(f"BioCV laboratory dataset, left lateral camera: {len(subs)} healthy adults, mean ± SD across subjects (myogait 0.9.3 defaults)", fontsize=9.5)
fig.tight_layout()
fig.savefig("D:/projets/myogait_hip_pelvis/results/doc_validation_curves.png", dpi=200)
print("saved")
