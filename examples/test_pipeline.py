"""Full pipeline test on a real gait video."""
import myogait as mg
import time, tempfile, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

video = "/home/ffer/gait_benchmark/data/videos/20250909_143902.mp4"
print("=== PIPELINE COMPLET myogait v" + mg.__version__ + " ===\n")

# 1. EXTRACTION
print("1. Extraction (MediaPipe)...")
t0 = time.time()
data = mg.extract(video, model="mediapipe", max_frames=300)
t1 = time.time()
n = len(data["frames"])
fps = data["meta"]["fps"]
w, h = data["meta"]["width"], data["meta"]["height"]
print(f"   -> {n} frames, {fps:.1f} fps, {w}x{h}")
print(f"   -> Direction: {data['extraction']['direction_detected']}")
print(f"   -> Temps: {t1-t0:.1f}s\n")

# 2. NORMALISATION
print("2. Normalisation (butterworth + pixel_ratio + center)...")
data = mg.normalize(data, filters=["butterworth"], pixel_ratio=True, center=True)
print(f"   -> Steps: {data['normalization']['steps_applied']}")
print(f"   -> frames_raw preserved: {'frames_raw' in data}\n")

# 3. ANGLES
print("3. Calcul des angles (sagittal_vertical_axis)...")
data = mg.compute_angles(data, method="sagittal_vertical_axis")
af = data["angles"]["frames"][150]
print(f"   -> Method: {data['angles']['method']}")
print(f"   -> Frame 150: hip_L={af['hip_L']}, knee_L={af['knee_L']}, ankle_L={af['ankle_L']}")
print(f"   -> Trunk: {af['trunk_angle']}, Pelvis: {af['pelvis_tilt']}\n")

# 4. EVENTS
print("4. Detection des events (Zeni)...")
data = mg.detect_events(data, method="zeni")
ev = data["events"]
lhs = len(ev["left_hs"])
rhs = len(ev["right_hs"])
lto = len(ev["left_to"])
rto = len(ev["right_to"])
print(f"   -> HS gauche: {lhs}, HS droit: {rhs}")
print(f"   -> TO gauche: {lto}, TO droit: {rto}")
print(f"   -> Total events: {lhs+rhs+lto+rto}\n")

# 5. CYCLES
print("5. Segmentation des cycles...")
cycles = mg.segment_cycles(data)
nc = len(cycles["cycles"])
print(f"   -> {nc} cycles detectes")
for c in cycles["cycles"][:3]:
    print(f"      Cycle {c['cycle_id']} ({c['side']}): {c['duration']:.3f}s, stance={c['stance_pct']:.1f}%")
for side in ["left", "right"]:
    if side in cycles.get("summary", {}):
        nn = cycles["summary"][side]["n_cycles"]
        print(f"   -> {side}: {nn} cycles, mean/std calculees")
print()

# 6. ANALYSE
print("6. Analyse de la marche...")
stats = mg.analyze_gait(data, cycles)
sp = stats["spatiotemporal"]
print(f"   -> Cadence: {sp['cadence_steps_per_min']:.1f} pas/min")
print(f"   -> Duree foulee: {sp['stride_time_mean_s']:.3f} +/- {sp['stride_time_std_s']:.3f} s")
spl = sp.get("stance_pct_left", 0)
spr = sp.get("stance_pct_right", 0)
print(f"   -> Stance: L={spl:.1f}%, R={spr:.1f}%")
print(f"   -> Cycles: {sp['n_cycles_total']}")
sym = stats.get("symmetry", {})
hip_si = sym.get("hip_rom_si", "N/A")
print(f"   -> Symetrie ROM hanche: {hip_si}")
var = stats.get("variability", {})
cv_val = var.get("cycle_duration_cv", "N/A")
print(f"   -> Variabilite cycle: CV={cv_val}")
flags = stats.get("flags", [])
if flags:
    print(f"   -> Drapeaux: {flags}")
print()

# 7. VALIDATION
print("7. Validation biomecanique...")
report = mg.validate_biomechanical(data, cycles)
s = report["summary"]
print(f"   -> Valide: {report['valid']}")
print(f"   -> Violations: {s['total']} (critical={s['critical']}, warning={s['warning']}, info={s['info']})\n")

# 8. EXPORT
print("8. Exports...")
with tempfile.TemporaryDirectory() as td:
    files = mg.export_csv(data, td, cycles, stats)
    print(f"   -> CSV: {len(files)} fichiers")
    mot_p = os.path.join(td, "gait.mot")
    mg.export_mot(data, mot_p)
    print(f"   -> MOT: {os.path.getsize(mot_p)} bytes")
    trc_p = os.path.join(td, "markers.trc")
    mg.export_trc(data, trc_p)
    print(f"   -> TRC: {os.path.getsize(trc_p)} bytes")
print()

# 9. PLOTS
print("9. Plots...")
fig1 = mg.plot_angles(data)
fig2 = mg.plot_events(data)
fig3 = mg.plot_cycles(cycles)
fig4 = mg.plot_summary(data, cycles, stats)
plt.close("all")
print("   -> plot_angles, plot_events, plot_cycles, plot_summary: OK\n")

# 10. REPORT PDF
print("10. Rapport PDF...")
with tempfile.TemporaryDirectory() as td:
    pdf_p = os.path.join(td, "rapport.pdf")
    mg.generate_report(data, cycles, stats, pdf_p)
    print(f"   -> PDF: {os.path.getsize(pdf_p) // 1024} KB")
    plt.close("all")
print()

# 11. JSON round-trip
print("11. JSON save/load...")
with tempfile.TemporaryDirectory() as td:
    jp = os.path.join(td, "result.json")
    mg.save_json(data, jp)
    loaded = mg.load_json(jp)
    sz = os.path.getsize(jp) // 1024
    nf = len(loaded["frames"])
    ha = loaded["angles"] is not None
    he = loaded["events"] is not None
    print(f"   -> JSON: {sz} KB, {nf} frames, angles={ha}, events={he}")
print()

t_total = time.time() - t0
print(f"=== PIPELINE COMPLET EN {t_total:.1f}s ===")
print("Tout fonctionne de bout en bout.")
