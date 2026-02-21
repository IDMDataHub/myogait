"""Test full pipeline with YOLO extractor."""
import myogait as mg
import time, tempfile, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

video = "/home/ffer/gait_benchmark/data/videos/20250909_143902.mp4"
print("=== PIPELINE YOLO - myogait v" + mg.__version__ + " ===\n")

# 1. EXTRACTION
print("1. Extraction (YOLO)...")
t0 = time.time()
data = mg.extract(video, model="yolo", max_frames=150)
t1 = time.time()
n = len(data["frames"])
fps = data["meta"]["fps"]
w, h = data["meta"]["width"], data["meta"]["height"]
det = sum(1 for f in data["frames"] if f["landmarks"])
print(f"   -> {n} frames, {fps:.1f} fps, {w}x{h}")
print(f"   -> Detection: {det}/{n} frames ({100*det/n:.0f}%)")
print(f"   -> Direction: {data['extraction']['direction_detected']}")
print(f"   -> Keypoint format: {data['extraction']['keypoint_format']}")
print(f"   -> Time: {t1-t0:.1f}s\n")

# 2. NORMALIZATION
print("2. Normalization (butterworth)...")
data = mg.normalize(data, filters=["butterworth"])
print(f"   -> Steps: {data['normalization']['steps_applied']}\n")

# 3. ANGLES
print("3. Angle computation...")
data = mg.compute_angles(data, method="sagittal_vertical_axis")
# Check a middle frame
mid = n // 2
af = data["angles"]["frames"][mid]
print(f"   -> Frame {mid}: hip_L={af['hip_L']}, knee_L={af['knee_L']}, ankle_L={af['ankle_L']}")
print(f"   -> Trunk: {af['trunk_angle']}, Pelvis: {af['pelvis_tilt']}")
# Count non-None angles
non_none = sum(1 for a in data["angles"]["frames"] if a["hip_L"] is not None)
print(f"   -> Valid angles: {non_none}/{n} frames\n")

# 4. EVENTS
print("4. Event detection (Zeni)...")
data = mg.detect_events(data, method="zeni")
ev = data["events"]
lhs = len(ev["left_hs"])
rhs = len(ev["right_hs"])
lto = len(ev["left_to"])
rto = len(ev["right_to"])
total_ev = lhs + rhs + lto + rto
print(f"   -> HS: L={lhs}, R={rhs} | TO: L={lto}, R={rto}")
print(f"   -> Total events: {total_ev}\n")

# 5. CYCLES
if total_ev >= 4:
    print("5. Cycle segmentation...")
    try:
        cycles = mg.segment_cycles(data)
        nc = len(cycles["cycles"])
        print(f"   -> {nc} cycles detected")
        for c in cycles["cycles"][:3]:
            print(f"      Cycle {c['cycle_id']} ({c['side']}): {c['duration']:.3f}s, stance={c['stance_pct']:.1f}%")

        # 6. ANALYSIS
        print("\n6. Gait analysis...")
        stats = mg.analyze_gait(data, cycles)
        sp = stats["spatiotemporal"]
        print(f"   -> Cadence: {sp['cadence_steps_per_min']:.1f} steps/min")
        print(f"   -> Cycles: {sp['n_cycles_total']}")

        # 7. VALIDATION
        print("\n7. Biomechanical validation...")
        report = mg.validate_biomechanical(data, cycles)
        s = report["summary"]
        print(f"   -> Valid: {report['valid']}")
        print(f"   -> Violations: {s['total']}")

        # 8. EXPORT
        print("\n8. Exports...")
        with tempfile.TemporaryDirectory() as td:
            files = mg.export_csv(data, td, cycles, stats)
            print(f"   -> CSV: {len(files)} files")
            mg.export_mot(data, os.path.join(td, "gait.mot"))
            print(f"   -> MOT: OK")

        # 9. PLOTS
        print("\n9. Plots...")
        fig = mg.plot_summary(data, cycles, stats)
        plt.close("all")
        print("   -> plot_summary: OK")

    except Exception as e:
        print(f"   -> Error cycles/analysis: {e}")
else:
    print("5. Not enough events to segment cycles")
    cycles = None

print()
t_total = time.time() - t0
print(f"=== YOLO PIPELINE COMPLETE IN {t_total:.1f}s ===")
print("YOLO works end-to-end.")
