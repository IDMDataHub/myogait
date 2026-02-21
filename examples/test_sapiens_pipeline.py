"""Test full pipeline with Sapiens 0.3B extractor on CPU."""
import myogait as mg
import time, tempfile, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

video = "/home/ffer/gait_benchmark/data/videos/20250909_143902.mp4"
print("=== PIPELINE SAPIENS 0.3B (CPU) - myogait v" + mg.__version__ + " ===\n")

# 1. EXTRACTION
print("1. Extraction (Sapiens 0.3B, CPU, 60 frames)...")
t0 = time.time()
data = mg.extract(video, model="sapiens-quick", max_frames=60)
t1 = time.time()
n = len(data["frames"])
fps = data["meta"]["fps"]
w, h = data["meta"]["width"], data["meta"]["height"]
det = sum(1 for f in data["frames"] if f["landmarks"])
print(f"   -> {n} frames, {fps:.1f} fps, {w}x{h}")
print(f"   -> Detection: {det}/{n} frames ({100*det/n:.0f}%)")
print(f"   -> Direction: {data['extraction']['direction_detected']}")
print(f"   -> Format: {data['extraction']['keypoint_format']}")
print(f"   -> Time: {t1-t0:.1f}s ({(t1-t0)/n:.2f}s/frame)\n")

# 2. NORMALIZATION
print("2. Normalization (butterworth)...")
data = mg.normalize(data, filters=["butterworth"])
print(f"   -> OK\n")

# 3. ANGLES
print("3. Angle computation...")
data = mg.compute_angles(data, method="sagittal_vertical_axis", correction_factor=1.0, calibrate=False)
mid = n // 2
af = data["angles"]["frames"][mid]
print(f"   -> Frame {mid}: hip_L={af['hip_L']}, knee_L={af['knee_L']}, ankle_L={af['ankle_L']}")
non_none = sum(1 for a in data["angles"]["frames"] if a["hip_L"] is not None)
print(f"   -> Valid angles: {non_none}/{n}\n")

# 4. EVENTS
print("4. Event detection (Zeni)...")
data = mg.detect_events(data, method="zeni")
ev = data["events"]
total_ev = sum(len(ev[k]) for k in ["left_hs", "right_hs", "left_to", "right_to"])
print(f"   -> HS: L={len(ev['left_hs'])}, R={len(ev['right_hs'])}")
print(f"   -> TO: L={len(ev['left_to'])}, R={len(ev['right_to'])}")
print(f"   -> Total: {total_ev}\n")

# 5. CYCLES + ANALYSIS
if total_ev >= 4:
    print("5. Cycles + Analysis...")
    try:
        cycles = mg.segment_cycles(data)
        stats = mg.analyze_gait(data, cycles)
        sp = stats["spatiotemporal"]
        print(f"   -> {len(cycles['cycles'])} cycles, cadence={sp['cadence_steps_per_min']:.1f} steps/min")
    except Exception as e:
        print(f"   -> Not enough cycles: {e}")
        cycles, stats = None, None
else:
    print("5. Not enough events for cycles (60 frames = ~1s)")
    cycles, stats = None, None

# 6. EXPORT
print("\n6. Export...")
with tempfile.TemporaryDirectory() as td:
    mg.export_mot(data, os.path.join(td, "gait.mot"))
    mg.export_trc(data, os.path.join(td, "markers.trc"))
    print(f"   -> MOT + TRC: OK")

# 7. PLOTS
print("\n7. Plot...")
fig = mg.plot_angles(data)
plt.close("all")
print("   -> plot_angles: OK")

# 8. JSON
print("\n8. JSON round-trip...")
with tempfile.TemporaryDirectory() as td:
    jp = os.path.join(td, "result.json")
    mg.save_json(data, jp)
    loaded = mg.load_json(jp)
    print(f"   -> {os.path.getsize(jp)//1024} KB, angles={loaded['angles'] is not None}")

t_total = time.time() - t0
print(f"\n=== SAPIENS COMPLETE IN {t_total:.1f}s ===")
print("Sapiens works end-to-end (without sapiens_inference).")
