"""Full pipeline test with Sapiens extractor — v1 (0.3B) and v2 (0.4B).

Demonstrates:
- Sapiens v1 (sapiens-quick, 0.3B) and Sapiens 2 (sapiens2-quick, 0.4B)
- Frame coherence scoring (per-frame quality metric)
- Signed knee angles (recurvatum-aware)
- Depth and segmentation auxiliary models
"""
import os
import tempfile
import time

import myogait as mg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

video = "/home/ffer/gait_benchmark/data/videos/20250909_143902.mp4"

# ── Choose model: "sapiens-quick" (v1, 0.3B) or "sapiens2-quick" (v2, 0.4B)
# Sapiens 2 (ICLR 2026) provides +4 mAP over v1 with the same Goliath 308
# keypoint format.  Requires torch>=2.7 + safetensors.
# v2 also supports sapiens2-mid (0.8B), sapiens2-top (1B), sapiens2-ultra (5B).
MODEL = "sapiens-quick"
print(f"=== PIPELINE SAPIENS ({MODEL}) - myogait v{mg.__version__} ===\n")

# 1. EXTRACTION
print(f"1. Extraction ({MODEL}, 60 frames)...")
t0 = time.time()
data = mg.extract(video, model=MODEL, max_frames=60)
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

# 2. FRAME COHERENCE SCORE
print("2. Frame coherence scoring...")
data = mg.frame_coherence_score(data)
cs = data["coherence_summary"]
print(f"   -> Mean: {cs['mean']:.3f}, Min: {cs['min']:.3f}")
print(f"   -> Low coherence frames: {cs['low_coherence_frames']}/{n} "
      f"({100*cs['low_coherence_frames']/n:.1f}%)\n")

# 3. NORMALIZATION
print("3. Normalization (butterworth)...")
data = mg.normalize(data, filters=["butterworth"])
print("   -> OK\n")

# 4. ANGLES (signed knee = recurvatum-aware)
print("4. Angle computation (signed knee angles)...")
data = mg.compute_angles(
    data, method="sagittal_vertical_axis",
    correction_factor=1.0, calibrate=False,
)
mid = n // 2
af = data["angles"]["frames"][mid]
print(f"   -> Frame {mid}: hip_L={af['hip_L']}, knee_L={af['knee_L']}, ankle_L={af['ankle_L']}")
# Signed knee: negative values indicate recurvatum (hyperextension)
knee_vals = [
    f["knee_L"] for f in data["angles"]["frames"]
    if f["knee_L"] is not None
]
if knee_vals:
    print(f"   -> Knee L range: [{min(knee_vals):.1f}, {max(knee_vals):.1f}] deg")
non_none = sum(1 for a in data["angles"]["frames"] if a["hip_L"] is not None)
print(f"   -> Valid angles: {non_none}/{n}\n")

# 5. EVENTS
print("5. Event detection (Zeni)...")
data = mg.detect_events(data, method="zeni")
ev = data["events"]
total_ev = sum(len(ev[k]) for k in ["left_hs", "right_hs", "left_to", "right_to"])
print(f"   -> HS: L={len(ev['left_hs'])}, R={len(ev['right_hs'])}")
print(f"   -> TO: L={len(ev['left_to'])}, R={len(ev['right_to'])}")
print(f"   -> Total: {total_ev}\n")

# 6. CYCLES + ANALYSIS
if total_ev >= 4:
    print("6. Cycles + Analysis...")
    try:
        cycles = mg.segment_cycles(data)
        stats = mg.analyze_gait(data, cycles)
        sp = stats["spatiotemporal"]
        print(f"   -> {len(cycles['cycles'])} cycles, "
              f"cadence={sp['cadence_steps_per_min']:.1f} steps/min")
    except Exception as e:
        print(f"   -> Not enough cycles: {e}")
        cycles, stats = None, None
else:
    print("6. Not enough events for cycles (60 frames ~ 1s)")
    cycles, stats = None, None

# 7. EXPORT
print("\n7. Export...")
with tempfile.TemporaryDirectory() as td:
    mg.export_mot(data, os.path.join(td, "gait.mot"))
    mg.export_trc(data, os.path.join(td, "markers.trc"))
    print("   -> MOT + TRC: OK")

# 8. PLOTS
print("\n8. Plot...")
fig = mg.plot_angles(data)
plt.close("all")
print("   -> plot_angles: OK")

# 9. JSON
print("\n9. JSON round-trip...")
with tempfile.TemporaryDirectory() as td:
    jp = os.path.join(td, "result.json")
    mg.save_json(data, jp)
    loaded = mg.load_json(jp)
    print(f"   -> {os.path.getsize(jp)//1024} KB, angles={loaded['angles'] is not None}")
    # Coherence score survives round-trip
    print(f"   -> Coherence preserved: {'coherence_summary' in loaded}")

t_total = time.time() - t0
print(f"\n=== SAPIENS PIPELINE COMPLETE IN {t_total:.1f}s ===")
