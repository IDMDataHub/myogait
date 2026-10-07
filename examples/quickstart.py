"""Quick start: gait analysis of one walking video (or a pre-extracted pivot).

    python examples/quickstart.py walk.mp4                 # MediaPipe, CPU
    python examples/quickstart.py walk.myogait.json        # no pose model needed
    python examples/quickstart.py walk.mp4 sapiens2-quick  # GPU, most accurate

Film the subject from the side, whole body visible, walking across the frame.
"""
import sys

import myogait as mg

if len(sys.argv) < 2:
    sys.exit(__doc__)
source = sys.argv[1]
model = sys.argv[2] if len(sys.argv) > 2 else "mediapipe"

result = mg.run_auto(source, model=model)
print("Recipe:", result["recipe"]["name"], "-", "; ".join(result["reasons"]))
for warning in result["quality"]["warnings"]:
    print("Warning:", warning)

cycles = result["cycles"].get("cycles", [])
print(f"{len(cycles)} gait cycles")
stats = result["stats"] or {}
st = stats.get("spatiotemporal", {})
speed = stats.get("walking_speed", {})
print("Cadence (steps/min):", st.get("cadence_steps_per_min"))
# Metric distances need the subject's height: mg.set_subject(data, height_m=...)
if speed.get("calibrated"):
    print("Walking speed (m/s):", speed.get("speed_mean"))

if cycles:
    fig = mg.plot_normative_comparison(result["data"], result["cycles"], plane="sagittal")
    fig.savefig("quickstart_kinematics.png", dpi=150)
    print("Saved quickstart_kinematics.png")
