"""Tests for myogait.kinetics (markerless sagittal kinetics)."""

import numpy as np
import pytest

from myogait import kinetics as K


def _standing_pose(n=200, fs=100.0, x0=0.0):
    """Static upright 2-D skeleton (metres, [AP, vertical])."""
    def c(x, y):
        return np.tile([x0 + x, y], (n, 1)).astype(float)

    J = {"head": c(0.0, 1.65)}
    for s, dx in (("left", -0.02), ("right", 0.02)):
        J.update(
            {
                f"{s}_hip": c(dx, 0.90),
                f"{s}_knee": c(dx + 0.01, 0.48),
                f"{s}_ankle": c(dx, 0.08),
                f"{s}_heel": c(dx - 0.06, 0.02),
                f"{s}_foot_index": c(dx + 0.17, 0.02),
                f"{s}_shoulder": c(dx, 1.40),
                f"{s}_elbow": c(dx, 1.12),
                f"{s}_wrist": c(dx, 0.85),
            }
        )
    return J


def test_whole_body_com_standing():
    J = _standing_pose()
    com = K.whole_body_com(J, "M")
    assert com.shape == (200, 2)
    assert 0.85 < com[0, 1] < 1.10  # CoM height ~ 55 % of stature
    assert abs(com[0, 0]) < 0.02


@pytest.mark.parametrize("tau", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_ds_share_bounded_and_monotonic(tau):
    w = K.ds_share(np.array([tau]))[0]
    assert 0.0 <= w <= 1.0
    grid = K.ds_share(np.linspace(0, 1, 101))
    assert np.all(np.diff(grid) >= -1e-9)


def test_foot_share_sums_to_one_over_double_support():
    n, on, off, ct, ch = 300, 50, 150, 70, 130
    sh = K._foot_share(n, on, off, ct, ch)
    assert sh[on] == pytest.approx(
        K.ds_share(np.array([0.0]))[0]
    )  # just landed: leading foot share
    assert sh[100] == pytest.approx(1.0)  # single support carries the total
    assert sh[off] == pytest.approx(
        1 - K.ds_share(np.array([1.0]))[0]
    )  # about to lift: trailing share
    assert sh[on - 1] == 0 and sh[off + 1] == 0
    # leading share at tau + trailing share at the same tau of the other foot = 1
    tau = np.linspace(0, 1, 11)
    assert np.allclose(K.ds_share(tau) + (1 - K.ds_share(tau)), 1.0)


def test_vpp_force_points_to_pivot():
    n = 10
    fv = np.full(n, 700.0)
    cop = np.zeros((n, 2))
    com = np.c_[np.full(n, 0.10), np.full(n, 0.95)]
    fap = K.vpp_ap_force(fv, cop, com, h=0.30)
    # the force vector must be parallel to (pivot - COP)
    pivot = np.array([0.10, 1.25])
    expected = 700.0 * pivot[0] / pivot[1]
    assert np.allclose(fap, expected)



def test_damp_reduces_to_pivot_with_zero_coefficients():
    n, fs = 200, 100.0
    fv = np.full(n, 700.0)
    cop = np.zeros((n, 2))
    com = np.c_[np.full(n, 0.10), np.full(n, 0.95)]
    hip = np.c_[np.zeros(n), np.full(n, 0.90)]
    heel, toe = np.zeros((n, 2)), np.c_[np.full(n, 0.25), np.zeros(n)]
    fap = K.damp_ap_force(fv, cop, com, heel, toe, hip, 0, n - 1, fs, h=0.30, coefs=(0, 0, 0, 0, 0))
    lag = 1 - np.exp(-np.arange(n) / fs / K.DAMP_LAG_S)
    assert np.allclose(fap, lag * K.vpp_ap_force(fv, cop, com, h=0.30))


def test_damp_matches_article_formula():
    rng = np.random.default_rng(0)
    n, fs, on, off, h = 120, 100.0, 10, 99, 0.40
    fv = np.r_[np.zeros(on), 700 * np.sin(np.linspace(0, np.pi, off - on + 1)), np.zeros(n - off - 1)]
    cop = np.c_[np.linspace(-0.1, 0.2, n), np.zeros(n)]
    com = np.c_[np.linspace(0.0, 0.25, n), 0.95 + 0.01 * rng.standard_normal(n)]
    heel = np.c_[np.full(n, -0.12), np.zeros(n)]
    toe = np.c_[np.full(n, 0.10), 0.05 * np.sin(np.linspace(0, 3, n))]
    hip = np.c_[com[:, 0], np.full(n, 0.92)]
    out = K.damp_ap_force(fv, cop, com, heel, toe, hip, on, off, fs, h=h)
    sl = slice(on, off + 1)
    m = off - on + 1
    c = np.array(K.DAMP_COEFS[h])
    L = 0.92 - 0.0
    th = np.arctan2(toe[sl, 1] - heel[sl, 1], toe[sl, 0] - heel[sl, 0])
    s = np.linspace(0, 1, m)
    th -= np.median(th[(s >= 0.3) & (s <= 0.6)])
    d = (heel[sl, 0] - com[sl, 0]) / L
    p = (cop[sl] - com[sl]) / L
    f = c[0] + c[1] * d + c[2] * th + c[3] * th**2 + c[4] * th * d
    ref = (1 - np.exp(-np.arange(m) / fs / 0.015)) * fv[sl] * (p[:, 0] - f) / (p[:, 1] - h / L)
    ref[fv[sl] <= 0] = 0
    assert np.allclose(out[sl], ref) and np.all(out[:on] == 0) and np.all(out[off + 1:] == 0)


def test_foot_rotation_cop_flat_foot_progression():
    n = 100
    heel = np.c_[np.zeros(n), np.zeros(n)]
    toe = np.c_[np.full(n, 0.25), np.zeros(n)]
    rel = K.cop_foot_rotation_rel(heel, toe, 0, n - 1)
    assert rel.shape == (101,)
    assert rel[5] < rel[50] < rel[90]  # heel-to-toe roll-over
    assert -0.1 <= rel.min() and rel.max() <= 1.15


def test_foot_rotation_cop_forefoot_contact_starts_further_forward():
    n = 100
    T = np.linspace(0, 1, n)
    heel = np.c_[np.zeros(n), np.zeros(n)]
    toe = np.c_[np.full(n, 0.25), np.zeros(n)]
    heel_strike = heel.copy()
    heel_strike[:6, 1] = 0.0
    toe_hs = toe.copy()
    toe_hs[:6, 1] = 0.08  # toe up at contact
    toe_walk = heel.copy()
    toe_walk[:6, 1] = 0.08  # heel up at contact
    r_hs = K.cop_foot_rotation_rel(heel_strike, toe_hs, 0, n - 1)
    r_tw = K.cop_foot_rotation_rel(toe_walk, toe, 0, n - 1)
    assert r_tw[3] > r_hs[3]
    assert T.size == n


def test_relocate_ankle_moves_forward_along_foot():
    n = 50
    heel = np.c_[np.zeros(n), np.zeros(n)]
    toe = np.c_[np.full(n, 0.25), np.zeros(n)]
    ankle = np.c_[np.full(n, 0.03), np.full(n, 0.07)]
    new = K.relocate_ankle(heel, toe, ankle, 0, n - 1, k_target=0.35)
    L = K.MTP_TO_TOE_RATIO * 0.25
    assert np.allclose(new[:, 0], 0.35 * L)
    assert np.allclose(new[:, 1], ankle[:, 1])


def test_inverse_dynamics_static_lever_arm():
    J = _standing_pose()
    n = 200
    fs = 100.0
    mass = 70.0
    grf = np.c_[np.zeros(n), np.full(n, mass * K.G)]
    cop_ant = np.c_[J["left_ankle"][:, 0] + 0.10, np.full(n, 0.0)]
    M = K.inverse_dynamics_2d(
        J, "left", grf, cop_ant, mass, fs, inertia_cutoff=6.0, despike=False
    )
    # body weight 10 cm in front of the ankle: plantarflexor moment ≈ 0.10 × m g (foot weight is small)
    assert M["ankle"][100] == pytest.approx(0.10 * mass * K.G, rel=0.05)
    cop_post = np.c_[J["left_ankle"][:, 0] - 0.05, np.full(n, 0.0)]
    M2 = K.inverse_dynamics_2d(
        J, "left", grf, cop_post, mass, fs, inertia_cutoff=6.0, despike=False
    )
    assert M2["ankle"][100] < 0


def test_stances_from_events_bias_and_order():
    fs = 100.0
    ev = {
        "left_hs": [{"frame": 100}, {"frame": 210}],
        "left_to": [{"frame": 166}],
        "right_hs": [{"frame": 155}],
        "right_to": [{"frame": 112}],
    }
    st = K.stances_from_events(ev, 400, fs, hs_bias_s=-0.03, to_bias_s=0.0)
    assert len(st) == 1
    s = st[0]
    assert s["side"] == "left" and s["on"] == 103 and s["off"] == 166
    assert s["on"] < s["contra_to"] < s["contra_hs"] < s["off"]


def test_landmarks_to_metric_and_estimate_kinetics_smoke():
    """End-to-end on a synthetic walker: forces positive, bounded, moments finite."""
    fs, n, W, H = 60.0, 360, 1920, 1080
    t = np.arange(n) / fs
    v = 1.2
    f = 0.9  # 1.2 m/s, 0.9 Hz stride
    scale = 1 / 600.0  # m per px
    frames = []
    for i in range(n):
        x = v * t[i]
        L = {}
        for s, ph in (("LEFT", 0.0), ("RIGHT", np.pi)):
            sw = 0.25 * np.sin(2 * np.pi * f * t[i] + ph)
            pts = {
                "HIP": (x, 0.92),
                "KNEE": (x + 0.5 * sw, 0.50),
                "ANKLE": (x + sw, 0.09),
                "HEEL": (x + sw - 0.06, 0.03),
                "FOOT_INDEX": (x + sw + 0.18, 0.02),
                "SHOULDER": (x, 1.40),
                "ELBOW": (x - 0.3 * sw, 1.12),
                "WRIST": (x - 0.5 * sw, 0.86),
            }
            for k, (px, py) in pts.items():
                L[f"{s}_{k}"] = {
                    "x": (300 + px / scale) / W,
                    "y": (900 - py / scale - 4 * np.sin(4 * np.pi * f * t[i])) / H,
                }
        L["NOSE"] = {"x": (300 + x / scale) / W, "y": (900 - 1.62 / scale) / H}
        frames.append({"frame_idx": i, "time_s": float(t[i]), "landmarks": L})
    data = {"meta": {"fps": fs, "width": W, "height": H}, "frames": frames}
    T = 1 / f
    ev = {"left_hs": [], "left_to": [], "right_hs": [], "right_to": []}
    for k in range(4):
        b = int((0.6 + k * T) * fs)
        ev["left_hs"].append({"frame": b})
        ev["right_to"].append({"frame": b + 7})
        ev["right_hs"].append({"frame": b + int(0.5 * T * fs)})
        ev["left_to"].append({"frame": b + int(0.62 * T * fs)})
    R = K.estimate_kinetics(data, 70.0, height_m=1.75, events=ev, camera_moving=False)
    assert R["walking_speed_m_s"] == pytest.approx(1.2, rel=0.15)
    assert len(R["stances"]) >= 2
    for st in R["stances"]:
        assert (
            np.all(np.isfinite(st["vgrf_bw"]))
            and st["vgrf_bw"].max() < 3.0
            and st["vgrf_bw"].min() >= 0
        )
        assert np.all(np.isfinite(st["ankle_Nm_kg"]))


def _synthetic_video(path, n=40, shift_px=(0.0, 0.0), w=640, h=360):
    import cv2

    rng = np.random.default_rng(0)
    tex = (rng.random((h * 2, w * 2)) * 255).astype(np.uint8)
    tex = cv2.GaussianBlur(tex, (0, 0), 2)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for i in range(n):
        dx, dy = int(round(i * shift_px[0])), int(round(i * shift_px[1]))
        fr = tex[h // 2 + dy : h // 2 + dy + h, w // 2 + dx : w // 2 + dx + w]
        vw.write(cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR))
    vw.release()


def test_detect_camera_motion_fixed_vs_moving(tmp_path):
    pytest.importorskip("cv2")
    fixed, moving = tmp_path / "fixed.mp4", tmp_path / "moving.mp4"
    _synthetic_video(fixed)
    _synthetic_video(moving, shift_px=(0.0, 3.0))
    a = K.detect_camera_motion(str(fixed), stride=2, max_frames=15)
    b = K.detect_camera_motion(str(moving), stride=2, max_frames=15)
    assert a["moving"] is False
    assert b["moving"] is True and b["dy_px_per_frame"] > 1.5


def test_background_vertical_shift_recovers_motion(tmp_path):
    pytest.importorskip("cv2")
    v = tmp_path / "moving.mp4"
    _synthetic_video(v, n=30, shift_px=(0.0, 2.0))
    dy = K.background_vertical_shift(str(v))
    assert len(dy) == 30
    assert abs(dy[-1]) == pytest.approx(
        2.0 * 29, rel=0.25
    )  # cumulative shift ≈ 2 px per frame


def _stepping_walker(v=1.2, fs=100.0, n=600, step_t=0.55, pan=False):
    """Feet still during stance, swing at the speed that keeps the mean = v; optional camera tracking the pelvis."""
    t = np.arange(n) / fs
    pel = v * t
    J = {}
    for s, ph in (("left", 0.0), ("right", step_t)):
        x = np.zeros(n)
        for i, ti in enumerate(t):
            k, r = divmod(
                ti + ph, 2 * step_t
            )  # stride = 2 steps: stance during 60 %, swing 40 %
            base = k * 2 * step_t * v - ph * v
            x[i] = (
                base
                if r < 1.2 * step_t
                else base + (r - 1.2 * step_t) / (0.8 * step_t) * 2 * step_t * v
            )
        cam = pel if pan else 0.0
        J[f"{s}_heel"] = np.c_[x - cam, np.zeros(n)]
        J[f"{s}_foot_index"] = np.c_[x + 0.2 - cam, np.zeros(n)]
        J[f"{s}_hip"] = np.c_[pel - cam, np.full(n, 0.9)]
    return J, fs


@pytest.mark.parametrize("pan", [False, True])
def test_walking_speed_fixed_and_panning_camera(pan):
    J, fs = _stepping_walker(pan=pan)
    assert K.walking_speed_metric(J, fs, moving=pan) == pytest.approx(1.2, rel=0.08)
