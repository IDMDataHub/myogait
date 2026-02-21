"""Tests for OpenSim .trc and .mot export improvements.

Covers:
- Incremental X/Y/Z sub-headers in .trc
- Height-based unit conversion in .trc
- Normalized fallback when height_m is absent
- Depth support (use_depth) in .trc
- OpenSim marker renaming (opensim_model) in .trc
- Pelvis translations (pelvis_tx) in .mot
- Extended angles in .mot when present
"""

import pytest
from pathlib import Path

from conftest import make_walking_data, make_walking_data_with_depth, run_full_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_subject_height(data, height_m=1.75):
    """Attach height_m to data so that _compute_height_scale succeeds."""
    if "subject" not in data or data["subject"] is None:
        data["subject"] = {}
    data["subject"]["height_m"] = height_m
    # Also put it in meta.subject for the lookup path used by _compute_height_scale
    data.setdefault("meta", {}).setdefault("subject", {})["height_m"] = height_m


def _add_extended_angles(data):
    """Inject synthetic extended angle keys into angle frames."""
    if data.get("angles") and data["angles"].get("frames"):
        for af in data["angles"]["frames"]:
            af["shoulder_flex_L"] = 15.0
            af["shoulder_flex_R"] = 14.0
            af["elbow_flex_L"] = 45.0
            af["elbow_flex_R"] = 44.0
            af["head_angle"] = 5.0


def _add_frontal_angles(data):
    """Inject synthetic frontal angle keys into angle frames."""
    if data.get("angles") and data["angles"].get("frames"):
        for af in data["angles"]["frames"]:
            af["pelvis_list"] = 2.0
            af["hip_adduction_L"] = 3.5
            af["hip_adduction_R"] = 3.3


# TRC file line layout (0-indexed):
#   0: PathFileType   4   (X/Y/Z)   filename.trc
#   1: DataRate  CameraRate  NumFrames  NumMarkers  Units  ...
#   2: 30.00    30.00    N    12    m    ...       (values)
#   3: Frame#  Time  MARKER1      MARKER2      ...  (marker names)
#   4:              X1  Y1  Z1   X2  Y2  Z2   ...  (XYZ sub-headers)
#   5: (blank line)
#   6: 1  0.000000  ...                              (first data row)

_LINE_UNITS_VALUES = 2     # units value is in field index 4 of line 2
_LINE_MARKER_NAMES = 3     # marker name row
_LINE_XYZ_SUBHEADER = 4    # XYZ sub-header row
_LINE_FIRST_DATA = 6       # first data row (after blank line 5)


# ===========================================================================
# TRC tests
# ===========================================================================


class TestTrcHeaders:
    """Verify .trc column sub-headers are incrementally numbered."""

    def test_xyz_headers_incremental(self, tmp_path):
        """X/Y/Z sub-headers should be X1,Y1,Z1, X2,Y2,Z2, etc."""
        from myogait.export import export_trc
        data = make_walking_data(30)
        _add_subject_height(data)
        trc_path = str(tmp_path / "test.trc")
        export_trc(data, trc_path)

        content = Path(trc_path).read_text()
        lines = content.split("\n")
        sub_header_line = lines[_LINE_XYZ_SUBHEADER]
        parts = sub_header_line.split("\t")
        # First two are empty (Frame# and Time placeholders)
        xyz_parts = parts[2:]
        n_markers = len(xyz_parts) // 3
        assert n_markers >= 1
        for i in range(n_markers):
            idx = i + 1
            assert xyz_parts[i * 3] == f"X{idx}", f"Expected X{idx}, got {xyz_parts[i * 3]}"
            assert xyz_parts[i * 3 + 1] == f"Y{idx}", f"Expected Y{idx}, got {xyz_parts[i * 3 + 1]}"
            assert xyz_parts[i * 3 + 2] == f"Z{idx}", f"Expected Z{idx}, got {xyz_parts[i * 3 + 2]}"

    def test_no_repeated_x1y1z1(self, tmp_path):
        """The old bug repeated X1,Y1,Z1 for every marker. Verify it is fixed."""
        from myogait.export import export_trc
        data = make_walking_data(10)
        _add_subject_height(data)
        trc_path = str(tmp_path / "test.trc")
        export_trc(data, trc_path)

        content = Path(trc_path).read_text()
        lines = content.split("\n")
        sub_header_line = lines[_LINE_XYZ_SUBHEADER]
        # Count occurrences of "X1" -- should be exactly 1
        assert sub_header_line.split("\t").count("X1") == 1


class TestTrcHeightConversion:
    """Verify height_m-based unit conversion."""

    def test_with_height_m_coordinates_not_normalized(self, tmp_path):
        """With height_m set, coordinates should be converted out of 0-1 range."""
        from myogait.export import export_trc
        data = make_walking_data(30)
        _add_subject_height(data, height_m=1.75)
        trc_path = str(tmp_path / "scaled.trc")
        export_trc(data, trc_path)

        content = Path(trc_path).read_text()
        lines = content.split("\n")
        data_line = lines[_LINE_FIRST_DATA]
        parts = data_line.split("\t")
        # First data coordinate is at index 2 (X of first marker)
        x_val = float(parts[2])
        # Walking data: x ~ 0.5 normalized, scaled should be > 1.0
        assert x_val > 1.0, f"Expected converted coordinate > 1.0, got {x_val}"

    def test_without_height_m_units_normalized(self, tmp_path):
        """Without height_m, units header should say 'normalized'."""
        from myogait.export import export_trc
        data = make_walking_data(10)
        # Ensure no height_m anywhere
        data.get("meta", {}).pop("subject", None)
        if data.get("subject") is not None:
            data.pop("subject", None)
        trc_path = str(tmp_path / "norm.trc")
        export_trc(data, trc_path)

        content = Path(trc_path).read_text()
        lines = content.split("\n")
        values_line = lines[_LINE_UNITS_VALUES]
        parts = values_line.split("\t")
        # Units is the 5th field (index 4)
        assert parts[4] == "normalized", f"Expected 'normalized', got {parts[4]}"

    def test_without_height_m_coordinates_between_0_and_1(self, tmp_path):
        """Without height_m, coordinates should remain in [0, 1] range."""
        from myogait.export import export_trc
        data = make_walking_data(10)
        data.get("meta", {}).pop("subject", None)
        if data.get("subject") is not None:
            data.pop("subject", None)
        trc_path = str(tmp_path / "norm.trc")
        export_trc(data, trc_path)

        content = Path(trc_path).read_text()
        lines = content.split("\n")
        data_line = lines[_LINE_FIRST_DATA]
        parts = data_line.split("\t")
        x_val = float(parts[2])
        y_val = float(parts[3])
        assert 0.0 <= x_val <= 1.0, f"Expected normalized x in [0,1], got {x_val}"
        assert 0.0 <= y_val <= 1.0, f"Expected normalized y in [0,1], got {y_val}"


class TestTrcDepth:
    """Verify use_depth support in .trc export."""

    def test_use_depth_true_with_depth_data(self, tmp_path):
        """With use_depth=True and depth data, Z should be non-zero."""
        from myogait.export import export_trc
        data = make_walking_data_with_depth(30)
        _add_subject_height(data, height_m=1.75)
        trc_path = str(tmp_path / "depth.trc")
        export_trc(data, trc_path, use_depth=True)

        content = Path(trc_path).read_text()
        lines = content.split("\n")
        data_line = lines[_LINE_FIRST_DATA]
        parts = data_line.split("\t")
        # Z of first marker is at index 4 (Frame#=0, Time=1, X=2, Y=3, Z=4)
        z_val = float(parts[4])
        assert z_val != 0.0, f"Expected non-zero Z with depth data, got {z_val}"

    def test_use_depth_false_z_is_zero(self, tmp_path):
        """With use_depth=False (default), Z should be 0.0 even if depth data exists."""
        from myogait.export import export_trc
        data = make_walking_data_with_depth(10)
        _add_subject_height(data, height_m=1.75)
        trc_path = str(tmp_path / "no_depth.trc")
        export_trc(data, trc_path, use_depth=False)

        content = Path(trc_path).read_text()
        lines = content.split("\n")
        data_line = lines[_LINE_FIRST_DATA]
        parts = data_line.split("\t")
        z_val = float(parts[4])
        assert z_val == 0.0, f"Expected Z=0.0 without use_depth, got {z_val}"

    def test_depth_scale_multiplier(self, tmp_path):
        """depth_scale should multiply the raw depth value."""
        from myogait.export import export_trc
        data = make_walking_data_with_depth(10)
        _add_subject_height(data, height_m=1.75)

        trc_path1 = str(tmp_path / "depth_s1.trc")
        export_trc(data, trc_path1, use_depth=True, depth_scale=1.0)
        trc_path2 = str(tmp_path / "depth_s2.trc")
        export_trc(data, trc_path2, use_depth=True, depth_scale=2.0)

        line1 = Path(trc_path1).read_text().split("\n")[_LINE_FIRST_DATA]
        line2 = Path(trc_path2).read_text().split("\n")[_LINE_FIRST_DATA]
        z1 = float(line1.split("\t")[4])
        z2 = float(line2.split("\t")[4])
        assert z2 == pytest.approx(z1 * 2.0, rel=1e-4), (
            f"Expected Z with scale=2.0 to be ~2x Z with scale=1.0 ({z1} vs {z2})"
        )


class TestTrcOpensimModel:
    """Verify opensim_model marker renaming in .trc."""

    def test_gait2392_renames_markers(self, tmp_path):
        """opensim_model='gait2392' should rename markers in header."""
        from myogait.export import export_trc
        data = make_walking_data(10)
        _add_subject_height(data, height_m=1.75)
        trc_path = str(tmp_path / "osim.trc")
        export_trc(data, trc_path, opensim_model="gait2392")

        content = Path(trc_path).read_text()
        lines = content.split("\n")
        header1_line = lines[_LINE_MARKER_NAMES]
        # Expect renamed markers
        assert "R.ASIS" in header1_line, "Expected R.ASIS for RIGHT_HIP"
        assert "L.Knee.Lat" in header1_line, "Expected L.Knee.Lat for LEFT_KNEE"
        # Original names should NOT appear
        assert "RIGHT_HIP" not in header1_line

    def test_unknown_opensim_model_raises(self, tmp_path):
        """Unknown opensim_model should raise ValueError."""
        from myogait.export import export_trc
        data = make_walking_data(5)
        trc_path = str(tmp_path / "bad.trc")
        with pytest.raises(ValueError, match="Unknown opensim_model"):
            export_trc(data, trc_path, opensim_model="nonexistent_model")


# ===========================================================================
# MOT tests
# ===========================================================================


class TestMotPelvisTranslations:
    """Verify pelvis_tx/ty/tz columns in .mot output."""

    def test_pelvis_tx_present(self, tmp_path):
        """pelvis_tx should appear in the .mot column header."""
        from myogait.export import export_mot
        data, _, _ = run_full_pipeline()
        mot_path = str(tmp_path / "gait.mot")
        export_mot(data, mot_path)

        content = Path(mot_path).read_text()
        assert "pelvis_tx" in content
        assert "pelvis_ty" in content
        assert "pelvis_tz" in content

    def test_pelvis_tx_values_nonzero(self, tmp_path):
        """pelvis_tx values should be non-zero (hip midpoint is not at origin)."""
        from myogait.export import export_mot
        data, _, _ = run_full_pipeline()
        mot_path = str(tmp_path / "gait.mot")
        export_mot(data, mot_path)

        content = Path(mot_path).read_text()
        lines = content.strip().split("\n")
        # Find the column header line (first line after 'endheader')
        header_idx = None
        for i, line in enumerate(lines):
            if line.strip() == "endheader":
                header_idx = i + 1
                break
        assert header_idx is not None
        col_names = lines[header_idx].split("\t")
        tx_col_idx = col_names.index("pelvis_tx")

        # Check first data row
        data_line = lines[header_idx + 1]
        tx_val = float(data_line.split("\t")[tx_col_idx])
        assert tx_val != 0.0, "pelvis_tx should not be zero for walking data"


class TestMotExtendedAngles:
    """Verify extended angles are added to .mot when present."""

    def test_extended_angles_present_when_data_has_them(self, tmp_path):
        """Extended angles (arm_flex_l, etc.) should appear in .mot if present in data."""
        from myogait.export import export_mot
        data, _, _ = run_full_pipeline()
        _add_extended_angles(data)
        mot_path = str(tmp_path / "extended.mot")
        export_mot(data, mot_path)

        content = Path(mot_path).read_text()
        assert "arm_flex_l" in content
        assert "arm_flex_r" in content
        assert "elbow_flexion_l" in content
        assert "head_flexion" in content

    def test_extended_angles_absent_when_data_lacks_them(self, tmp_path):
        """Extended angles should NOT appear if not in data."""
        from myogait.export import export_mot
        data, _, _ = run_full_pipeline()
        mot_path = str(tmp_path / "basic.mot")
        export_mot(data, mot_path)

        content = Path(mot_path).read_text()
        assert "arm_flex_l" not in content
        assert "elbow_flexion_l" not in content

    def test_frontal_angles_present_when_data_has_them(self, tmp_path):
        """Frontal angles (pelvis_list, hip_adduction) should appear if present."""
        from myogait.export import export_mot
        data, _, _ = run_full_pipeline()
        _add_frontal_angles(data)
        mot_path = str(tmp_path / "frontal.mot")
        export_mot(data, mot_path)

        content = Path(mot_path).read_text()
        assert "pelvis_list" in content
        assert "hip_adduction_l" in content
        assert "hip_adduction_r" in content


class TestMotBackwardCompatibility:
    """Ensure the updated export_mot still passes all existing checks."""

    def test_mot_still_has_endheader(self, tmp_path):
        """The .mot should still have the standard OpenSim header."""
        from myogait.export import export_mot
        data, _, _ = run_full_pipeline()
        mot_path = str(tmp_path / "gait.mot")
        export_mot(data, mot_path)

        content = Path(mot_path).read_text()
        assert "endheader" in content
        assert "hip_flexion_l" in content

    def test_mot_row_count(self, tmp_path):
        """Number of data rows should match number of angle frames."""
        from myogait.export import export_mot
        data, _, _ = run_full_pipeline()
        mot_path = str(tmp_path / "gait.mot")
        export_mot(data, mot_path)

        content = Path(mot_path).read_text()
        lines = content.strip().split("\n")
        # Find endheader
        for i, line in enumerate(lines):
            if line.strip() == "endheader":
                # Column header is next line, then data
                n_data_lines = len(lines) - i - 2
                assert n_data_lines == len(data["angles"]["frames"])
                break
