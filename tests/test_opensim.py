"""Tests for myogait.opensim module.

Tests cover XML generation for OpenSim Scale Tool, Inverse Kinematics,
and MocoTrack setup files, as well as marker name mapping.
"""

import xml.etree.ElementTree as ET

import pytest

from conftest import make_walking_data
from myogait.opensim import (
    export_ik_setup,
    export_moco_setup,
    export_opensim_scale_setup,
    get_opensim_marker_names,
)


# ── Helper ────────────────────────────────────────────────────────────


def _parse_xml(path):
    """Parse an XML file and return the root element."""
    tree = ET.parse(str(path))
    return tree.getroot()


# ── export_opensim_scale_setup ────────────────────────────────────────


class TestExportOpensimScaleSetup:
    """Tests for export_opensim_scale_setup."""

    def test_creates_valid_xml(self, tmp_path):
        """The output file must be parseable XML."""
        data = make_walking_data(n_frames=60, fps=30.0)
        out = str(tmp_path / "scale_setup.xml")
        result = export_opensim_scale_setup(data, output_path=out)
        assert result == out
        # Must not raise
        root = _parse_xml(result)
        assert root is not None

    def test_contains_scale_tool(self, tmp_path):
        """The XML must contain a ScaleTool element."""
        data = make_walking_data(n_frames=60, fps=30.0)
        out = str(tmp_path / "scale.xml")
        export_opensim_scale_setup(data, output_path=out)
        root = _parse_xml(out)
        scale_tool = root.find("ScaleTool")
        assert scale_tool is not None

    def test_contains_mass_and_height(self, tmp_path):
        """ScaleTool must have mass and height elements."""
        data = make_walking_data(n_frames=60, fps=30.0)
        out = str(tmp_path / "scale.xml")
        export_opensim_scale_setup(data, output_path=out)
        root = _parse_xml(out)
        scale_tool = root.find("ScaleTool")
        mass_el = scale_tool.find("mass")
        height_el = scale_tool.find("height")
        assert mass_el is not None
        assert height_el is not None
        # Defaults when no subject info
        assert float(mass_el.text) == 75.0
        assert float(height_el.text) == 1.75

    def test_with_subject_metadata(self, tmp_path):
        """When subject metadata is present, mass/height should reflect it."""
        data = make_walking_data(n_frames=60, fps=30.0)
        data["subject"] = {"weight_kg": 82.5, "height_m": 1.83, "name": "JohnDoe"}
        out = str(tmp_path / "scale.xml")
        export_opensim_scale_setup(data, output_path=out)
        root = _parse_xml(out)
        scale_tool = root.find("ScaleTool")
        assert scale_tool.get("name") == "JohnDoe"
        assert float(scale_tool.find("mass").text) == 82.5
        assert float(scale_tool.find("height").text) == 1.83

    def test_with_meta_subject(self, tmp_path):
        """Subject metadata in data['meta']['subject'] should also work."""
        data = make_walking_data(n_frames=60, fps=30.0)
        data["meta"]["subject"] = {"weight_kg": 65.0, "height_m": 1.60}
        out = str(tmp_path / "scale.xml")
        export_opensim_scale_setup(data, output_path=out)
        root = _parse_xml(out)
        scale_tool = root.find("ScaleTool")
        assert float(scale_tool.find("mass").text) == 65.0
        assert float(scale_tool.find("height").text) == 1.60

    def test_contains_generic_model_maker(self, tmp_path):
        """GenericModelMaker must be present with model_file."""
        data = make_walking_data(n_frames=60, fps=30.0)
        out = str(tmp_path / "scale.xml")
        export_opensim_scale_setup(data, output_path=out, model_file="my_model.osim")
        root = _parse_xml(out)
        gmm = root.find(".//GenericModelMaker")
        assert gmm is not None
        mf = gmm.find("model_file")
        assert mf is not None
        assert mf.text == "my_model.osim"

    def test_contains_model_scaler(self, tmp_path):
        """ModelScaler must be present."""
        data = make_walking_data(n_frames=60, fps=30.0)
        out = str(tmp_path / "scale.xml")
        export_opensim_scale_setup(data, output_path=out)
        root = _parse_xml(out)
        ms = root.find(".//ModelScaler")
        assert ms is not None

    def test_contains_measurement_set(self, tmp_path):
        """ModelScaler must contain MeasurementSet with measurements."""
        data = make_walking_data(n_frames=60, fps=30.0)
        out = str(tmp_path / "scale.xml")
        export_opensim_scale_setup(data, output_path=out)
        root = _parse_xml(out)
        meas_set = root.find(".//MeasurementSet")
        assert meas_set is not None
        measurements = meas_set.findall(".//Measurement")
        assert len(measurements) == 3  # femur, tibia, trunk
        names = {m.get("name") for m in measurements}
        assert "femur_length" in names
        assert "tibia_length" in names
        assert "trunk_height" in names

    def test_contains_marker_placer(self, tmp_path):
        """MarkerPlacer must be present with time range."""
        data = make_walking_data(n_frames=60, fps=30.0)
        out = str(tmp_path / "scale.xml")
        export_opensim_scale_setup(data, output_path=out, static_frames=(0, 15))
        root = _parse_xml(out)
        mp = root.find(".//MarkerPlacer")
        assert mp is not None
        tr = mp.find("time_range")
        assert tr is not None
        assert "0.000000" in tr.text

    def test_output_model_in_marker_placer(self, tmp_path):
        """MarkerPlacer must reference the output model file."""
        data = make_walking_data(n_frames=60, fps=30.0)
        out = str(tmp_path / "scale.xml")
        export_opensim_scale_setup(
            data, output_path=out, output_model="my_scaled.osim"
        )
        root = _parse_xml(out)
        mp = root.find(".//MarkerPlacer")
        om = mp.find("output_model_file")
        assert om is not None
        assert om.text == "my_scaled.osim"

    def test_without_subject_defaults(self, tmp_path):
        """Without any subject metadata, defaults should be used."""
        data = make_walking_data(n_frames=60, fps=30.0)
        # Ensure no subject at all
        data["subject"] = None
        out = str(tmp_path / "scale.xml")
        export_opensim_scale_setup(data, output_path=out)
        root = _parse_xml(out)
        scale_tool = root.find("ScaleTool")
        assert float(scale_tool.find("mass").text) == 75.0
        assert float(scale_tool.find("height").text) == 1.75


# ── export_ik_setup ───────────────────────────────────────────────────


class TestExportIkSetup:
    """Tests for export_ik_setup."""

    def test_creates_valid_xml(self, tmp_path):
        """The output file must be parseable XML."""
        out = str(tmp_path / "ik_setup.xml")
        result = export_ik_setup("trial.trc", output_path=out)
        assert result == out
        root = _parse_xml(result)
        assert root is not None

    def test_contains_ik_tool(self, tmp_path):
        """The XML must contain an InverseKinematicsTool element."""
        out = str(tmp_path / "ik.xml")
        export_ik_setup("trial.trc", output_path=out)
        root = _parse_xml(out)
        ik = root.find("InverseKinematicsTool")
        assert ik is not None

    def test_model_file(self, tmp_path):
        """model_file element must match the provided model."""
        out = str(tmp_path / "ik.xml")
        export_ik_setup("trial.trc", output_path=out, model_file="custom.osim")
        root = _parse_xml(out)
        ik = root.find("InverseKinematicsTool")
        mf = ik.find("model_file")
        assert mf.text == "custom.osim"

    def test_marker_file(self, tmp_path):
        """marker_file element must match the provided trc_path."""
        out = str(tmp_path / "ik.xml")
        export_ik_setup("my_trial.trc", output_path=out)
        root = _parse_xml(out)
        ik = root.find("InverseKinematicsTool")
        mf = ik.find("marker_file")
        assert mf.text == "my_trial.trc"

    def test_time_range(self, tmp_path):
        """time_range must reflect start_time and end_time."""
        out = str(tmp_path / "ik.xml")
        export_ik_setup("trial.trc", output_path=out, start_time=0.5, end_time=2.3)
        root = _parse_xml(out)
        ik = root.find("InverseKinematicsTool")
        tr = ik.find("time_range")
        assert "0.500000" in tr.text
        assert "2.300000" in tr.text

    def test_output_motion(self, tmp_path):
        """output_motion_file must match the argument."""
        out = str(tmp_path / "ik.xml")
        export_ik_setup("trial.trc", output_path=out, output_motion="results.mot")
        root = _parse_xml(out)
        ik = root.find("InverseKinematicsTool")
        om = ik.find("output_motion_file")
        assert om.text == "results.mot"

    def test_ik_task_set(self, tmp_path):
        """IKTaskSet must contain IKMarkerTask elements."""
        out = str(tmp_path / "ik.xml")
        export_ik_setup("trial.trc", output_path=out)
        root = _parse_xml(out)
        tasks = root.findall(".//IKMarkerTask")
        assert len(tasks) > 0
        for task in tasks:
            weight = task.find("weight")
            assert weight is not None
            assert float(weight.text) == 1.0

    def test_accuracy(self, tmp_path):
        """accuracy element must be 1e-5."""
        out = str(tmp_path / "ik.xml")
        export_ik_setup("trial.trc", output_path=out)
        root = _parse_xml(out)
        ik = root.find("InverseKinematicsTool")
        acc = ik.find("accuracy")
        assert acc is not None
        assert float(acc.text) == pytest.approx(1e-5)

    def test_default_time_range(self, tmp_path):
        """Default time range should be 0.0 to 1.0."""
        out = str(tmp_path / "ik.xml")
        export_ik_setup("trial.trc", output_path=out)
        root = _parse_xml(out)
        ik = root.find("InverseKinematicsTool")
        tr = ik.find("time_range")
        parts = tr.text.strip().split()
        assert float(parts[0]) == pytest.approx(0.0)
        assert float(parts[1]) == pytest.approx(1.0)


# ── export_moco_setup ─────────────────────────────────────────────────


class TestExportMocoSetup:
    """Tests for export_moco_setup."""

    def test_creates_valid_xml(self, tmp_path):
        """The output file must be parseable XML."""
        out = str(tmp_path / "moco.xml")
        result = export_moco_setup("ik_results.mot", output_path=out)
        assert result == out
        root = _parse_xml(result)
        assert root is not None

    def test_contains_moco_study(self, tmp_path):
        """The XML must contain a MocoStudy element."""
        out = str(tmp_path / "moco.xml")
        export_moco_setup("ik_results.mot", output_path=out)
        root = _parse_xml(out)
        study = root.find("MocoStudy")
        assert study is not None

    def test_model_file(self, tmp_path):
        """model_file must match the provided path."""
        out = str(tmp_path / "moco.xml")
        export_moco_setup("ik.mot", output_path=out, model_file="my_model.osim")
        root = _parse_xml(out)
        mf = root.find(".//model_file")
        assert mf is not None
        assert mf.text == "my_model.osim"

    def test_reference_file(self, tmp_path):
        """MocoStateTrackingGoal must reference the .mot file."""
        out = str(tmp_path / "moco.xml")
        export_moco_setup("my_ik.mot", output_path=out)
        root = _parse_xml(out)
        ref = root.find(".//reference_file")
        assert ref is not None
        assert ref.text == "my_ik.mot"

    def test_has_comments(self, tmp_path):
        """The file must contain XML comments as guidance."""
        out = str(tmp_path / "moco.xml")
        export_moco_setup("ik.mot", output_path=out)
        with open(out) as f:
            content = f.read()
        assert "<!--" in content
        assert "myogait" in content

    def test_solver_settings(self, tmp_path):
        """Solver section must have mesh intervals and tolerance."""
        out = str(tmp_path / "moco.xml")
        export_moco_setup("ik.mot", output_path=out)
        root = _parse_xml(out)
        solver = root.find(".//MocoCasADiSolver")
        assert solver is not None
        mesh = solver.find("num_mesh_intervals")
        assert mesh is not None
        tol = solver.find("convergence_tolerance")
        assert tol is not None

    def test_time_bounds(self, tmp_path):
        """Time bounds should reflect start_time and end_time."""
        out = str(tmp_path / "moco.xml")
        export_moco_setup("ik.mot", output_path=out, start_time=0.2, end_time=1.5)
        root = _parse_xml(out)
        t0 = root.find(".//time_initial_bounds")
        t1 = root.find(".//time_final_bounds")
        assert t0 is not None
        assert t1 is not None
        assert float(t0.text) == pytest.approx(0.2)
        assert float(t1.text) == pytest.approx(1.5)


# ── get_opensim_marker_names ──────────────────────────────────────────


class TestGetOpensimMarkerNames:
    """Tests for get_opensim_marker_names."""

    def test_gait2392_returns_dict(self):
        """gait2392 mapping must return a non-empty dict."""
        mapping = get_opensim_marker_names("gait2392")
        assert isinstance(mapping, dict)
        assert len(mapping) > 0

    def test_gait2392_expected_keys(self):
        """gait2392 mapping must contain standard landmarks."""
        mapping = get_opensim_marker_names("gait2392")
        assert "LEFT_HIP" in mapping
        assert "RIGHT_KNEE" in mapping
        assert "LEFT_ANKLE" in mapping

    def test_gait2392_expected_values(self):
        """gait2392 mapping must map to standard OpenSim marker names."""
        mapping = get_opensim_marker_names("gait2392")
        assert mapping["RIGHT_SHOULDER"] == "R.Acromion"
        assert mapping["LEFT_HEEL"] == "L.Heel"

    def test_rajagopal2015(self):
        """rajagopal2015 model must also be supported."""
        mapping = get_opensim_marker_names("rajagopal2015")
        assert isinstance(mapping, dict)
        assert "LEFT_FOOT_INDEX" in mapping
        assert mapping["LEFT_FOOT_INDEX"] == "L.Toe.Tip"

    def test_unknown_model_raises(self):
        """An unknown model name must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown OpenSim model"):
            get_opensim_marker_names("nonexistent_model")

    def test_returns_copy(self):
        """The returned dict must be a copy, not a reference to the constant."""
        from myogait.constants import OPENSIM_MARKER_MAP
        mapping = get_opensim_marker_names("gait2392")
        mapping["NEW_KEY"] = "test"
        assert "NEW_KEY" not in OPENSIM_MARKER_MAP["gait2392"]
