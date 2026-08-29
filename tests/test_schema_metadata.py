"""Subject/study metadata persistence: set_subject segments + set_study.

These are the library edit primitives that let a pivot carry its full
anthropometry and study identifiers, so metadata round-trips between the
``myogait`` API and the app instead of living only in a session config.
"""
import json

import myogait as mg
from myogait.schema import (
    SUBJECT_SEGMENT_FIELDS,
    load_json,
    save_json,
    set_study,
    set_subject,
)


def _empty():
    from myogait.schema import create_empty
    return create_empty("t.mp4", fps=30.0, width=1920, height=1080, n_frames=1)


def test_set_subject_persists_measured_segments():
    data = set_subject(_empty(), height_m=1.75, femur_length_mm=420.0,
                       tibia_length_mm=400.0, foot_length_mm=250.0)
    s = data["subject"]
    assert s["height_m"] == 1.75
    assert s["femur_length_mm"] == 420.0
    assert s["tibia_length_mm"] == 400.0
    assert s["foot_length_mm"] == 250.0
    # Unset segments stay absent, not written as None.
    assert "upper_arm_length_mm" not in s


def test_all_segment_fields_are_accepted():
    kwargs = {name: float(i + 1) for i, name in enumerate(SUBJECT_SEGMENT_FIELDS)}
    s = set_subject(_empty(), **kwargs)["subject"]
    for name, value in kwargs.items():
        assert s[name] == value


def test_set_subject_keeps_demographic_fields():
    s = set_subject(_empty(), age=34, sex="F", pathology="CMT",
                    femur_length_mm=410.0)["subject"]
    assert s["age"] == 34 and s["sex"] == "F" and s["pathology"] == "CMT"
    assert s["femur_length_mm"] == 410.0


def test_set_study_writes_only_given_keys():
    data = set_study(_empty(), patient_id="P07", condition="post-op")
    st = data["study"]
    assert st == {"patient_id": "P07", "condition": "post-op"}


def test_set_study_merges_and_preserves_other_fields():
    data = _empty()
    set_study(data, patient_id="P07", run="WALK_01", group="dmd",
              condition="baseline")
    # Re-tag only the condition; everything else must survive.
    set_study(data, condition="post-op")
    assert data["study"] == {
        "patient_id": "P07", "run": "WALK_01", "group": "dmd",
        "condition": "post-op",
    }


def test_metadata_round_trips_through_save_load(tmp_path):
    data = _empty()
    set_subject(data, height_m=1.68, age=52, femur_length_mm=415.0,
                tibia_length_mm=395.0)
    set_study(data, patient_id="P02", run="r1", condition="pre")
    p = tmp_path / "trial.json"
    save_json(data, str(p))
    reloaded = load_json(str(p))
    assert reloaded["subject"]["femur_length_mm"] == 415.0
    assert reloaded["subject"]["age"] == 52
    assert reloaded["study"]["condition"] == "pre"
    # And the on-disk JSON is plain, machine-editable text.
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert raw["study"]["patient_id"] == "P02"


def test_old_pivot_without_metadata_still_loads(tmp_path):
    data = _empty()  # no subject / study set
    p = tmp_path / "bare.json"
    save_json(data, str(p))
    reloaded = load_json(str(p))
    assert "frames" in reloaded and "meta" in reloaded


def test_exported_at_top_level_of_package():
    assert mg.set_study is set_study
    assert callable(mg.set_subject)
