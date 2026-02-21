"""Command-line interface for myogait.

Provides subcommands for video-based gait analysis:

    myogait extract video.mp4 --model mediapipe --output result.json
    myogait run video.mp4 --model mediapipe --output result.json
    myogait analyze result.json --csv --pdf --output-dir ./plots
    myogait batch *.mp4 --output-dir ./results --config config.json
    myogait download sapiens-0.3b
    myogait info result.json
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from importlib.metadata import version as pkg_version, PackageNotFoundError


def _get_version() -> str:
    """Return package version without importing the full myogait package."""
    try:
        return pkg_version("myogait")
    except PackageNotFoundError:
        # Fallback for editable/local runs where metadata may be unavailable.
        return "0.0.0+local"


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_extract(args):
    """Run extraction only."""
    from . import extract, save_json

    t0 = time.time()
    data = extract(
        args.video,
        model=args.model,
        max_frames=args.max_frames,
        with_depth=getattr(args, "with_depth", False),
        with_seg=getattr(args, "with_seg", False),
    )
    elapsed = time.time() - t0

    n = len(data["frames"])
    detected = sum(1 for f in data["frames"] if f["confidence"] > 0.3)
    print(f"Extracted {n} frames ({detected} detected) in {elapsed:.1f}s")

    output = args.output or str(Path(args.video).with_suffix(".json"))
    save_json(data, output)
    print(f"Saved to {output}")


def cmd_run(args):
    """Run full pipeline: extract → normalize → angles → events."""
    from . import extract, normalize, compute_angles, detect_events, save_json

    t0 = time.time()

    # Extract
    print(f"Extracting with {args.model}...")
    data = extract(
        args.video,
        model=args.model,
        max_frames=args.max_frames,
        with_depth=getattr(args, "with_depth", False),
        with_seg=getattr(args, "with_seg", False),
    )
    n = len(data["frames"])
    detected = sum(1 for f in data["frames"] if f["confidence"] > 0.3)
    print(f"  {n} frames, {detected} detected ({100*detected/max(n,1):.0f}%)")

    # Normalize
    if not args.no_filter:
        print(f"Normalizing (butterworth cutoff={args.cutoff}Hz)...")
        data = normalize(
            data,
            filters=["butterworth"],
            butterworth_cutoff=args.cutoff,
        )

    # Angles
    print(f"Computing angles (correction={args.correction})...")
    data = compute_angles(
        data,
        correction_factor=args.correction,
        calibrate=not args.no_calibration,
    )
    valid = sum(1 for af in data["angles"]["frames"] if af.get("knee_L") is not None)
    print(f"  {valid} frames with valid angles")

    # Events
    if not args.no_events:
        method = getattr(args, "events_method", "zeni")
        print(f"Detecting gait events ({method})...")
        data = detect_events(data, method=method)
        ev = data["events"]
        n_ev = sum(len(ev.get(k, [])) for k in ["left_hs", "right_hs", "left_to", "right_to"])
        print(f"  {n_ev} events detected")

    elapsed = time.time() - t0

    output = args.output or str(Path(args.video).with_suffix(".json"))
    save_json(data, output)
    size_kb = Path(output).stat().st_size / 1024
    print(f"Saved to {output} ({size_kb:.0f} KB) in {elapsed:.1f}s")


def cmd_analyze(args):
    """Analyze a myogait JSON file: events → cycles → stats → plots."""
    from . import load_json, save_json, detect_events, segment_cycles, analyze_gait
    from .plotting import plot_summary, plot_angles, plot_cycles, plot_events

    data = load_json(args.json_file)

    if not data.get("angles"):
        print("Error: JSON has no angles. Run the full pipeline first.")
        sys.exit(1)

    # Detect events if not present
    if not data.get("events"):
        print("Detecting gait events (Zeni)...")
        data = detect_events(data)
        # Save back
        save_json(data, args.json_file)

    ev = data["events"]
    n_ev = sum(len(ev.get(k, [])) for k in ["left_hs", "right_hs", "left_to", "right_to"])
    print(f"Events: {n_ev} ({ev.get('method', '?')})")

    # Segment cycles
    print("Segmenting gait cycles...")
    cycles_result = segment_cycles(data)
    n_cycles = len(cycles_result.get("cycles", []))
    print(f"  {n_cycles} valid cycles")

    # Statistics
    print("Computing statistics...")
    stats = analyze_gait(data, cycles_result)

    st = stats.get("spatiotemporal", {})
    print(f"  Cadence: {st.get('cadence_steps_per_min', 'N/A')} steps/min")
    print(f"  Stride: {st.get('stride_time_mean_s', 'N/A')} +/- {st.get('stride_time_std_s', 'N/A')} s")
    print(f"  Stance L: {st.get('stance_pct_left', 'N/A')}%  R: {st.get('stance_pct_right', 'N/A')}%")

    sym = stats.get("symmetry", {})
    print(f"  Symmetry overall: {sym.get('overall_si', 'N/A')}%")

    flags = stats.get("pathology_flags", [])
    if flags:
        for f in flags:
            print(f"  !! {f}")

    # Plots
    if not args.no_plots:
        import matplotlib.pyplot as plt
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating plots in {out_dir}/...")

        fig = plot_summary(data, cycles_result, stats)
        fig.savefig(out_dir / "summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig = plot_angles(data)
        fig.savefig(out_dir / "angles.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig = plot_events(data)
        fig.savefig(out_dir / "events.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        for side in ("left", "right"):
            fig = plot_cycles(cycles_result, side=side)
            fig.savefig(out_dir / f"cycles_{side}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"  Saved: summary.png, angles.png, events.png, cycles_left.png, cycles_right.png")

    # PDF report
    if args.pdf:
        from .report import generate_report
        pdf_path = str(Path(args.output_dir) / "report.pdf")
        generate_report(data, cycles_result, stats, pdf_path)
        print(f"  PDF report: {pdf_path}")

    # CSV export
    if args.csv:
        from .export import export_csv
        files = export_csv(data, args.output_dir, cycles_result, stats)
        print(f"  CSV: {len(files)} files exported")

    # OpenSim exports
    if args.mot:
        from .export import export_mot
        mot_path = str(Path(args.output_dir) / "kinematics.mot")
        export_mot(data, mot_path)
        print(f"  MOT: {mot_path}")

    if args.trc:
        from .export import export_trc
        trc_path = str(Path(args.output_dir) / "markers.trc")
        export_trc(data, trc_path)
        print(f"  TRC: {trc_path}")

    # Excel export
    if args.excel:
        from .export import export_excel
        xlsx_path = str(Path(args.output_dir) / "gait_analysis.xlsx")
        export_excel(data, xlsx_path, cycles_result, stats)
        print(f"  Excel: {xlsx_path}")

    print("Done.")


def cmd_batch(args):
    """Run full pipeline on multiple video files."""
    from . import extract, normalize, compute_angles, detect_events, save_json
    from . import segment_cycles, analyze_gait

    import glob as globmod
    import time

    # Gather input files
    files = []
    for pattern in args.inputs:
        matched = globmod.glob(pattern)
        files.extend(matched)

    files = sorted(set(files))
    if not files:
        print("No files matched.")
        sys.exit(1)

    print(f"Batch processing {len(files)} files...")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config if provided
    cfg = {}
    if args.config:
        from .config import load_config
        cfg = load_config(args.config)

    results = []
    for i, filepath in enumerate(files):
        name = Path(filepath).stem
        print(f"\n[{i+1}/{len(files)}] {name}")
        t0 = time.time()

        try:
            data = extract(
                filepath,
                model=cfg.get("extract", {}).get("model", args.model),
                max_frames=cfg.get("extract", {}).get("max_frames"),
            )

            data = normalize(
                data,
                filters=cfg.get("normalize", {}).get("filters", ["butterworth"]),
                butterworth_cutoff=cfg.get("normalize", {}).get("butterworth_cutoff", 4.0),
            )

            data = compute_angles(
                data,
                method=cfg.get("angles", {}).get("method", "sagittal_vertical_axis"),
                correction_factor=cfg.get("angles", {}).get("correction_factor", 0.8),
            )

            data = detect_events(
                data,
                method=cfg.get("events", {}).get("method", "zeni"),
            )

            json_path = out_dir / f"{name}.json"
            save_json(data, str(json_path))

            cycles = segment_cycles(data)
            stats = analyze_gait(data, cycles)

            n_ev = sum(len(data["events"].get(k, []))
                       for k in ["left_hs", "right_hs", "left_to", "right_to"])
            elapsed = time.time() - t0

            print(f"  OK: {len(data['frames'])} frames, {n_ev} events, "
                  f"{len(cycles.get('cycles', []))} cycles in {elapsed:.1f}s")

            if args.csv:
                from .export import export_csv
                export_csv(data, str(out_dir), cycles, stats, prefix=f"{name}_")

            if args.pdf:
                from .report import generate_report
                generate_report(data, cycles, stats, str(out_dir / f"{name}_report.pdf"))

            results.append({"file": name, "status": "ok", "events": n_ev,
                           "cycles": len(cycles.get("cycles", []))})

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"file": name, "status": "error", "error": str(e)})

    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nBatch complete: {ok}/{len(results)} succeeded")
    return results


def cmd_download(args):
    """Download model weights from Facebook's HuggingFace repos."""
    from .models.sapiens import download_model, _MODELS
    from .models.sapiens_depth import _DEPTH_MODELS
    from .models.sapiens_seg import _SEG_MODELS

    all_models = {}
    for size, (fn, repo) in _MODELS.items():
        all_models[f"sapiens-{size}"] = ("Pose", size, repo, "sapiens")
    for size, (fn, repo) in _DEPTH_MODELS.items():
        all_models[f"sapiens-depth-{size}"] = ("Depth", size, repo, "depth")
    for size, (fn, repo) in _SEG_MODELS.items():
        all_models[f"sapiens-seg-{size}"] = ("Seg", size, repo, "seg")

    if args.list:
        print("Available models (auto-downloaded on first use):\n")
        print(f"  {'Name':<22} {'Type':<8} {'HuggingFace repo'}")
        print(f"  {'-'*22} {'-'*8} {'-'*50}")
        for name, (mtype, size, repo, _) in sorted(all_models.items()):
            print(f"  {name:<22} {mtype:<8} {repo}")
        print(f"\nModels stored in: ~/.myogait/models/")
        return

    model_name = args.model
    if not model_name:
        print("Error: specify a model name (e.g. sapiens-0.3b). Use --list to see options.")
        sys.exit(1)

    if model_name not in all_models:
        print(f"Error: unknown model '{model_name}'. Use --list to see options.")
        sys.exit(1)

    mtype, size, repo, category = all_models[model_name]
    dest = args.dest or None

    print(f"Downloading {model_name} from {repo}...")
    if category == "sapiens":
        path = download_model(size, dest=dest)
    elif category == "depth":
        from .models.sapiens_depth import _find_depth_model
        path = _find_depth_model(size)
    elif category == "seg":
        from .models.sapiens_seg import _find_seg_model
        path = _find_seg_model(size)
    print(f"Model downloaded to: {path}")


def cmd_info(args):
    """Display info about a myogait JSON file."""
    from . import load_json

    data = load_json(args.json_file)

    meta = data.get("meta", {})
    print(f"myogait v{data.get('myogait_version', '?')}")
    print(f"Source: {meta.get('video_path', '?')}")
    print(f"FPS: {meta.get('fps', '?')}, Resolution: {meta.get('width', '?')}x{meta.get('height', '?')}")
    print(f"Frames: {meta.get('n_frames', '?')}, Duration: {meta.get('duration_s', '?')}s")

    frames = data.get("frames", [])
    detected = sum(1 for f in frames if f.get("confidence", 0) > 0.3)
    print(f"Detected: {detected}/{len(frames)} ({100*detected/len(frames):.0f}%)" if frames else "No frames")

    ext = data.get("extraction")
    if ext:
        print(f"Model: {ext.get('model', '?')} ({ext.get('model_detail', '')})")
        print(f"Direction: {ext.get('direction_detected', '?')}")

    norm = data.get("normalization")
    if norm:
        print(f"Filters: {norm.get('steps_applied', norm.get('filters', []))}")

    angles = data.get("angles")
    if angles:
        valid = sum(1 for af in angles["frames"] if af.get("knee_L") is not None)
        print(f"Angles: {valid} valid frames, method={angles.get('method', '?')}")
        print(f"  correction={angles.get('correction_factor')}, calibrated={angles.get('calibrated')}")

    events = data.get("events")
    if events:
        print(f"Events: method={events.get('method', '?')}")
        for key in ["left_hs", "right_hs", "left_to", "right_to"]:
            n = len(events.get(key, []))
            if n > 0:
                print(f"  {key}: {n} events")
    else:
        print("Events: none")


def main():
    parser = argparse.ArgumentParser(
        prog="myogait",
        description="Markerless video-based gait analysis toolkit",
    )
    parser.add_argument("--version", action="version", version=f"myogait {_get_version()}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # extract
    p_extract = sub.add_parser("extract", help="Extract pose landmarks from video")
    p_extract.add_argument("video", help="Path to video file")
    p_extract.add_argument("-m", "--model", default="mediapipe", help="Pose model (default: mediapipe)")
    p_extract.add_argument("-o", "--output", help="Output JSON path (default: video.json)")
    p_extract.add_argument("--max-frames", type=int, help="Max frames to process")
    p_extract.add_argument("--with-depth", action="store_true", help="Run Sapiens depth estimation")
    p_extract.add_argument("--with-seg", action="store_true", help="Run Sapiens body segmentation")
    p_extract.set_defaults(func=cmd_extract)

    # run (full pipeline)
    p_run = sub.add_parser("run", help="Run full pipeline: extract → normalize → angles → events")
    p_run.add_argument("video", help="Path to video file")
    p_run.add_argument("-m", "--model", default="mediapipe", help="Pose model (default: mediapipe)")
    p_run.add_argument("-o", "--output", help="Output JSON path (default: video.json)")
    p_run.add_argument("--max-frames", type=int, help="Max frames to process")
    p_run.add_argument("--cutoff", type=float, default=4.0, help="Butterworth cutoff Hz (default: 4.0)")
    p_run.add_argument("--correction", type=float, default=0.8, help="Angle correction factor (default: 0.8)")
    p_run.add_argument("--no-filter", action="store_true", help="Skip filtering")
    p_run.add_argument("--no-calibration", action="store_true", help="Skip neutral calibration")
    p_run.add_argument("--no-events", action="store_true", help="Skip event detection")
    p_run.add_argument("--events-method", default="zeni",
                       choices=["zeni", "crossing", "velocity", "oconnor"],
                       help="Event detection method (default: zeni)")
    p_run.add_argument("--with-depth", action="store_true", help="Run Sapiens depth estimation")
    p_run.add_argument("--with-seg", action="store_true", help="Run Sapiens body segmentation")
    p_run.set_defaults(func=cmd_run)

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze a myogait JSON: events → cycles → stats → plots")
    p_analyze.add_argument("json_file", help="Path to myogait JSON file (with angles)")
    p_analyze.add_argument("-o", "--output-dir", default=".", help="Directory for plots (default: .)")
    p_analyze.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    p_analyze.add_argument("--pdf", action="store_true", help="Generate PDF report")
    p_analyze.add_argument("--csv", action="store_true", help="Export CSV files")
    p_analyze.add_argument("--mot", action="store_true", help="Export OpenSim .mot file")
    p_analyze.add_argument("--trc", action="store_true", help="Export OpenSim .trc file")
    p_analyze.add_argument("--excel", action="store_true", help="Export Excel workbook")
    p_analyze.set_defaults(func=cmd_analyze)

    # batch
    p_batch = sub.add_parser("batch", help="Run pipeline on multiple video files")
    p_batch.add_argument("inputs", nargs="+", help="Video file paths or glob patterns")
    p_batch.add_argument("-m", "--model", default="mediapipe", help="Pose model (default: mediapipe)")
    p_batch.add_argument("-o", "--output-dir", default="./batch_output", help="Output directory")
    p_batch.add_argument("--config", help="Config file (JSON/YAML)")
    p_batch.add_argument("--csv", action="store_true", help="Also export CSV files")
    p_batch.add_argument("--pdf", action="store_true", help="Also generate PDF reports")
    p_batch.set_defaults(func=cmd_batch)

    # download
    p_dl = sub.add_parser("download", help="Download model weights from HuggingFace Hub")
    p_dl.add_argument("model", nargs="?", default="", help="Model to download (e.g. sapiens-0.3b)")
    p_dl.add_argument("--list", action="store_true", help="List available models")
    p_dl.add_argument("--dest", help="Destination directory (default: ~/.myogait/models/)")
    p_dl.set_defaults(func=cmd_download)

    # info
    p_info = sub.add_parser("info", help="Show info about a myogait JSON file")
    p_info.add_argument("json_file", help="Path to myogait JSON file")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
