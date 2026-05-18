import run_pipeline


def test_run_pipeline_demo_prints_interpolated_frames(capsys) -> None:
    run_pipeline.main()

    output = capsys.readouterr().out
    frame_lines = [line for line in output.splitlines() if line.startswith("Frame ")]
    assert len(frame_lines) > 1
    assert all(
        line.startswith(f"Frame {index}:") for index, line in enumerate(frame_lines)
    )
