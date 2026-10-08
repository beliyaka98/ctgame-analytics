"""End-to-end tests of the command-line interface and the HTML report."""

import json
import subprocess
import sys

import pandas as pd
import pytest

from ctgame.cli import main
from ctgame.report import build_report, load_study


@pytest.fixture
def data_dir(tmp_path):
    folder = tmp_path / "data"
    assert main(["simulate", "--out", str(folder), "--students", "60", "--seed", "3"]) == 0
    return folder


def test_simulate_writes_all_files(data_dir):
    assert {p.name for p in data_dir.iterdir()} == {
        "logs.csv",
        "tasks.csv",
        "pilot.csv",
        "students_truth.csv",
    }


def test_profile(data_dir, tmp_path, capsys):
    out = tmp_path / "results"
    code = main(
        [
            "profile",
            "--logs",
            str(data_dir / "logs.csv"),
            "--tasks",
            str(data_dir / "tasks.csv"),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert "clusters: k =" in capsys.readouterr().out
    assert (out / "task_factors.png").is_file()
    assert len(pd.read_csv(out / "clusters.csv")) == 60


def test_profile_with_behaviour_features(data_dir, tmp_path):
    code = main(
        [
            "profile",
            "--logs",
            str(data_dir / "logs.csv"),
            "--with-behaviour",
            "--tasks",
            str(data_dir / "tasks.csv"),
            "--out",
            str(tmp_path / "r"),
        ]
    )
    assert code == 0


def test_recommend(data_dir, capsys):
    code = main(
        ["recommend", "--logs", str(data_dir / "logs.csv"), "--student", "S001", "--n", "2"]
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert code == 0
    assert lines[0].split()[:2] == ["task_id", "reason"]
    assert len(lines) == 3


def test_ancova(data_dir, capsys):
    assert main(["ancova", "--data", str(data_dir / "pilot.csv")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("F(1, 57)")
    assert "adjusted mean treatment" in out


def test_ancova_warns_when_assumptions_fail(tmp_path, capsys):
    pilot = pd.DataFrame(
        {
            "group": ["control"] * 6 + ["treatment"] * 6,
            "pre": [1, 2, 3, 4, 5, 6] * 2,
            "post": [1, 2, 3, 4, 5, 6, 12, 10, 8, 6, 4, 2],  # opposite slopes
        }
    )
    pilot.to_csv(tmp_path / "pilot.csv", index=False)
    assert main(["ancova", "--data", str(tmp_path / "pilot.csv")]) == 0
    assert "warning" in capsys.readouterr().out


def test_report(data_dir, tmp_path, capsys):
    out = tmp_path / "site"
    assert main(["report", "--data", str(data_dir), "--out", str(out)]) == 0
    assert "report written" in capsys.readouterr().out
    page = (out / "index.html").read_text(encoding="utf-8")
    for section in ("1. Data", "Latent skill factors", "Learner profiles", "ANCOVA"):
        assert section in page
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["data"]["students"] == 60
    assert summary["nmf"]["converged"] is True


def test_build_report_returns_the_summary(data_dir, tmp_path):
    summary = build_report(**load_study(data_dir), out_dir=tmp_path / "x", n_components=2)
    assert summary["nmf"]["components"] == 2
    assert summary["clusters"]["k"] >= 2


def test_invalid_data_exits_with_code_2(data_dir, tmp_path, capsys):
    logs = pd.read_csv(data_dir / "logs.csv")
    logs.loc[0, "correct"] = 5
    logs.drop(columns="time_sec").to_csv(tmp_path / "bad.csv", index=False)
    assert main(["recommend", "--logs", str(tmp_path / "bad.csv"), "--student", "S001"]) == 2
    assert "missing column 'time_sec'" in capsys.readouterr().err


def test_missing_files_and_unknown_student_exit_with_code_2(data_dir, tmp_path, capsys):
    assert main(["report", "--data", str(tmp_path / "nowhere")]) == 2
    assert main(["recommend", "--logs", str(data_dir / "logs.csv"), "--student", "X"]) == 2
    err = capsys.readouterr().err
    assert "missing logs.csv" in err
    assert "unknown student" in err


def test_module_entry_point():
    run = subprocess.run(
        [sys.executable, "-m", "ctgame", "--version"], capture_output=True, text=True, check=False
    )
    assert run.returncode == 0
    assert run.stdout.startswith("ctgame ")


def test_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("ctgame ")
