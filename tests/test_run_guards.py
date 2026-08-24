import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from grpo import abort_reason  # noqa: E402
from sft import resolve_resume  # noqa: E402


def make_ckpt(tmp_path: Path, step: int = 3) -> Path:
    d = tmp_path / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "trainer_state.json").write_text("{}")
    return tmp_path


# --- resume guard -----------------------------------------------------------


def test_fresh_output_dir_starts_from_scratch(tmp_path):
    assert resolve_resume(tmp_path / "sft3", resume=False) is False


def test_existing_checkpoints_refuse_a_silent_fresh_run(tmp_path):
    make_ckpt(tmp_path)
    with pytest.raises(ValueError, match="--resume"):
        resolve_resume(tmp_path, resume=False)


def test_resume_flag_picks_up_existing_checkpoints(tmp_path):
    make_ckpt(tmp_path)
    assert resolve_resume(tmp_path, resume=True) is True


def test_resume_without_checkpoints_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="nothing to resume"):
        resolve_resume(tmp_path / "missing", resume=True)


def test_output_dir_without_checkpoints_is_not_stale(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert resolve_resume(tmp_path, resume=False) is False


# --- divergence guard -------------------------------------------------------


def tick(step, train, spot, tokens=400, trunc=0.0):
    return {"step": step, "train_reward": train, "spotcheck_total": spot,
            "tokens": tokens, "truncated_rate": trunc}


def test_healthy_run_is_not_aborted():
    hist = [tick(1, 0.40, 0.42), tick(2, 0.48, 0.47), tick(3, 0.55, 0.53)]
    assert abort_reason(hist) is None


def test_short_history_never_aborts():
    hist = [tick(1, 0.40, 0.60), tick(2, 0.90, 0.20)]
    assert abort_reason(hist) is None


def test_train_reward_climbing_while_heldout_stalls_aborts():
    hist = [tick(1, 0.40, 0.55), tick(2, 0.60, 0.54), tick(3, 0.78, 0.53)]
    assert "reward" in abort_reason(hist)


def test_heldout_gain_excuses_a_climbing_train_reward():
    hist = [tick(1, 0.40, 0.55), tick(2, 0.60, 0.60), tick(3, 0.78, 0.66)]
    assert abort_reason(hist) is None


def test_flat_train_reward_with_flat_heldout_is_not_divergence():
    hist = [tick(1, 0.50, 0.55), tick(2, 0.51, 0.54), tick(3, 0.52, 0.53)]
    assert abort_reason(hist) is None


def test_sustained_length_collapse_aborts():
    hist = [tick(1, 0.40, 0.50, tokens=400), tick(2, 0.45, 0.52, tokens=380),
            tick(3, 0.50, 0.54, tokens=120), tick(4, 0.52, 0.56, tokens=90),
            tick(5, 0.54, 0.58, tokens=80)]
    assert "length" in abort_reason(hist)


def test_steady_short_output_is_a_style_not_a_collapse():
    hist = [tick(1, 0.40, 0.50, tokens=180), tick(2, 0.45, 0.52, tokens=175),
            tick(3, 0.50, 0.54, tokens=170)]
    assert abort_reason(hist) is None


def test_one_short_spotcheck_day_does_not_abort():
    # spotcheck cycles held-out days, so a single quiet day is legitimately short
    hist = [tick(1, 0.40, 0.50, tokens=400), tick(2, 0.45, 0.52, tokens=90),
            tick(3, 0.50, 0.54, tokens=390)]
    assert abort_reason(hist) is None


def test_sustained_truncation_aborts():
    hist = [tick(1, 0.40, 0.50, trunc=0.7), tick(2, 0.45, 0.52, trunc=0.8),
            tick(3, 0.50, 0.54, trunc=0.9)]
    assert "truncat" in abort_reason(hist)


def test_occasional_truncation_is_tolerated():
    hist = [tick(1, 0.40, 0.50, trunc=0.0), tick(2, 0.45, 0.52, trunc=0.5),
            tick(3, 0.50, 0.54, trunc=0.0)]
    assert abort_reason(hist) is None


def test_only_the_recent_window_is_judged():
    # an early stall must not keep firing once the run recovers
    hist = [tick(1, 0.40, 0.55), tick(2, 0.60, 0.54), tick(3, 0.78, 0.53),
            tick(4, 0.80, 0.62), tick(5, 0.82, 0.70), tick(6, 0.84, 0.78)]
    assert abort_reason(hist) is None
