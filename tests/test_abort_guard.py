import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from grpo import abort_reason, spotcheck_entry  # noqa: E402


def tick(step, train, spot, tokens=400, ref=400, trunc=0.0):
    return {"step": step, "train_reward": train, "spotcheck_total": spot,
            "tokens": tokens, "ref_tokens": ref, "truncated_rate": trunc}


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
    hist = [tick(1, 0.40, 0.50, tokens=180, ref=180), tick(2, 0.45, 0.52, tokens=175, ref=180),
            tick(3, 0.50, 0.54, tokens=170, ref=180)]
    assert abort_reason(hist) is None


def test_one_short_spotcheck_day_does_not_abort():
    # spotcheck cycles held-out days, so a single quiet day is legitimately short
    hist = [tick(1, 0.40, 0.50, tokens=400), tick(2, 0.45, 0.52, tokens=90),
            tick(3, 0.50, 0.54, tokens=390)]
    assert abort_reason(hist) is None


def test_short_days_in_a_row_are_not_a_collapse_when_their_labels_are_short():
    # the real eval set's reference digests run 133-316 tokens and the spotcheck cycles
    # them in order, so rows 3-5 are three genuinely short days back to back. Judged
    # against a cross-day peak (316) that read as collapse; judged against each day's own
    # label it is a policy tracking its targets.
    hist = [tick(1, 0.40, 0.50, tokens=204, ref=204), tick(2, 0.45, 0.52, tokens=133, ref=133),
            tick(3, 0.50, 0.54, tokens=150, ref=150)]
    assert abort_reason(hist) is None


def test_collapse_on_a_long_day_aborts_even_after_short_days():
    # the mirror case: absolute token counts alone can't tell these two apart
    hist = [tick(1, 0.40, 0.50, tokens=133, ref=133), tick(2, 0.45, 0.52, tokens=60, ref=316),
            tick(3, 0.50, 0.54, tokens=55, ref=292)]
    assert "length" in abort_reason(hist)


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


def test_entry_averages_k_rollouts_of_the_pinned_day():
    # Finding 4: grpo3 died on spotchecks 1.0, 1.0, 0.05 -- three different days at
    # ONE sampled rollout each, so a single unlucky draw gated a GPU window. Each tick
    # must carry the mean over k rollouts of one fixed day.
    e = spotcheck_entry(step=20, train_reward=0.5, truncated_rate=0.1,
                        totals=[1.0, 1.0, 0.05, 0.95],
                        token_counts=[400, 380, 420, 400], ref_tokens=400)
    assert e["spotcheck_total"] == 0.75
    assert e["tokens"] == 400
    assert e["ref_tokens"] == 400
    assert e["rollouts"] == 4


def test_day_hopping_n1_spotchecks_abort_on_noise():
    # the grpo3 killer, preserved as documentation: train climbing while three
    # DIFFERENT days are sampled ONCE each. The 0.05 is a single penalty-heavy
    # draw from a hard day -- indistinguishable from divergence under this input.
    hist = [tick(20, 0.40, 1.00), tick(40, 0.60, 1.00), tick(60, 0.78, 0.05)]
    assert "reward" in abort_reason(hist)


def test_one_bad_rollout_cannot_gate_a_window():
    # the SAME training climb measured the fixed way: every tick a 4-rollout mean
    # of one pinned day. A catastrophic draw (0.05, the grpo3 value) lands inside
    # a tick and is absorbed by the other three; endpoint means hold, window stays.
    hist = [
        spotcheck_entry(20, 0.40, 0.0, [0.90, 0.95, 0.95, 1.00], [400] * 4, 400),
        spotcheck_entry(40, 0.60, 0.0, [0.05, 1.00, 1.00, 1.00], [400] * 4, 400),
        spotcheck_entry(60, 0.78, 0.0, [0.95, 1.00, 0.95, 1.00], [400] * 4, 400),
    ]
    assert abort_reason(hist) is None


def test_real_same_day_regression_still_aborts():
    # denoising must not blind the guard: a persistent drop across ticks of the SAME
    # day is genuine divergence, and train climbing through it is exactly the tell
    hist = [
        spotcheck_entry(20, 0.40, 0.0, [0.95, 0.90, 0.95, 0.90], [400] * 4, 400),
        spotcheck_entry(40, 0.62, 0.0, [0.60, 0.55, 0.65, 0.60], [380] * 4, 400),
        spotcheck_entry(60, 0.80, 0.0, [0.30, 0.25, 0.35, 0.30], [350] * 4, 400),
    ]
    assert "reward" in abort_reason(hist)
