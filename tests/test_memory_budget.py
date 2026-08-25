import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from grpo import (  # noqa: E402
    attention_spike_bytes,
    max_sequence_for_budget,
    oversized_rows,
    plan_dataset,
)

GIB = 2 ** 30


def test_spike_reproduces_the_observed_t4_oom():
    # the run died in backward on 2026-08-24 asking for exactly 3.58 GiB: batch 8
    # (prompts-per-step 1 x gens 8), SmolLM2-135M's 9 heads, the 3014-token prompt
    # plus ~640 completion tokens, fp32. this is the incident, as a number.
    got = attention_spike_bytes(batch=8, heads=9, seq_len=3654)
    assert got / GIB == pytest.approx(3.58, abs=0.01)


def test_spike_is_quadratic_in_sequence_not_linear():
    # the whole point: halving the batch buys half, halving the length buys four times
    base = attention_spike_bytes(batch=8, heads=9, seq_len=3654)
    assert attention_spike_bytes(batch=4, heads=9, seq_len=3654) == base / 2
    assert attention_spike_bytes(batch=8, heads=9, seq_len=1827) == base / 4


def test_max_sequence_inverts_the_spike():
    seq = max_sequence_for_budget(batch=8, heads=9, budget_bytes=int(1.5 * GIB))
    assert attention_spike_bytes(batch=8, heads=9, seq_len=seq) <= 1.5 * GIB
    assert attention_spike_bytes(batch=8, heads=9, seq_len=seq + 1) > 1.5 * GIB


def test_oversized_rows_reports_indices_over_the_cap():
    assert oversized_rows([100, 900, 250, 1200], max_prompt=800) == [1, 3]


def test_nothing_is_dropped_when_everything_fits():
    assert oversized_rows([100, 200, 300], max_prompt=800) == []


def test_plan_keeps_rows_that_fit_and_drops_the_rest():
    lens = [500, 3014, 700, 2334]
    plan = plan_dataset(lens, batch=8, heads=9, max_completion=768,
                        budget_bytes=int(1.5 * GIB), max_drop_frac=0.5)
    assert plan.keep == [0, 2]
    assert plan.drop == [1, 3]
    assert plan.max_prompt > 0


def test_plan_refuses_when_the_config_would_gut_the_dataset():
    # dropping a few long synthetic days is data curation; dropping most of the set
    # means the batch or completion cap is wrong, and that is a config bug not a
    # data problem -- fail closed rather than train on the remainder
    lens = [3000] * 8 + [400] * 2
    with pytest.raises(SystemExit, match="80%"):
        plan_dataset(lens, batch=8, heads=9, max_completion=768,
                     budget_bytes=int(1.5 * GIB), max_drop_frac=0.25)


def test_plan_allows_a_deliberate_high_drop_fraction():
    lens = [3000] * 8 + [400] * 2
    plan = plan_dataset(lens, batch=8, heads=9, max_completion=768,
                        budget_bytes=int(1.5 * GIB), max_drop_frac=1.0)
    assert plan.keep == [8, 9]


def test_shorter_completions_buy_back_long_prompts():
    # the trade the operator actually has: max_completion comes out of the same budget
    lens = [1800]
    assert plan_dataset(lens, batch=8, heads=9, max_completion=768,
                        budget_bytes=int(1.5 * GIB), max_drop_frac=1.0).drop == [0]
    assert plan_dataset(lens, batch=8, heads=9, max_completion=512,
                        budget_bytes=int(1.5 * GIB), max_drop_frac=1.0).drop == []
