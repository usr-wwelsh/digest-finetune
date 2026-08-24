import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from reward import parse_sections, score_digest  # noqa: E402

ACTIVITY = {
    "usr-wwelsh/turbolab": {"commits": [
        {"sha": "ac4a9c1", "message": "chore: build warning fixes"},
        {"sha": "d49b6c9", "message": "fix: systemd absolute path"},
    ]},
    "usr-wwelsh/botdocs": {"commits": [
        {"sha": "1111111", "message": "feat: add sitemap generator"},
    ]},
}

GOOD = """# Developer Journal

## Summary

Turbolab cleaned up build warnings and fixed systemd absolute path handling. botdocs gained a sitemap generator.

## Per-Repo Activity

### usr-wwelsh/turbolab

Removed build warnings and switched systemd unit resolution to absolute paths.

### usr-wwelsh/botdocs

Added a sitemap generator feature.
"""


def test_good_digest_scores_high():
    s = score_digest(GOOD, ACTIVITY)
    assert s["format"] == 1.0
    assert s["coverage"] == 1.0
    assert s["grounding"] >= 0.6
    assert s["total"] > 0.85


def test_weights_sum_to_one():
    bad = score_digest("", ACTIVITY)
    best_case = 0.30 * 1.0 + 0.45 * 1.0 + 0.25 * 1.0
    assert best_case == 1.0


def test_hallucinated_repo_detected():
    bad = GOOD.replace("### usr-wwelsh/botdocs\n\nAdded a sitemap generator feature.",
                       "### usr-wwelsh/turbolab-gnome\n\nAdded a sitemap generator feature.")
    s = score_digest(bad, ACTIVITY)
    assert s["format"] < 1.0
    assert s["coverage"] < 1.0


def test_hallucinated_section_claim_drops_grounding():
    bad = GOOD.replace(
        "Removed build warnings and switched systemd unit resolution to absolute paths.",
        "Deployed quantum blockchain infrastructure to production.")
    assert score_digest(bad, ACTIVITY)["grounding"] < 0.5


def test_empty_summary_fails_format():
    bad = GOOD.replace("## Summary\n\nTurbolab cleaned up build warnings and fixed systemd absolute path handling. botdocs gained a sitemap generator.\n", "## Summary\n\n")
    assert score_digest(bad, ACTIVITY)["format"] < score_digest(GOOD, ACTIVITY)["format"]


def test_grounding_saturates_at_five_source_tokens():
    solo = {"a/b": {"commits": [{"sha": "1", "message": "alpha beta gamma delta epsilon zeta"}]}}
    digest = """# Journal

## Summary

Work happened today across the board.

## Per-Repo Activity

### b

Alpha beta gamma delta epsilon mentioned here.
"""
    s = score_digest(digest, solo)
    assert s["grounding"] == 1.0


def test_no_sections_zero_grounding():
    bare = "# Journal\n\n## Summary\n\nNothing structured here at all.\n"
    assert score_digest(bare, ACTIVITY)["grounding"] == 0.0


def test_missing_repo_section_fails_coverage():
    bad = GOOD.replace("### usr-wwelsh/botdocs\n\nAdded a sitemap generator feature.\n", "")
    assert score_digest(bad, ACTIVITY)["coverage"] < 1.0


def test_banned_phrase():
    bad = GOOD + "\nNext steps: you should refactor everything.\n"
    s = score_digest(bad, ACTIVITY)
    assert s["penalties"] > 0


def test_duplicate_section_penalized():
    dup = GOOD + "\n### usr-wwelsh/turbolab\n\nRemoved build warnings and switched systemd unit resolution to absolute paths.\n"
    s = score_digest(dup, ACTIVITY)
    assert s["penalties"] > 0


def test_copy_pasted_bodies_penalized():
    bad = GOOD.replace(
        "Added a sitemap generator feature.",
        "Removed build warnings and switched systemd unit resolution to absolute paths.")
    s = score_digest(bad, ACTIVITY)
    assert s["penalties"] > 0
    assert s["total"] < score_digest(GOOD, ACTIVITY)["total"]


def test_hallucinated_repo_variants_penalized():
    spammy = GOOD + """
### usr-wwelsh/turbolab-dev

Removed build warnings and switched systemd unit resolution to absolute paths.

### usr-wwelsh/turbolab-dev-dev

Removed build warnings and switched systemd unit resolution to absolute paths.
"""
    s = score_digest(spammy, ACTIVITY)
    assert s["total"] < score_digest(GOOD, ACTIVITY)["total"]


def test_short_repo_name_accepted():
    alt = GOOD.replace("### usr-wwelsh/turbolab", "### turbolab")
    s = score_digest(alt, ACTIVITY)
    assert s["format"] == 1.0
