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
    assert s["grounding"] > 0.5
    assert s["total"] > 0.8


def test_hallucinated_repo_detected():
    bad = GOOD.replace("### usr-wwelsh/botdocs\n\nAdded a sitemap generator feature.",
                       "### usr-wwelsh/turbolab-gnome\n\nAdded a sitemap generator feature.")
    s = score_digest(bad, ACTIVITY)
    assert s["format"] < 1.0
    assert s["coverage"] < 1.0


def test_hallucinated_claim_drops_grounding():
    bad = GOOD.replace("cleaned up build warnings",
                       "deployed quantum blockchain infrastructure")
    s = score_digest(bad, ACTIVITY)
    assert s["grounding"] < 0.7


def test_empty_summary_fails_format():
    bad = GOOD.replace("## Summary\n\nTurbolab cleaned up build warnings and fixed systemd absolute path handling. botdocs gained a sitemap generator.\n", "## Summary\n\n")
    assert score_digest(bad, ACTIVITY)["format"] == 0.0


def test_repetition_penalized():
    para = "\n\nThe same sentence repeats here exactly."
    bad = GOOD.replace("Added a sitemap generator feature.", f"Added a sitemap generator feature.{para}{para}{para}")
    assert score_digest(bad, ACTIVITY)["repetition"] < score_digest(GOOD, ACTIVITY)["repetition"]


def test_missing_repo_section_fails_coverage():
    bad = GOOD.replace("### usr-wwelsh/botdocs\n\nAdded a sitemap generator feature.\n", "")
    assert score_digest(bad, ACTIVITY)["coverage"] < 1.0


def test_banned_phrase():
    bad = GOOD + "\nNext steps: you should refactor everything.\n"
    s = score_digest(bad, ACTIVITY)
    assert s["penalties"] > 0


def test_short_repo_name_accepted():
    alt = GOOD.replace("### usr-wwelsh/turbolab", "### turbolab")
    s = score_digest(alt, ACTIVITY)
    assert s["format"] == 1.0
