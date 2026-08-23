import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_dataset import build_prompt, extract_target, load_pair  # noqa: E402


def test_extract_target_skips_preamble():
    md = "# Git Digest — 2026-08-22\n\n_Lookback: 1 day(s)_\n\n## Summary\n\nShipped stuff.\n"
    assert extract_target(md) == "## Summary\n\nShipped stuff."


def test_extract_target_prefers_journal_marker():
    md = "# Developer Journal — 2026-08-22\n\n## Summary\n\nBody."
    target = extract_target(md)
    assert target.startswith("# Developer Journal")


def test_build_prompt_matches_go_format():
    activity = {"usr-wwelsh/botdocs": {"commits": [
        {"sha": "d211a452af15c96abe259b0430ece9cdf03a8a82", "message": "docs: document --watch\n\nlong body"}
    ]}}
    prompt = build_prompt("2026-08-22", 1, "usr-wwelsh", activity)
    assert "GitHub commits for usr-wwelsh, 2026-08-22 (last 1 day(s)):" in prompt
    assert "**usr-wwelsh/botdocs**\n" in prompt
    assert "- d211a45: docs: document --watch\n" in prompt
    assert "\n\n" not in "- d211a45: docs: document --watch"
    assert prompt.rstrip().endswith("Do not invent features, filenames, or behaviors not evidenced in the data.")


def test_load_pair(tmp_path):
    (tmp_path / "2026-08-22-1d-commits.json").write_text(
        '{"usr-wwelsh/x": {"commits": [{"sha": "abc1234ffffffff", "message": "fix: y"}]}}'
    )
    (tmp_path / "2026-08-22-1d.md").write_text("# Git Digest — 2026-08-22\n\n## Summary\n\nDid x.\n")
    pair = load_pair(tmp_path / "2026-08-22-1d.md")
    assert pair["date"] == "2026-08-22"
    assert "- abc1234: fix: y" in pair["prompt"]
    assert pair["completion"] == "## Summary\n\nDid x."
