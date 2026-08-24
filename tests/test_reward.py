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

Alpha beta gamma delta epsilon all landed today alongside general polish,
cleanup work, and small consistency improvements throughout.
"""
    s = score_digest(digest, solo)
    assert s["grounding"] == 1.0


def test_bare_source_word_list_discounted():
    solo = {"a/b": {"commits": [{"sha": "1", "message": "alpha beta gamma delta epsilon zeta"}]}}
    digest = """# Journal

## Summary

Work happened today across the board.

## Per-Repo Activity

### b

Alpha beta gamma delta epsilon mentioned here.
"""
    s = score_digest(digest, solo)
    assert s["grounding"] < 1.0


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


def test_empty_body_section_fails_coverage():
    gamed = """# Developer Journal

## Summary

Turbolab and botdocs both saw active development work today across the board.

## Per-Repo Activity

### usr-wwelsh/turbolab

### usr-wwelsh/botdocs
"""
    s = score_digest(gamed, ACTIVITY)
    assert s["coverage"] == 0.0
    assert s["total"] < 0.4


def test_unrelated_filler_body_fails_coverage():
    gamed = GOOD.replace(
        "Removed build warnings and switched systemd unit resolution to absolute paths.",
        "banana mango kayak umbrella").replace(
        "Added a sitemap generator feature.",
        "giraffe volcano ticket window")
    s = score_digest(gamed, ACTIVITY)
    assert s["coverage"] == 0.0
    assert s["total"] < 0.4


def test_copied_sha_body_fails_coverage():
    gamed = GOOD.replace(
        "Removed build warnings and switched systemd unit resolution to absolute paths.",
        "ac4a9c1 d49b6c9 commit reference").replace(
        "Added a sitemap generator feature.",
        "1111111 commit reference here")
    s = score_digest(gamed, ACTIVITY)
    assert s["coverage"] == 0.0


def test_single_keyword_plus_filler_fails_coverage():
    gamed = GOOD.replace(
        "Removed build warnings and switched systemd unit resolution to absolute paths.",
        "systemd banana mango").replace(
        "Added a sitemap generator feature.",
        "sitemap giraffe volcano")
    s = score_digest(gamed, ACTIVITY)
    assert s["coverage"] == 0.0
    assert s["total"] < 0.4


def test_terse_but_grounded_body_passes_coverage():
    terse = GOOD.replace(
        "Removed build warnings and switched systemd unit resolution to absolute paths.",
        "Fixed build warnings and systemd path.")
    assert score_digest(terse, ACTIVITY)["coverage"] == 1.0


def test_verbatim_commit_dump_scores_below_honest():
    solo = {"usr-wwelsh/turbolab": {"commits": [
        {"sha": s, "message": m} for s, m in [
            ("1", "feat: add ram-aware quant selection"),
            ("2", "fix: stream upload timeouts"),
            ("3", "chore: bump llama cpp backend"),
            ("4", "feat: openai compatible chat completions"),
            ("5", "fix: web ui token counter overflow"),
            ("6", "docs: readme quickstart section"),
        ]]}}
    def digest(body):
        return f"""# Developer Journal

## Summary

Turbolab shipped several backend and interface improvements across the day.

## Per-Repo Activity

### usr-wwelsh/turbolab

{body}
"""
    good = digest(
        "Work focused on making quantization pick the right size for available memory "
        "and on steadier uploads. The chat endpoint gained compatibility improvements, "
        "and the web interface got a small counting fix alongside setup docs.")
    dump = digest("\n".join(
        f"- {m}" for _, m in solo["usr-wwelsh/turbolab"]["commits"]))
    assert score_digest(dump, solo)["total"] < score_digest(good, solo)["total"]


def test_small_similar_sections_on_big_day_not_nuked():
    big = {
        f"usr-wwelsh/repo{i}": {"commits": [
            {"sha": str(i), "message": f"docs: read{me} badge update"}]}
        for i, me in enumerate(["me"] * 6)
    }
    lines = ["# Developer Journal", "", "## Summary", "",
             "Quiet day across six repositories with only documentation housekeeping landed.", "",
             "## Per-Repo Activity"]
    for i in range(6):
        lines += ["", f"### usr-wwelsh/repo{i}", "", "Readme badge documentation touched briefly."]
    digest = "\n".join(lines) + "\n"
    s = score_digest(digest, big)
    assert s["penalties"] == 0.0
    assert s["total"] > 0.6


def test_short_repo_name_accepted():
    alt = GOOD.replace("### usr-wwelsh/turbolab", "### turbolab")
    s = score_digest(alt, ACTIVITY)
    assert s["format"] == 1.0


FILES_ACTIVITY = {
    "usr-wwelsh/turbolab": {"commits": [
        {"sha": "ac4a9c1", "message": "chore: build warning fixes", "files": [
            {"filename": "web/src/App.svelte", "additions": 1, "deletions": 1},
            {"filename": "web/src/lib/Memory.svelte", "additions": 6, "deletions": 1},
        ]},
        {"sha": "d49b6c9", "message": "fix: systemd absolute path", "files": [
            {"filename": "cmd/setup.go", "additions": 38, "deletions": 6},
        ]},
    ]},
}

FILES_GOOD = """# Developer Journal

## Summary

Turbolab cleaned up build warnings in App.svelte and fixed systemd absolute path handling in setup.go.

## Per-Repo Activity

### usr-wwelsh/turbolab

Removed build warnings from App.svelte and switched setup.go to absolute paths for systemd.
"""


def test_fabricated_filename_penalized():
    # real overlap padded with a specific file that was never touched -- exactly the
    # pattern that let a hallucinated claim score 0.787 in practice: enough true keywords
    # to max grounding, with an invented specific detail costing nothing under the old scoring
    padded = FILES_GOOD.replace(
        "Removed build warnings from App.svelte and switched setup.go to absolute paths for systemd.",
        "Removed build warnings from App.svelte and libMemory.svelte, and switched setup.go "
        "to absolute paths for systemd.")
    good_score = score_digest(FILES_GOOD, FILES_ACTIVITY)
    bad_score = score_digest(padded, FILES_ACTIVITY)
    assert bad_score["penalties"] > good_score["penalties"]
    assert bad_score["total"] < good_score["total"]


def test_real_filename_not_penalized():
    # Memory.svelte *was* touched -- citing it by basename shouldn't cost anything
    real = FILES_GOOD.replace(
        "Removed build warnings from App.svelte and switched setup.go to absolute paths for systemd.",
        "Removed build warnings from App.svelte and Memory.svelte, and switched setup.go "
        "to absolute paths for systemd.")
    assert score_digest(real, FILES_ACTIVITY)["penalties"] == 0.0


def test_tech_brand_name_not_penalized():
    # "Node.js" matches the filename-mention shape but isn't a file claim
    brand = FILES_GOOD.replace(
        "Turbolab cleaned up build warnings in App.svelte and fixed systemd absolute path handling in setup.go.",
        "Turbolab cleaned up build warnings in its Node.js tooling and fixed systemd absolute path handling.")
    assert score_digest(brand, FILES_ACTIVITY)["penalties"] == 0.0


def test_known_dependency_name_not_penalized():
    # llama.cpp is a real upstream dependency name, not a file claim
    dep = FILES_GOOD.replace(
        "Turbolab cleaned up build warnings in App.svelte and fixed systemd absolute path handling in setup.go.",
        "Turbolab cleaned up a llama.cpp memory leak and fixed systemd absolute path handling in setup.go.")
    assert score_digest(dep, FILES_ACTIVITY)["penalties"] == 0.0


def test_filename_from_commit_message_not_penalized():
    # a filename named only in the commit message (e.g. a generated output, not itself
    # part of the diff) is still grounded -- the message is shown to the model too
    activity = {"usr-wwelsh/botdocs": {"commits": [
        {"sha": "1", "message": "feat: emit sitemap.xml via baseUrl config", "files": [
            {"filename": "src/builder/site-generator.ts", "additions": 20, "deletions": 0},
        ]},
    ]}}
    digest = """# Developer Journal

## Summary

botdocs now emits sitemap.xml during the build.

## Per-Repo Activity

### usr-wwelsh/botdocs

Wired sitemap.xml generation into the site builder via a baseUrl config option.
"""
    assert score_digest(digest, activity)["penalties"] == 0.0
