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


def test_one_fabricated_header_costs_like_a_headline_lie():
    # Finding 2: the 1.000-scoring sample attributed yesterday's omniMux work to a
    # repo that never ran that day; under 0.15/section, misattributing a whole day's
    # work was cheaper than a single hallucinated filename (0.3). Price it like the
    # headline lie it is.
    miscamed = GOOD.replace("### usr-wwelsh/botdocs", "### usr-wwelsh/not-a-real-repo")
    s = score_digest(miscamed, ACTIVITY)
    assert s["penalties"] >= 0.35


def test_two_fabricated_headers_dominate_the_score():
    double = GOOD.replace(
        "### usr-wwelsh/botdocs", "### usr-wwelsh/not-real").replace(
        "Added a sitemap generator feature.",
        "Removed build warnings and switched systemd unit resolution to absolute paths."
    ) + "\n### usr-wwelsh/also-fake\n\nMore filler prose claiming unrelated work happened here.\n"
    s = score_digest(double, ACTIVITY)
    assert s["penalties"] >= 0.7
    assert s["total"] < 0.5


def test_invented_model_name_penalized():
    # Finding 3: a generation scoring 1.000 called the model GPT-2 -- a capitalised
    # proper noun absent from every byte of its prompt. The scorer must notice.
    invented = GOOD.replace(
        "Removed build warnings and switched systemd unit resolution to absolute paths.",
        "Removed build warnings using GPT-2 embeddings and switched systemd paths.")
    s = score_digest(invented, ACTIVITY)
    assert s["penalties"] > 0
    assert s["total"] < score_digest(GOOD, ACTIVITY)["total"]


def test_model_name_present_in_prompt_not_penalized():
    # the same shape becomes legitimate the moment the prompt itself names the model
    grounded_activity = {
        "usr-wwelsh/turbolab": {"commits": [
            {"sha": "ac4a9c1", "message": "chore: build warning fixes"},
            {"sha": "d49b6c9", "message": "fix: systemd absolute path"},
            {"sha": "e777777", "message": "chore: bump smollm2 tokenizer"},
        ]},
        "usr-wwelsh/botdocs": ACTIVITY["usr-wwelsh/botdocs"],
    }
    named = GOOD.replace(
        "Removed build warnings and switched systemd unit resolution to absolute paths.",
        "Removed build warnings and refreshed the SmolLM2 tokenizer paths.")
    assert score_digest(named, grounded_activity)["penalties"] == 0.0


def test_sentence_start_capital_not_penalized():
    # ordinary prose starts sentences with capitals; only mid-sentence proper nouns
    # absent from the prompt are fabrication candidates
    plain = GOOD.replace(
        "Removed build warnings and switched systemd unit resolution to absolute paths.",
        "Systemd units now resolve through absolute paths, closing the warning batch.")
    assert score_digest(plain, ACTIVITY)["penalties"] == 0.0


def test_wrong_repo_count_claim_penalized():
    # Finding 3: "Two active projects today" over a one-repo day, "Three active
    # repos" in the body -- headline arithmetic the activity dict flatly contradicts
    liar = GOOD.replace(
        "Turbolab cleaned up build warnings and fixed systemd absolute path handling. "
        "botdocs gained a sitemap generator.",
        "Three active repositories shipped work today across the board.")
    s = score_digest(liar, ACTIVITY)
    assert s["penalties"] > 0
    assert s["total"] < score_digest(GOOD, ACTIVITY)["total"]


def test_correct_repo_count_claim_not_penalized():
    counted = GOOD.replace(
        "Turbolab cleaned up build warnings and fixed systemd absolute path handling. "
        "botdocs gained a sitemap generator.",
        "Two active repositories shipped work today across the board.")
    assert score_digest(counted, ACTIVITY)["penalties"] == 0.0


QUOTE_ACTIVITY = {
    "usr-wwelsh/turbolab": {"commits": [
        {"sha": "ac4a9c1", "message": "chore: build warning fixes"},
        {"sha": "d49b6c9", "message": "fix: systemd absolute path"},
        {"sha": "e555555", "message": "chore: raised the grpo learning rate to "
                                     "3e-5 before the next training window"},
    ]},
}


def test_verbatim_quote_of_commit_subject_not_penalized():
    # Finding 3 inversion: the only sample that accurately tracked "raise the GRPO LR
    # to 3e-5" scored LOWEST because restating one true thing tripped the summary/body
    # echo penalty. Quoting the prompt's own words is accuracy, not self-plagiarism --
    # the exemption belongs where the shared run traces to real commit material.
    quoted = """# Developer Journal

## Summary

Turbolab raised the grpo learning rate to 3e-5 before the next training window today.

## Per-Repo Activity

### usr-wwelsh/turbolab

Raised the grpo learning rate to 3e-5 before the next training window, alongside
the systemd path cleanup.
"""
    s = score_digest(quoted, QUOTE_ACTIVITY)
    assert s["penalties"] == 0.0


def test_verbatim_quote_of_invented_text_still_penalized():
    # the mirror: an identical echo with NO source to ground it stays charged
    invented = """# Developer Journal

## Summary

The pipeline now handles backpressure across every consumer queue gracefully today.

## Per-Repo Activity

### usr-wwelsh/turbolab

The pipeline now handles backpressure across every consumer queue gracefully.
"""
    s = score_digest(invented, ACTIVITY)
    assert s["penalties"] > 0


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
             "Readme badge work dominated a quiet day of docs upkeep across six repositories.", "",
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


def test_fabricated_filename_in_summary_penalized():
    # same exploit as the body case but hidden in the summary, which penalties()
    # used to never scan -- a hallucinated specific detail cost nothing there
    padded = FILES_GOOD.replace(
        "Turbolab cleaned up build warnings in App.svelte and fixed systemd absolute path handling in setup.go.",
        "Turbolab cleaned up build warnings in App.svelte and Widget.svelte, and fixed systemd "
        "absolute path handling in setup.go.")
    good_score = score_digest(FILES_GOOD, FILES_ACTIVITY)
    bad_score = score_digest(padded, FILES_ACTIVITY)
    assert bad_score["penalties"] > good_score["penalties"]
    assert bad_score["total"] < good_score["total"]


def test_ungrounded_summary_scores_below_grounded_summary():
    # the summary is 1/3 of format score; swapping it for fluent-but-unrelated
    # prose used to leave the total untouched -- free fabrication channel
    swapped = FILES_GOOD.replace(
        "Turbolab cleaned up build warnings in App.svelte and fixed systemd absolute path handling in setup.go.",
        "A productive day of refactoring across the board with meaningful progress on "
        "infrastructure and several quality-of-life improvements landing safely.")
    good_score = score_digest(FILES_GOOD, FILES_ACTIVITY)
    bad_score = score_digest(swapped, FILES_ACTIVITY)
    assert bad_score["format"] < good_score["format"]
    assert bad_score["total"] < good_score["total"]


def test_grounded_terse_summary_still_passes_format():
    # grounding the summary must not punish legitimate short-hand phrasing
    terse = FILES_GOOD.replace(
        "Turbolab cleaned up build warnings in App.svelte and fixed systemd absolute path handling in setup.go.",
        "Build warnings cleanup in App.svelte plus a systemd path fix in setup.go.")
    assert score_digest(terse, FILES_ACTIVITY)["format"] == 1.0


def test_orphan_prose_under_per_repo_header_penalized():
    # text sitting between "## Per-Repo Activity" and the first "###" header used to
    # be invisible to every scorer -- neither summary nor section, free hallucination space
    stray = FILES_GOOD.replace(
        "## Per-Repo Activity\n",
        "## Per-Repo Activity\n\nUnverified claims about src/app/Ghost.svelte appeared from nowhere.\n")
    good = score_digest(FILES_GOOD, FILES_ACTIVITY)
    bad = score_digest(stray, FILES_ACTIVITY)
    assert bad["total"] < good["total"]
    assert bad["penalties"] > 0


def test_blank_space_under_per_repo_header_not_penalized():
    spaced = FILES_GOOD.replace("## Per-Repo Activity\n", "## Per-Repo Activity\n\n\n")
    assert score_digest(spaced, FILES_ACTIVITY)["penalties"] == 0.0


def test_degenerate_activity_does_not_crash():
    for activity in (
        {"a/b": {"commits": [{}]}},
        {"a/b": {"commits": [{"sha": "1", "message": "ok", "files": [{}]}]}},
    ):
        s = score_digest("### a/b\n\nSome prose about things.\n", activity)
        assert 0.0 <= s["total"] <= 1.0


def test_two_keyword_accident_earns_only_partial_coverage():
    # a body whose ONLY tie to the repo is two generic words that happen to appear
    # in its commits ("cleanup", "scroll") used to pass the binary gate at full
    # credit -- exactly the gradient GRPO climbs. It must earn partial credit at most
    lucky = GOOD.replace(
        "Removed build warnings and switched systemd unit resolution to absolute paths.",
        "Cleanup and scroll work landed.").replace(
        "Added a sitemap generator feature.",
        "Cleanup and scroll efforts continued.")
    solo = {"usr-wwelsh/turbolab": {"commits": [
        {"sha": "1", "message": "chore: cleanup scroll handling"}]}}
    s = score_digest(lucky, solo)
    assert s["coverage"] < 1.0


def test_grounded_prose_still_full_coverage():
    assert score_digest(GOOD, ACTIVITY)["coverage"] == 1.0


def test_omitting_weak_section_cannot_raise_total():
    # grounding averaged over *present* sections only, so deleting a section with
    # no coverage credit but a low grounding hit RAISED the total (+0.038 on a real
    # synthetic day) -- hiding a repo you can't write about must never pay
    three = dict(ACTIVITY)
    three["usr-wwelsh/kickdrummer"] = {"commits": [
        {"sha": "9", "message": "feat: midi drum transcription pipeline"}]}
    full = GOOD.replace(
        "## Per-Repo Activity\n",
        """## Per-Repo Activity

### usr-wwelsh/kickdrummer

Sequencer groundwork and refactor momentum.
""")
    without_it = GOOD  # kickdrummer section simply absent
    assert score_digest(without_it, three)["total"] <= score_digest(full, three)["total"]
    assert score_digest(without_it, three)["grounding"] <= score_digest(full, three)["grounding"]


def test_missing_section_zeroes_grounding_for_that_repo():
    solo = {"a/b": {"commits": [{"sha": "1", "message": "alpha beta gamma delta"}]}}
    half = """# Journal

## Summary

Alpha beta work happened today alongside other things worth noting here.

## Per-Repo Activity

### b

Alpha beta gamma delta epsilon prose.
"""
    s = score_digest(half, {"a/b": solo["a/b"], "c/d": solo["a/b"]})
    # c/d has no section at all -- grounding must reflect that as a 0, not skip it
    assert s["grounding"] <= 0.55


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


LOOP_FRAGMENT = "bot_created_time DATE NOT NULL, bot_created_time DATETIME NOT NULL, "


def test_self_repeating_section_penalized():
    # the sft3 failure: one section looping a fragment until the token cap, which
    # scored a clean 1.0 because content_tokens() is a set and the cross-section
    # near-dup loop never runs when there is only one section
    looped = GOOD.replace(
        "Added a sitemap generator feature.",
        "Added a sitemap generator table: " + LOOP_FRAGMENT * 30)
    s = score_digest(looped, ACTIVITY)
    assert s["penalties"] > 0
    assert s["total"] < score_digest(GOOD, ACTIVITY)["total"]


def test_summary_restated_as_body_penalized():
    # samples 1 and 3: the body near-verbatim restates the Summary, which the
    # body-vs-body loop never compares
    echoed = GOOD.replace(
        "Removed build warnings and switched systemd unit resolution to absolute paths.",
        "Turbolab cleaned up build warnings and fixed systemd absolute path handling.")
    s = score_digest(echoed, ACTIVITY)
    assert s["penalties"] > 0


def test_truncated_digest_penalized():
    assert score_digest(GOOD, ACTIVITY, truncated=True)["total"] < score_digest(GOOD, ACTIVITY)["total"]


def test_untruncated_default_unchanged():
    assert score_digest(GOOD, ACTIVITY, truncated=False) == score_digest(GOOD, ACTIVITY)


def test_normal_prose_not_flagged_as_repetitive():
    assert score_digest(GOOD, ACTIVITY)["penalties"] == 0.0


def _digest_with_shared_run(words: int) -> str:
    # a shared run of exactly `words` words between the summary and the one body
    run = "the change introduces a new vector graphic asset for the launcher icon set"
    shared = " ".join(run.split()[:words])
    return f"""# Developer Journal

## Summary

Turbolab cleaned up build warnings; {shared} landed alongside the systemd path fix.

## Per-Repo Activity

### usr-wwelsh/turbolab

Absolute paths now resolve for systemd. {shared} shipped across the same batch.
"""


def test_summary_echo_boundary():
    # real teacher digests reach a 9-word shared run legitimately; the sft3 echoes
    # ran 31, 28 and 10. One word of margin, so pin both sides.
    assert score_digest(_digest_with_shared_run(9), ACTIVITY)["penalties"] == 0.0
    assert score_digest(_digest_with_shared_run(10), ACTIVITY)["penalties"] > 0


YESTERDAY = """# Developer Journal

## Summary

omniMux tightened its search pipeline and excluded unknown-artist downloads.

## Per-Repo Activity

### usr-wwelsh/omniMux

Search now excludes unknown-artist downloads so the library stays clean.
"""


def test_lifted_previous_digest_penalized():
    # Finding 2 (2026-08-24): production prepends yesterday's digest, and a sample
    # lifted its body wholesale (a 47-word run), misattributed another repo's work,
    # and scored 1.000 -- score_digest never saw the preamble, so copying beat
    # every honest generation. Passing the preamble as `previous` must close that.
    lifted = GOOD.replace(
        "Removed build warnings and switched systemd unit resolution to absolute paths.",
        "Search now excludes unknown-artist downloads so the library stays clean.")
    without_prev = score_digest(lifted, ACTIVITY)
    s = score_digest(lifted, ACTIVITY, previous=YESTERDAY)
    assert s["penalties"] > without_prev["penalties"]
    assert s["total"] < without_prev["total"]
    assert s["total"] < score_digest(GOOD, ACTIVITY)["total"]


def test_previous_default_off_is_backward_compatible():
    # ~70 call sites pass only (digest, activity); the preamble path must be opt-in
    assert score_digest(GOOD, ACTIVITY, previous=None) == score_digest(GOOD, ACTIVITY)


def _digest_sharing_previous(run_words: int) -> tuple[str, str]:
    # a shared verbatim run of exactly `run_words` words between yesterday's body
    # and today's body, everything else independently worded
    source = "uploads now retry with exponential backoff after a failed gateway round"
    shared = " ".join(source.split()[:run_words])
    previous = f"""# Developer Journal

## Summary

The media server saw steady progress on transfer robustness overall today.

## Per-Repo Activity

### usr-wwelsh/omniMux

{source} trip behaviour hardened under load.
"""
    digest = f"""# Developer Journal

## Summary

Turbolab cleaned up build warnings and fixed systemd absolute path handling today.

## Per-Repo Activity

### usr-wwelsh/turbolab

Network resilience work: {shared} trip handling got sturdier as well.
"""
    return digest, previous


def test_previous_shared_run_boundary():
    # the same calibrated n=10 margin as the summary echo rule: reworded cross-day
    # prose shares vocabulary but never a 10-word verbatim run
    nine_digest, nine_prev = _digest_sharing_previous(9)
    ten_digest, ten_prev = _digest_sharing_previous(10)
    assert score_digest(nine_digest, ACTIVITY, previous=nine_prev)["penalties"] == 0.0
    assert score_digest(ten_digest, ACTIVITY, previous=ten_prev)["penalties"] > 0
