import re

from digest_format import include_patches, is_low_signal_file

BANNED = ["next steps", "you should", "todo:", "as an ai"]

STOPWORDS = set("""a an and are as at be been but by can did do does for from had has have how i in into is it its
just like more new not of on or our out over should so some such than that the their them then there these they this
to up use used using was we were what when which while who will with would you your also added fix fixed fixes
update updated updates change changes changed""".split())

# generic package-manifest keys (package.json etc.) that leak into diff text on init/scaffold
# commits -- guessable for any JS/TS repo regardless of what the project actually does, so they
# don't count as evidence a claim is grounded.
MANIFEST_STOPWORDS = set("""name version private public main module modules scripts type types author
engines description dependencies devdependencies peerdependencies license""".split())
STOPWORDS |= MANIFEST_STOPWORDS

WORD_RE = re.compile(r"[a-z0-9_]+")

# code-file extensions worth fact-checking mentions of; deliberately narrow so we don't
# flag prose like version numbers, IPs, or "e.g." as filenames
CODE_EXTENSIONS = {
    "js", "jsx", "ts", "tsx", "py", "go", "rs", "c", "h", "cpp", "hpp", "java", "kt", "kts",
    "svelte", "vue", "json", "yaml", "yml", "md", "css", "html", "xml", "sh", "toml", "sql",
    "rb", "php", "dart", "gradle", "cfg", "ini", "env", "proto",
}
# common tech/framework/tool names that look like a "file.ext" mention but aren't a file claim
NON_FILE_DOTTED_NAMES = {
    "node.js", "vue.js", "react.js", "next.js", "nuxt.js", "d3.js", "three.js",
    "express.js", "socket.io", "chart.js", "vite.js", "llama.cpp",
}
FILE_MENTION_RE = re.compile(r"\b[\w][\w/-]*\.([A-Za-z]{1,10})\b")


def real_filenames(activity: dict) -> set[str]:
    """Filenames the model could plausibly cite: diffed files, plus any filename-shaped
    token in a commit message -- messages are shown in the prompt same as file stats, so
    e.g. a message naming a generated output file (not itself part of the diff) is grounded."""
    names = set()
    for data in activity.values():
        for c in data.get("commits", []):
            for f in c.get("files", []):
                name = f.get("filename", "")
                if name:
                    names.add(name.strip().lower().split("/")[-1])
            for m in FILE_MENTION_RE.finditer(c.get("message", "")):
                if m.group(1).lower() in CODE_EXTENSIONS:
                    names.add(m.group(0).lower().split("/")[-1])
    return names


def fabricated_file_mentions(body: str, real: set[str]) -> int:
    count = 0
    for m in FILE_MENTION_RE.finditer(body):
        if m.group(1).lower() not in CODE_EXTENSIONS:
            continue
        mentioned = m.group(0).lower().split("/")[-1]
        if mentioned in NON_FILE_DOTTED_NAMES or mentioned in real:
            continue
        count += 1
    return count


def _stem(t: str) -> str:
    for suf in ("ing", "ies", "ed", "es", "s"):
        if len(t) > 4 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def content_tokens(text: str) -> set[str]:
    return {_stem(t) for t in WORD_RE.findall(text.lower())
            if len(t) >= 3 and t not in STOPWORDS}


def parse_sections(digest: str) -> tuple[str, list[tuple[str, str]]]:
    """Returns (summary, sections). Prose under "## Per-Repo Activity" before the first
    "###" header becomes a pseudo-section named "" so it can't hide from scoring -- it
    fails header_precision, trips the excess-header penalty, and its body is scanned
    for fabricated files like any other."""
    summary_lines: list[str] = []
    sections: list[tuple[str, str]] = []
    mode = ""
    seen_per_repo = False
    orphan: list[str] = []
    cur_name, cur_body = "", []
    for line in digest.splitlines():
        if line.startswith("### "):
            if cur_name:
                sections.append((cur_name, "\n".join(cur_body).strip()))
            elif orphan:
                sections.append(("", "\n".join(orphan).strip()))
                orphan = []
            cur_name, cur_body = line[4:].strip(), []
            mode = "repo"
        elif line.startswith("## "):
            if cur_name:
                sections.append((cur_name, "\n".join(cur_body).strip()))
                cur_name, cur_body = "", []
            title = line[3:].strip().lower()
            mode = "summary" if title == "summary" else ""
            seen_per_repo = title == "per-repo activity"
            orphan = []
        elif mode == "summary":
            summary_lines.append(line)
        elif mode == "repo":
            cur_body.append(line)
        elif seen_per_repo and line.strip():
            orphan.append(line)
    if cur_name:
        sections.append((cur_name, "\n".join(cur_body).strip()))
    elif orphan:
        sections.append(("", "\n".join(orphan).strip()))
    return "\n".join(summary_lines).strip(), sections


def norm_repo(name: str) -> str:
    return name.strip().lower().split("/")[-1]


def source_tokens(activity: dict, repo: str) -> set[str]:
    """Tokens the model could plausibly have gotten from ITS OWN prompt for this repo:
    commit messages + filenames always, diff content only on sparse-commit days -- must
    track build_prompt()'s include_patches() decision exactly, or grounding ends up
    verifying claims against information the model was never actually shown."""
    data = activity.get(repo) or {}
    show_patches = include_patches(activity)
    toks = content_tokens(repo)
    for c in data.get("commits", []):
        toks |= content_tokens(c.get("message", ""))
        for f in c.get("files", []):
            fname = f.get("filename", "")
            if not fname or is_low_signal_file(fname):
                continue
            toks |= content_tokens(fname)
            if show_patches:
                toks |= content_tokens(f.get("patch", ""))
    return toks


def format_score(digest: str, repos: list[str], activity: dict) -> float:
    """The summary gate checks groundedness too: a fluent-but-unrelated summary is the
    cheapest fabrication channel in the format (never verified otherwise), so require
    content traceable to ANY repo's own prompt material."""
    summary, sections = parse_sections(digest)
    allowed = {norm_repo(r) for r in repos}
    headers = [norm_repo(n) for n, _ in sections]
    valid_headers = [h for h in headers if h in allowed]

    summary_toks = content_tokens(summary)
    any_source: set[str] = set()
    for r in repos:
        any_source |= source_tokens(activity, r)
    has_summary = len(summary.split()) >= 10 and len(summary_toks & any_source) >= 2
    has_section_header = "## Per-Repo Activity" in digest
    header_precision = len(valid_headers) / len(headers) if headers else 0.0

    return (has_summary + has_section_header + header_precision) / 3


def coverage_score(digest: str, activity: dict, min_tokens: int = 3, target_overlap: int = 4) -> float:
    """A repo only counts as covered if its section has enough content AND content
    words actually traceable to that repo's commits. Credit is continuous in the
    overlap size (saturating at `target_overlap`): a binary gate let a body built
    from two generic words that happened to appear in commit messages ("cleanup",
    "scroll") bank full credit -- exactly the gradient an RL run climbs."""
    repos = list(activity.keys())
    allowed = {norm_repo(r) for r in repos}
    _, sections = parse_sections(digest)
    credit: dict[str, float] = {}
    for name, body in sections:
        rn = norm_repo(name)
        if rn not in allowed:
            continue
        match = next(r for r in repos if norm_repo(r) == rn)
        toks = content_tokens(body)
        if len(toks) < min_tokens:
            continue
        ov = len(toks & source_tokens(activity, match))
        if ov < 2:
            continue
        # expected overlap scales with how much the body actually says: a tight
        # 3-word paraphrase of a 1-commit repo keeps full credit, while a long
        # hollow body can't buy saturation with two lucky generic words
        eff = max(2, min(target_overlap, len(toks) * 3 // 4))
        credit[rn] = max(credit.get(rn, 0.0), min(1.0, ov / eff))
    return sum(credit.values()) / len(allowed) if allowed else 1.0


def grounding_score(digest: str, activity: dict, target: int = 5) -> float:
    """Grounding averages over EVERY expected repo, not just sections present: a
    missing (or content-free) section scores 0 for that repo. Averaging over present
    sections only let omitting a weak section raise the mean (+0.038 total on a real
    day) -- quiet omission must never pay."""
    if not activity:
        return 0.0
    _, sections = parse_sections(digest)
    hits = {norm_repo(r): 0.0 for r in activity}
    for name, body in sections:
        rn = norm_repo(name)
        match = next((r for r in activity if norm_repo(r) == rn), None)
        if rn not in hits:
            continue
        toks = content_tokens(body)
        ov = len(toks & source_tokens(activity, match))
        hit = min(1.0, ov / target)
        # a section mostly made of source words is a copy/dump, not prose --
        # verbatim commit lists and keyword dumps otherwise max this score
        if len(toks) >= 6 and ov > 0.6 * len(toks):
            hit *= 0.5
        hits[rn] = max(hits[rn], hit)
    return sum(hits.values()) / len(hits)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _repetition_ratio(text: str) -> float:
    """Distinct word trigrams over total. 1.0 is ordinary prose, near 0 is a loop.
    content_tokens() is a set, so a body that repeats one fragment until the token
    cap used to look identical to a terse one -- this is the only signal that sees
    the repetition. Real digests bottom out at 0.977; a looped body scores ~0.05."""
    words = WORD_RE.findall(text.lower())
    if len(words) < 12:
        return 1.0
    grams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return len(set(grams)) / len(grams)


def _shares_long_span(a: str, b: str, n: int = 10) -> bool:
    """True if a and b share a run of n words. Reworded prose does not.

    n=10 is measured, not guessed: across the 109 real rows the longest shared
    summary/body run is 9 words (2026-08-14-synth5), while the sft3 echoes run
    31, 28, and 10. The margin is one word, so test_summary_echo_boundary pins
    both sides -- widening this rule starts penalising real teacher digests."""
    wa, wb = WORD_RE.findall(a.lower()), WORD_RE.findall(b.lower())
    if len(wa) < n or len(wb) < n:
        return False
    grams = {tuple(wa[i:i + n]) for i in range(len(wa) - n + 1)}
    return any(tuple(wb[i:i + n]) in grams for i in range(len(wb) - n + 1))


def penalties(digest: str, summary: str, sections: list[tuple[str, str]], repos: list[str], activity: dict) -> float:
    low = digest.lower()
    p = sum(1 for b in BANNED if b in low) * 0.5

    real = real_filenames(activity)
    # scan the summary too -- a hallucinated filename there used to be free
    fabricated = fabricated_file_mentions(summary, real) \
        + sum(fabricated_file_mentions(b, real) for _, b in sections)
    p += min(0.5, fabricated * 0.3)

    headers = [n for n, _ in sections]
    if len(set(headers)) < len(headers):
        p += 0.5

    allowed = {norm_repo(r) for r in repos}
    excess = max(0, len(headers) - len(allowed))
    p += min(0.5, excess * 0.15)

    bodies = [content_tokens(b) for _, b in sections]
    near_dup = False
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            # short sections on big days legitimately share words; only flag
            # substantial bodies, and charge once no matter how many pairs
            if min(len(bodies[i]), len(bodies[j])) >= 6 and _jaccard(bodies[i], bodies[j]) > 0.6:
                near_dup = True
                break
        if near_dup:
            break
    if near_dup:
        p += 0.3

    # a body that restates the summary was free while only bodies were compared
    # pairwise. Jaccard cannot separate this: a solo-repo day legitimately rewords
    # the summary into its one body at 0.615, barely under a copy-paste at 0.667.
    # Verbatim span reuse is the honest signal -- rewording never keeps 8 words in
    # a row, and copy-paste keeps whole sentences.
    if any(_shares_long_span(summary, b) for _, b in sections):
        p += 0.3

    # charge self-repetition once, however many sections loop
    if any(_repetition_ratio(t) < 0.5 for t in [summary] + [b for _, b in sections]):
        p += 0.4

    return min(1.0, p)


def score_digest(digest: str, activity: dict, truncated: bool = False) -> dict:
    repos = list(activity.keys())
    summary, sections = parse_sections(digest)
    f = format_score(digest, repos, activity)
    c = coverage_score(digest, activity)
    g = grounding_score(digest, activity)
    p = penalties(digest, summary, sections, repos, activity)
    # running into the token cap means the digest never finished; callers that
    # know the generation length pass this in, and it defaults off for scoring
    # reference completions that were never generated
    if truncated:
        p = min(1.0, p + 0.5)
    total = max(0.0, 0.30 * f + 0.45 * c + 0.25 * g - p)
    return {"format": f, "coverage": c, "grounding": round(g, 3),
            "penalties": p, "total": round(total, 3)}
