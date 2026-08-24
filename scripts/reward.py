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


def _stem(t: str) -> str:
    for suf in ("ing", "ies", "ed", "es", "s"):
        if len(t) > 4 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def content_tokens(text: str) -> set[str]:
    return {_stem(t) for t in WORD_RE.findall(text.lower())
            if len(t) >= 3 and t not in STOPWORDS}


def parse_sections(digest: str) -> tuple[str, list[tuple[str, str]]]:
    summary_lines: list[str] = []
    sections: list[tuple[str, str]] = []
    mode = ""
    cur_name, cur_body = "", []
    for line in digest.splitlines():
        if line.startswith("### "):
            if cur_name:
                sections.append((cur_name, "\n".join(cur_body).strip()))
            cur_name, cur_body = line[4:].strip(), []
            mode = "repo"
        elif line.startswith("## "):
            if cur_name:
                sections.append((cur_name, "\n".join(cur_body).strip()))
                cur_name, cur_body = "", []
            title = line[3:].strip().lower()
            mode = "summary" if title == "summary" else ""
        elif mode == "summary":
            summary_lines.append(line)
        elif mode == "repo":
            cur_body.append(line)
    if cur_name:
        sections.append((cur_name, "\n".join(cur_body).strip()))
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
        toks |= content_tokens(c["message"])
        for f in c.get("files", []):
            if is_low_signal_file(f["filename"]):
                continue
            toks |= content_tokens(f["filename"])
            if show_patches:
                toks |= content_tokens(f.get("patch", ""))
    return toks


def format_score(digest: str, repos: list[str]) -> float:
    summary, sections = parse_sections(digest)
    allowed = {norm_repo(r) for r in repos}
    headers = [norm_repo(n) for n, _ in sections]
    valid_headers = [h for h in headers if h in allowed]

    has_summary = len(summary.split()) >= 10
    has_section_header = "## Per-Repo Activity" in digest
    header_precision = len(valid_headers) / len(headers) if headers else 0.0

    return (has_summary + has_section_header + header_precision) / 3


def coverage_score(digest: str, activity: dict, min_tokens: int = 3, min_overlap: int = 2) -> float:
    """A repo only counts as covered if its section has enough content AND
    at least `min_overlap` words actually traceable to that repo's commits --
    otherwise headers-with-filler-bodies (nonsense, copied SHAs) score as covered."""
    repos = list(activity.keys())
    allowed = {norm_repo(r) for r in repos}
    _, sections = parse_sections(digest)
    seen = set()
    for name, body in sections:
        rn = norm_repo(name)
        if rn not in allowed:
            continue
        match = next(r for r in repos if norm_repo(r) == rn)
        toks = content_tokens(body)
        if len(toks) < min_tokens:
            continue
        if len(toks & source_tokens(activity, match)) < min_overlap:
            continue
        seen.add(rn)
    return len(seen & allowed) / len(allowed) if allowed else 1.0


def grounding_score(digest: str, activity: dict, target: int = 5) -> float:
    _, sections = parse_sections(digest)
    if not sections:
        return 0.0
    hits = []
    for name, body in sections:
        match = next((r for r in activity if norm_repo(r) == norm_repo(name)), None)
        src = source_tokens(activity, match) if match else set()
        toks = content_tokens(body)
        ov = len(toks & src)
        hit = min(1.0, ov / target)
        # a section mostly made of source words is a copy/dump, not prose --
        # verbatim commit lists and keyword dumps otherwise max this score
        if len(toks) >= 6 and ov > 0.6 * len(toks):
            hit *= 0.5
        hits.append(hit)
    return sum(hits) / len(hits)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def penalties(digest: str, sections: list[tuple[str, str]], repos: list[str]) -> float:
    low = digest.lower()
    p = sum(1 for b in BANNED if b in low) * 0.5

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

    return min(1.0, p)


def score_digest(digest: str, activity: dict) -> dict:
    repos = list(activity.keys())
    _, sections = parse_sections(digest)
    f = format_score(digest, repos)
    c = coverage_score(digest, activity)
    g = grounding_score(digest, activity)
    p = penalties(digest, sections, repos)
    total = max(0.0, 0.30 * f + 0.45 * c + 0.25 * g - p)
    return {"format": f, "coverage": c, "grounding": round(g, 3),
            "penalties": p, "total": round(total, 3)}
