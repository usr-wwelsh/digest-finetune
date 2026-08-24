import re

BANNED = ["next steps", "you should", "todo:", "as an ai"]

STOPWORDS = set("""a an and are as at be been but by can did do does for from had has have how i in into is it its
just like more new not of on or our out over should so some such than that the their them then there these they this
to up use used using was we were what when which while who will with would you your also added fix fixed fixes
update updated updates change changes changed""".split())

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
    data = activity.get(repo) or {}
    msgs = " ".join(c["message"] for c in data.get("commits", []))
    toks = content_tokens(msgs)
    toks.update(content_tokens(repo))
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


def coverage_score(digest: str, repos: list[str]) -> float:
    _, sections = parse_sections(digest)
    seen = {norm_repo(n) for n, _ in sections}
    allowed = {norm_repo(r) for r in repos}
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
        hits.append(min(1.0, len(toks & src) / target))
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
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            if headers[i] != headers[j] and _jaccard(bodies[i], bodies[j]) > 0.6:
                p += 0.3
                break

    return min(1.0, p)


def score_digest(digest: str, activity: dict) -> dict:
    repos = list(activity.keys())
    _, sections = parse_sections(digest)
    f = format_score(digest, repos)
    c = coverage_score(digest, repos)
    g = grounding_score(digest, activity)
    p = penalties(digest, sections, repos)
    total = max(0.0, 0.30 * f + 0.45 * c + 0.25 * g - p)
    return {"format": f, "coverage": c, "grounding": round(g, 3),
            "penalties": p, "total": round(total, 3)}
