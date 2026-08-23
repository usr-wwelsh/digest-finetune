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
    if len(summary.split()) < 10:
        return 0.0
    if "## Per-Repo Activity" not in digest:
        return 0.0
    headers = [norm_repo(n) for n, _ in sections]
    allowed = {norm_repo(r) for r in repos}
    if not headers or any(h not in allowed for h in headers):
        return 0.0
    return 1.0


def coverage_score(digest: str, repos: list[str]) -> float:
    _, sections = parse_sections(digest)
    seen = {norm_repo(n) for n, _ in sections}
    allowed = {norm_repo(r) for r in repos}
    return len(seen & allowed) / len(allowed) if allowed else 1.0


def grounding_score(digest: str, activity: dict) -> float:
    _, sections = parse_sections(digest)
    if not sections:
        return 0.0
    ratios = []
    for name, body in sections:
        match = next((r for r in activity if norm_repo(r) == norm_repo(name)), None)
        src = source_tokens(activity, match) if match else set()
        toks = content_tokens(body)
        if not toks:
            ratios.append(0.0)
            continue
        ratios.append(len(toks & src) / len(toks))
    return sum(ratios) / len(ratios)


def repetition_score(digest: str) -> float:
    sentences = [s.strip().lower() for s in re.split(r"[.!?\n]", digest)
                 if len(s.split()) >= 4]
    if len(sentences) < 2:
        return 1.0
    return len(set(sentences)) / len(sentences)


def penalties(digest: str) -> float:
    low = digest.lower()
    return min(1.0, sum(1 for p in BANNED if p in low) * 0.5)


def score_digest(digest: str, activity: dict) -> dict:
    repos = list(activity.keys())
    f = format_score(digest, repos)
    c = coverage_score(digest, repos)
    g = grounding_score(digest, activity)
    r = repetition_score(digest)
    p = penalties(digest)
    total = max(0.0, 0.30 * f + 0.20 * c + 0.35 * g + 0.15 * r - p)
    return {"format": f, "coverage": c, "grounding": round(g, 3),
            "repetition": round(r, 3), "penalties": p, "total": round(total, 3)}
