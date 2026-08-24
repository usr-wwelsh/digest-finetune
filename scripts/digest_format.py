"""Constants mirroring git-digest's main.go prompt construction (sparseCommitThreshold,
maxPatchLines, file cap) so the training prompt/reward exactly match what production shows
the model. Keep these in lockstep with ~/Documents/git-digest/main.go if it changes."""

SPARSE_COMMIT_THRESHOLD = 5
MAX_PATCH_LINES = 5
MAX_PATCH_LINE_CHARS = 300
MAX_FILES = 5

# generated/build-output paths and lockfiles: often single-line-minified (a "5 line" cap
# doesn't bound size when one line is 60k+ chars) and never meaningful project signal either
# way, so their diffs are dropped from prompts and reward scoring alike.
GENERATED_PATH_PARTS = {"dist", "build", "node_modules", "vendor", ".venv", "target"}
BOILERPLATE_BASENAMES = {
    "license", "license.md", "license.txt", "notice",
    "package-lock.json", "yarn.lock", "bun.lock", "bun.lockb", "pnpm-lock.yaml",
    "cargo.lock", "go.sum", "poetry.lock", "uv.lock", "gemfile.lock", "composer.lock",
}


def is_low_signal_file(filename: str) -> bool:
    lower = filename.lower()
    parts = lower.split("/")
    basename = parts[-1]
    if basename in BOILERPLATE_BASENAMES:
        return True
    if any(p in GENERATED_PATH_PARTS for p in parts[:-1]):
        return True
    if basename.endswith((".min.js", ".min.css", ".lock")):
        return True
    return False


def total_commits(activity: dict) -> int:
    return sum(len(data.get("commits", [])) for data in activity.values())


def include_patches(activity: dict) -> bool:
    return total_commits(activity) <= SPARSE_COMMIT_THRESHOLD
