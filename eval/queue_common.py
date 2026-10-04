import os, re

# Defaults follow env.example.sh; set the environment variables to override.
WORK = os.environ.get("WORK", "work")
RUNS = os.environ.get("HN04B_RUNS", os.path.join(WORK, "runs"))


def slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def tag_of(j, prompt_name):
    """Run directory name for a job row (model, source, lm_path) and a prompt file stem."""
    return "__".join([j["model"], slug(j["source"]), os.path.basename(j["lm_path"]).replace(".gguf", ""), prompt_name])
