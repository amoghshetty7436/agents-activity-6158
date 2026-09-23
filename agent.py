#!/usr/bin/env python3
"""Python -> Rust translation agent.

    python agent.py                  # run with defaults
    python agent.py --budget 40      # cap on model calls (graded: do not raise)

Provider: any OpenAI-compatible chat endpoint. Defaults to Google Gemini
(free tier). Configure with environment variables (or a local .env file):

    GEMINI_API_KEY=...        # or GOOGLE_API_KEY / OPENAI_API_KEY / LLM_API_KEY
    LLM_MODEL=gemini-2.5-flash
    LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai

The design notes for the five TODOs are in REPORT.md.
"""
from __future__ import annotations
import argparse, json, os, pathlib, subprocess, time, sys, ssl, re, urllib.request, urllib.error, hashlib

HERE   = pathlib.Path(__file__).parent
RUST   = HERE / "rust"
LIB    = RUST / "src" / "lib.rs"
PYSRC  = HERE / "reference" / "version.py"
LOGS   = HERE / "logs"

# --------------------------------------------------------------------- config
def _load_dotenv() -> None:
    """Minimal .env loader so we need no extra dependency."""
    env = HERE / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_dotenv()

BASE_URL = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
MODEL    = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
API_KEY  = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY") or "")

# macOS python.org builds ship without a wired-up CA bundle; use certifi's.
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = ssl.create_default_context()

# Shared run state the loop and should_stop() read. Kept out of the model's
# context on purpose (this is the WRITE lever from TODO 3).
STATE = {
    "score": None,          # rank() of the latest evaluation
    "metrics": None,        # parsed evaluate.py JSON of the latest evaluation
    "best_score": -1e9,
    "best_src": None,       # lib.rs text that produced best_score
    "evals": 0,
    "score_history": [],    # rank after each evaluation
    "no_call_turns": 0,     # consecutive model turns with no tool call
    "last_action": None,    # (name, args-hash) of the previous single tool call
    "repeat_count": 0,
    "oracle_calls": 0,      # cap speculative probing so the agent writes code
}
ORACLE_BUDGET = 8

# ============================================================== TODO 1
def call_model(messages: list[dict], tools: list[dict]) -> dict:
    """Send `messages` + `tools` to an OpenAI-compatible model; return its reply.

    Return shape expected by the loop:
        {"text": str | None,
         "tool_calls": [{"id": str, "name": str, "arguments": dict}, ...]}
    """
    if not API_KEY:
        sys.exit("No API key. Put GEMINI_API_KEY=... in agents-activity-6158/.env "
                 "(see the header of this file).")

    wire = _to_openai_messages(messages)
    body = {
        "model": MODEL,
        "messages": wire,
        "tools": [{"type": "function", "function": t} for t in tools],
        "tool_choice": "auto",
        "temperature": 0.1,
        "parallel_tool_calls": False,
    }
    data = json.dumps(body).encode()
    url = f"{BASE_URL}/chat/completions"

    last_err = None
    for attempt in range(6):
        req = urllib.request.Request(url, data=data, method="POST", headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        })
        try:
            with urllib.request.urlopen(req, timeout=180, context=_SSL_CTX) as r:
                resp = json.load(r)
            msg = resp["choices"][0]["message"]
            calls = []
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function", {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                # preserve Gemini's thought_signature (extra_content); the API
                # rejects a later turn whose tool call is missing it.
                calls.append({"id": tc.get("id") or f"call_{len(calls)}",
                              "name": fn.get("name", ""), "arguments": args or {},
                              "extra": tc.get("extra_content")})
            return {"text": msg.get("content"), "tool_calls": calls}
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            last_err = f"HTTP {e.code}: {detail}"
            if e.code in (429, 500, 502, 503, 529):      # transient / rate limit
                # honor the API's "retry in Ns" hint (per-minute limits), capped
                m = re.search(r"retry in ([0-9.]+)s", detail)
                wait = min(float(m.group(1)) + 2, 65) if m else min(2 ** attempt * 3, 45)
                time.sleep(wait)
                continue
            break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = str(e)
            time.sleep(min(2 ** attempt * 2, 30))
            continue
    sys.exit(f"call_model failed after retries: {last_err}")

def _to_openai_messages(messages: list[dict]) -> list[dict]:
    """Translate the loop's internal history into OpenAI wire format,
    preserving the tool_call <-> tool_result linkage by id."""
    out = []
    for m in messages:
        role = m["role"]
        if role == "assistant":
            entry = {"role": "assistant", "content": m.get("content") or ""}
            tcs = m.get("tool_calls") or []
            if tcs:
                entry["tool_calls"] = [{
                    "id": c["id"], "type": "function",
                    "function": {"name": c["name"],
                                 "arguments": json.dumps(c.get("arguments") or {})},
                    **({"extra_content": c["extra"]} if c.get("extra") else {}),
                } for c in tcs]
            out.append(entry)
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("id", ""),
                        "content": m.get("content", "")})
        else:                                              # system / user
            out.append({"role": role, "content": m.get("content", "")})
    return out

# ============================================================== TODO 2
def system_prompt() -> str:
    return """\
You are a meticulous compiler engineer migrating a Python module to Rust.

GOAL
Translate reference/version.py (a SemVer 2.0.0 implementation) into
rust/src/lib.rs. It must expose EXACTLY these items and compile with `cargo
build --release`:

    pub struct Version { pub major: u64, pub minor: u64, pub patch: u64,
                         pub prerelease: Option<String>, pub build: Option<String> }
    pub fn parse(s: &str) -> Result<Version, String>
    pub fn to_string(v: &Version) -> String
    pub fn compare(a: &Version, b: &Version) -> std::cmp::Ordering
    pub fn bump_major(v: &Version) -> Version
    pub fn bump_minor(v: &Version) -> Version
    pub fn bump_patch(v: &Version) -> Version

You may add private helpers and #[cfg(test)] tests. Do NOT change these signatures.
src/main.rs is fixed and calls these; never mention or edit it.

HARD RULES (evaluate.py fails you for breaking them):
  * no `unsafe`
  * no external crates - std only
  * no `todo!()`, `unimplemented!()`, or `panic!` (a real translation cannot panic)
  * do not call back into Python from Rust
  * avoid gratuitous `.clone()` / `.unwrap()`; own a `String` instead of cloning a
    borrow, and handle errors with `Result`/`match` instead of unwrapping.

SEMANTICS THAT FIRST DRAFTS GET WRONG (verify each with the `oracle` tool):
  * parse requires full major.minor.patch. Each is `0` or a non-zero digit with
    no leading zero -> "01.0.0" is INVALID. Store them as u64.
  * prerelease is the text after `-`; build is the text after `+`. `+` starts build.
  * a prerelease identifier is `[0-9A-Za-z-]+`; a purely-numeric identifier may not
    have a leading zero ("01" invalid, "0" ok), but one containing a letter/hyphen
    may ("0A" ok). Empty identifiers are invalid ("1.0.0-alpha..1", "1.0.0-").
  * a build identifier is `[0-9A-Za-z-]+` with NO leading-zero restriction ("01" ok).
  * PRECEDENCE: build metadata is IGNORED entirely. Compare (major,minor,patch)
    numerically first. If equal, a version WITH a prerelease is LOWER than the same
    version WITHOUT one. Otherwise compare prerelease identifiers left-to-right,
    dot-separated: numeric identifiers compare numerically and rank BELOW any
    non-numeric identifier; non-numeric compare by ASCII; if all shared identifiers
    are equal, the one with MORE identifiers is greater.
  * to_string: "major.minor.patch", append "-prerelease" if present, then "+build"
    if present. parse(to_string(v)) must round-trip.
  * bump_major -> (major+1, 0, 0), bump_minor -> (major, minor+1, 0),
    bump_patch -> (major, minor, patch+1). ALL bumps DROP prerelease AND build.

WORKFLOW  (model calls are scarce - be decisive, aim to finish in under ~12 calls)
1. The spec above is authoritative and already verified against the oracle, so
   you do NOT need to explore first. Optionally read_python ONCE if you want to
   see the grammar; do not read it twice.
2. write_rust with the COMPLETE file right away, implementing every function and
   a `#[cfg(test)] mod tests` block. Do not send a partial file.
3. cargo_build; fix every error (edit_rust for small fixes, write_rust for large).
4. cargo_test, then evaluate.
5. ONLY for a differential case you actually got wrong, call `oracle` to see the
   correct answer, then fix the real cause. Do not call oracle speculatively.
6. Stop when evaluate shows 100% differential, the spec chain passes, cargo tests
   pass, and zero violations - then say "DONE" with no tool call. Do not keep
   editing clean, passing code.

Write real Rust unit tests porting cases from reference/test_*.py (parse, compare,
bump). Be terse; let the tools do the talking - prose does not score."""

# ============================================================== TODO 3
# WRITE + SELECT + COMPRESS. Big, stale artifacts (old compiler dumps, old
# whole-file writes, repeated source reads) are the thing that fills the window
# and makes the agent repeat itself. We keep the system prompt, the task, and
# the last few turns verbatim, and shrink everything older to a short stub -
# the current file and source are always one tool call away, so no information
# is truly lost.
KEEP_RECENT_TURNS = 6
def build_context(history: list[dict], step: int) -> list[dict]:
    if len(history) <= 2:
        return history
    head = history[:2]                      # system + original task
    body = history[2:]

    # index of the message that begins the most-recent KEEP_RECENT_TURNS
    # assistant turns; everything before it is eligible for compression.
    assistant_idx = [i for i, m in enumerate(body) if m["role"] == "assistant"]
    cut = assistant_idx[-KEEP_RECENT_TURNS] if len(assistant_idx) > KEEP_RECENT_TURNS else 0

    out = []
    for i, m in enumerate(body):
        if i >= cut:
            out.append(m)
            continue
        out.append(_compress(m))
    return head + out

def _compress(m: dict) -> dict:
    """Shrink one old message, preserving role and tool-call structure/ids."""
    if m["role"] == "assistant":
        tcs = m.get("tool_calls") or []
        slim = []
        for c in tcs:
            args = c.get("arguments") or {}
            if c["name"] in ("write_rust",) and "content" in args:
                args = {**args, "content": f"<{len(str(args['content']))} bytes omitted>"}
            elif c["name"] == "edit_rust":
                args = {"find": "<omitted>", "replace": "<omitted>"}
            slim.append({**c, "arguments": args})
        return {**m, "content": (m.get("content") or "")[:200], "tool_calls": slim}
    if m["role"] == "tool":
        c = m.get("content", "")
        if len(c) > 240:
            c = c[:200] + f" ... <{len(c)} chars, re-run the tool if you need it again>"
        return {**m, "content": c}
    return m

# ============================================================== TODO 4
NO_PROGRESS_PATIENCE = 6        # evals without beating the best score
def should_stop(history, step, budget, last_score) -> tuple[bool, str]:
    if step >= budget:
        return True, f"budget exhausted ({budget} model calls)"

    m = STATE["metrics"]
    # Done: the file is correct AND clean. Believe the score, not the model.
    if (m and m.get("build") and not m.get("violations")
            and m.get("spec_precedence_chain")
            and m.get("differential_pct", 0) >= 99.9
            and (m.get("cargo_test") or {}).get("failed", 1) == 0):
        return True, "task complete: 100% differential, spec chain + tests pass, no violations"

    # Stuck: same single action repeated with no new information.
    if STATE["repeat_count"] >= 3:
        return True, "stuck: identical tool call repeated 3x"

    # Idle: the model stopped calling tools and is just chatting.
    if STATE["no_call_turns"] >= 2:
        return True, "model idle: no tool call for 2 turns (treated as give-up)"

    # No progress: enough evaluations have run without ever beating the best.
    hist = STATE["score_history"]
    if STATE["evals"] >= NO_PROGRESS_PATIENCE and len(hist) >= NO_PROGRESS_PATIENCE:
        recent = hist[-NO_PROGRESS_PATIENCE:]
        if max(recent) <= STATE["best_score"] - 1e-9 or max(recent) == min(recent):
            # allow it to keep going only if it has never built successfully
            if STATE["best_score"] > -1e9:
                return True, f"no progress in {NO_PROGRESS_PATIENCE} evaluations (best kept)"
    return False, ""

# ================================================================== tools
def _run(cmd, cwd=None, timeout=180):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout + p.stderr).strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"TIMEOUT after {timeout}s"

def _tail(text: str, n: int) -> str:
    lines = text.splitlines()
    return text if len(lines) <= n else "... (trimmed) ...\n" + "\n".join(lines[-n:])

def t_read_python(_args):
    return PYSRC.read_text() if PYSRC.exists() else "reference/version.py missing - run fetch_source.py"

def t_read_rust(_args):
    return LIB.read_text()

def t_write_rust(args):
    """Overwrite rust/src/lib.rs. `content` must be the WHOLE file."""
    content = args.get("content", "")
    if not content.strip():
        return "refused: empty content"
    LIB.write_text(content)
    return f"wrote {len(content)} bytes to rust/src/lib.rs"

def t_edit_rust(args):
    """Replace the first exact occurrence of `find` with `replace` in lib.rs.
    Cheaper than rewriting the whole file for a small fix."""
    find, repl = args.get("find", ""), args.get("replace", "")
    if not find:
        return "refused: `find` is empty"
    src = LIB.read_text()
    n = src.count(find)
    if n == 0:
        return "no match for `find`; read_rust and copy the exact text (whitespace matters)"
    if n > 1:
        return f"`find` matches {n} places; make it more specific so it is unique"
    LIB.write_text(src.replace(find, repl, 1))
    return "applied edit"

def t_cargo_build(_args):
    return _tail(_run(["cargo", "build", "--release"], cwd=RUST), 60)

def t_cargo_test(_args):
    return _tail(_run(["cargo", "test", "--release"], cwd=RUST), 40)

def t_oracle(args):
    """Reference behaviour of the real `semver` package. Development aid only -
    the Rust must not call Python. op in {parse, compare, bump, format}."""
    try:
        import semver
    except ImportError:
        return "oracle unavailable: `pip install -r requirements.txt`"
    STATE["oracle_calls"] += 1
    if STATE["oracle_calls"] > ORACLE_BUDGET:
        return ("oracle budget spent - the spec in your instructions is authoritative. "
                "Stop probing and write/fix rust/src/lib.rs now.")
    op = (args.get("op") or "").strip()
    try:
        if op == "parse":
            v = semver.Version.parse(args["version"])
            return json.dumps(v.to_dict())
        if op == "format":
            return str(semver.Version.parse(args["version"]))
        if op == "compare":
            return str(semver.Version.parse(args["a"]).compare(args["b"]))
        if op == "bump":
            kind = args["kind"]
            return str(getattr(semver.Version.parse(args["version"]), f"bump_{kind}")())
        return "unknown op; use parse|compare|bump|format"
    except (ValueError, TypeError) as e:
        return f"INVALID / raises: {type(e).__name__}: {e}"
    except KeyError as e:
        return f"missing argument: {e}"

def _rank(m) -> float:
    if not m or not m.get("build"):
        return -1e9
    base = float(m.get("differential_pct", 0.0))
    if m.get("violations"):
        base -= 1000.0                     # any clean build beats any violating build
    return base

def t_evaluate(_args):
    """Differential evaluation on the practice seed. Returns a compact summary
    and records structured metrics for the loop (score, rollback, stopping)."""
    tmp = LOGS / "_eval.json"
    proc = subprocess.run([sys.executable, str(HERE / "evaluate.py"),
                           "--n", "100", "--json", str(tmp)],
                          cwd=HERE, capture_output=True, text=True, timeout=240)
    stdout = proc.stdout + proc.stderr
    try:
        R = json.loads(tmp.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return "evaluate.py produced no JSON:\n" + _tail(stdout, 40)

    STATE["metrics"] = R
    STATE["score"] = _rank(R)
    STATE["evals"] += 1
    STATE["score_history"].append(STATE["score"])

    if not R.get("build"):
        return "BUILD FAILED - fix compiler errors first:\n" + _tail(stdout, 40)

    lines = [f"build: PASS   differential: {R.get('differential_pct')}%   "
             f"spec_chain: {'PASS' if R.get('spec_precedence_chain') else 'FAIL'}"]
    ct = R.get("cargo_test")
    if ct:
        lines.append(f"cargo test: {ct['passed']} passed, {ct['failed']} failed")
    for name, d in (R.get("differential") or {}).items():
        lines.append(f"  {name:<15} {d['pass']}/{d['total']}")
    q = R.get("quality", {})
    lines.append(f"quality: unsafe={q.get('unsafe_blocks')} clone={q.get('clone_calls')} "
                 f"unwrap={q.get('unwrap_calls')} todo={q.get('todo_macros')} "
                 f"panic={q.get('panic_macros')} deps={q.get('extra_dependencies')}")
    if R.get("violations"):
        lines.append(f"RULE VIOLATIONS: {', '.join(R['violations'])} -- fix these, they override correctness")
    # sample failures, extracted from the human report
    if "First failures:" in stdout:
        tail = stdout.split("First failures:", 1)[1].strip().splitlines()
        picked = [l for l in tail if l.strip().startswith("-")][:10]
        if picked:
            lines.append("failing cases:")
            lines.extend("  " + l.strip() for l in picked)
    return "\n".join(lines)

# TODO 5: finer actions than the skeleton's coarse set. `edit_rust` avoids
# rewriting (and re-tokenising) the whole file for a one-line fix; `oracle`
# lets the agent confirm semantics instead of guessing; `evaluate` returns a
# compact structured summary instead of a wall of text.
TOOLS = [
    dict(name="read_python", description="Read reference/version.py, the source to translate.",
         parameters={"type": "object", "properties": {}}, fn=t_read_python),
    dict(name="read_rust", description="Read the current rust/src/lib.rs.",
         parameters={"type": "object", "properties": {}}, fn=t_read_rust),
    dict(name="write_rust", description="Overwrite rust/src/lib.rs with the complete file contents.",
         parameters={"type": "object", "required": ["content"],
                     "properties": {"content": {"type": "string", "description": "the WHOLE file"}}},
         fn=t_write_rust),
    dict(name="edit_rust", description="Replace the first exact occurrence of `find` with `replace` in lib.rs. Use for small fixes.",
         parameters={"type": "object", "required": ["find", "replace"],
                     "properties": {"find": {"type": "string"}, "replace": {"type": "string"}}},
         fn=t_edit_rust),
    dict(name="cargo_build", description="Compile the crate (release). Returns compiler errors.",
         parameters={"type": "object", "properties": {}}, fn=t_cargo_build),
    dict(name="cargo_test", description="Run the crate's own #[cfg(test)] tests.",
         parameters={"type": "object", "properties": {}}, fn=t_cargo_test),
    dict(name="oracle", description="Reference behaviour of the real semver package (dev aid; the Rust must not call Python). "
                                    "Args: op=parse|compare|bump|format, plus version / a,b / kind,version.",
         parameters={"type": "object", "required": ["op"],
                     "properties": {"op": {"type": "string"}, "version": {"type": "string"},
                                    "a": {"type": "string"}, "b": {"type": "string"},
                                    "kind": {"type": "string"}}}, fn=t_oracle),
    dict(name="evaluate", description="Run differential evaluation (practice seed). Returns a compact score summary.",
         parameters={"type": "object", "properties": {}}, fn=t_evaluate),
]
BY_NAME = {t["name"]: t for t in TOOLS}
SCHEMAS = [{k: t[k] for k in ("name", "description", "parameters")} for t in TOOLS]

# =================================================================== loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=40, help="max model calls (graded cap: 40)")
    ap.add_argument("--task", default="Translate reference/version.py into rust/src/lib.rs. "
                                       "Follow the workflow; stop when it is correct and clean.")
    a = ap.parse_args()

    LOGS.mkdir(exist_ok=True)
    log = LOGS / f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    def rec(**kw):
        with log.open("a") as f:
            f.write(json.dumps({"t": time.time(), **kw}) + "\n")

    history = [{"role": "system", "content": system_prompt()},
               {"role": "user",   "content": a.task}]
    rec(event="start", model=MODEL, base_url=BASE_URL, budget=a.budget, task=a.task)

    step, last_score = 0, None
    while True:
        stop, why = should_stop(history, step, a.budget, last_score)
        if stop:
            print(f"\n[stop] {why}")
            rec(event="stop", reason=why, steps=step, best_score=STATE["best_score"])
            break

        step += 1
        reply = call_model(build_context(history, step), SCHEMAS)
        rec(event="model", step=step, reply=reply)

        if reply.get("text"):
            print(f"[{step}] {reply['text'][:200]}")
        history.append({"role": "assistant", "content": reply.get("text") or "",
                        "tool_calls": reply.get("tool_calls", [])})

        calls = reply.get("tool_calls") or []
        if not calls:
            STATE["no_call_turns"] += 1
            continue
        STATE["no_call_turns"] = 0

        # repeated-identical-action detection (single-call turns)
        if len(calls) == 1:
            sig = (calls[0]["name"],
                   hashlib.md5(json.dumps(calls[0].get("arguments") or {}, sort_keys=True).encode()).hexdigest())
            STATE["repeat_count"] = STATE["repeat_count"] + 1 if sig == STATE["last_action"] else 0
            STATE["last_action"] = sig
        else:
            STATE["repeat_count"] = 0
            STATE["last_action"] = None

        for c in calls:
            tool = BY_NAME.get(c["name"])
            out = (tool["fn"](c.get("arguments") or {}) if tool
                   else f"unknown tool {c['name']!r}")
            print(f"      -> {c['name']}: {str(out).splitlines()[0][:120] if out else ''}")
            rec(event="tool", step=step, name=c["name"], args=c.get("arguments"), output=str(out)[:4000])
            history.append({"role": "tool", "id": c["id"], "name": c["name"], "content": str(out)})

        # track the best-scoring lib.rs and keep it (rollback on regression)
        if STATE["score"] is not None and STATE["score"] > STATE["best_score"]:
            STATE["best_score"] = STATE["score"]
            STATE["best_src"] = LIB.read_text()
            last_score = STATE["score"]

    # restore the best version we ever saw, so a late regression cannot cost us
    if STATE["best_src"] is not None and LIB.read_text() != STATE["best_src"]:
        LIB.write_text(STATE["best_src"])
        print(f"[restore] reverted lib.rs to best-scoring version (rank {STATE['best_score']:.1f})")
        rec(event="restore", best_score=STATE["best_score"])

    print(f"\ntrajectory: {log}")
    print("final score:")
    subprocess.run([sys.executable, str(HERE / "evaluate.py")], cwd=HERE)

if __name__ == "__main__":
    main()
