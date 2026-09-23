# REPORT — Python → Rust translation agent

**Module:** `reference/version.py` (python-semver) → `rust/src/lib.rs`
**Model:** Google `gemini-3.5-flash-lite` (free tier) via the OpenAI-compatible API
**Run:** 5 model calls, stopped on "task complete" — `logs/run-20260923-123505.jsonl`

## 1. Final score, and where it lost points

Practice seed (`python evaluate.py`, seed 0, n=300):

| Gate | Result |
|---|---|
| cargo build | **PASS** |
| cargo test | **4 passed, 0 failed** |
| differential | **100.0%** (2425/2425) |
| semver.org precedence chain | **PASS** |
| quality | unsafe 0 · clone 0 · to_owned 0 · todo 0 · panic 0 · deps 0 · unwrap 10 · 283 lines |
| rule violations | **none** |

The only non-zero quality counter is `unwrap = 10`, and **all ten are in the
`#[cfg(test)]` block** (`parse("…").unwrap()` in tests, which is idiomatic). The
production code contains no `.unwrap()`, `.clone()`, `unsafe`, `panic!`, or
`todo!`, and no dependencies.

**On unseen seeds (grading uses a seed ≠ 0)** the score reads slightly under 100%
(seed 9999 → 99.87%, seed 7 → 99.82%, seed 12345 → 100%). I traced every one of
those "failures" and they are **not** translation bugs: `evaluate.py`'s random
generator sometimes emits a leading-zero numeric prerelease identifier (e.g.
`7.10.7-174.07.ysqd`, `8.23.30-k.01.8j`) into its *valid* pool. Those strings are
invalid SemVer; the real `semver` oracle rejects them, my Rust rejects them too —
they *agree* — but `chk_valid`/`chk_cmp` still score the case as a miss. To prove
faithfulness I ran an **independent oracle differential** (my harness vs real
`semver`, counting only cases where the oracle defines an answer) over **43,152
cases across 7 seeds: 0 genuine disagreements — 100.0000% faithful.** The lost
fraction of a percent is the harness's incomplete specification, not the code's.

## 2. How much came from the agent vs. the scaffold? (≈ 60% scaffold / 40% agent)

The model wrote every line of `lib.rs` in a single `write_rust` call and it was
correct first time — so the *codegen* is the agent's. But the *correctness* leans
on the scaffold. `system_prompt()` bakes in the semantics that first drafts get
wrong (build metadata ignored in precedence; a prerelease sorts below its absence;
numeric identifiers rank below alphanumeric; leading zeros illegal in the numeric
core and numeric prerelease ids but legal in build; bumps drop prerelease+build).
That prompt is effectively an executable spec I distilled by reading the source
and the oracle. A lite model handed the bare task would almost certainly have
missed precedence and leading-zeros (exactly as the README predicts). So the
scaffold supplied the *what*; the model supplied competent Rust for the *how* —
owned `String`s instead of clones, a clean `nat_cmp`/`cmp_prerelease_tag`, and
ported unit tests. The tools + loop (build → test → evaluate feedback, oracle,
context compression, stopping) are what would have *recovered* correctness had the
first draft been wrong; on this run they mostly served to confirm it.

## 3. What the agent did that I did not intend

- **First run: it over-probed.** Told to "confirm edge cases with the oracle," it
  made 11 oracle calls before writing anything and exhausted `gemini-3.6-flash`'s
  20-requests/day free quota without producing a single line of Rust. I hadn't
  anticipated the reward of caution colliding with a tiny budget. Fix: an oracle
  cap + a "write the complete file first, probe only failing cases" prompt.
- **It trusted the practice-seed signal.** `should_stop` fired at "100% on seed 0"
  and the agent declared done — but seed 0 is not the whole story. This is the
  assignment's own lesson playing out live: the reward signal (a test suite on one
  seed) is an incomplete specification, and the optimiser stopped the moment it
  was satisfied rather than when the task was truly finished.
- **It left its reasoning in the file** — a few "let's check the regex…" comments
  and one dead `is_numeric` variable survived into production code. Harmless, but
  not something I asked for.

## 4. Challenges and how I overcame them

1. **macOS python.org SSL** — `CERTIFICATE_VERIFY_FAILED` on every HTTPS call.
   Wired `certifi`'s CA bundle into the agent's SSL context.
2. **Gemini 3.x `thought_signature`** — the API 400s if a tool call is replayed
   without the `extra_content` signature it returned. `call_model` now preserves
   and re-sends it, and `build_context` keeps it through compression.
3. **Retired model names** — `gemini-2.5-flash`/`-lite` are gone for new keys;
   I discovered the live models via `/models` and moved to `gemini-3.5-flash-lite`.
4. **Free-tier quota (20/day)** — switched to a lite model with a larger quota and
   made the loop call-efficient (5 calls to done), plus honor the API's
   "retry in Ns" hint on 429s.
5. **A latent skeleton bug** — the tool-dispatch ternary was inverted
   (`"unknown tool" if tool else tool["fn"](…)`); it never fired in the stock
   skeleton because `call_model` raised first, but surfaced the instant it worked.
6. **Telling real bugs from harness noise** — I built an independent oracle
   differential so a sub-100% number couldn't be mistaken for a defect (or vice
   versa) before reporting it.

### Design notes (the five TODOs)

- **`call_model`** — OpenAI-compatible POST (provider set by env; Gemini default),
  signature-preserving, retry/backoff on 429/5xx.
- **`system_prompt`** — spec-heavy on purpose; the semantics are cheap to state
  and expensive for a small model to rediscover.
- **`build_context`** — WRITE/SELECT/COMPRESS: keep system + task + last 6 turns
  verbatim, shrink older whole-file writes and stale compiler dumps to stubs (the
  file and source are one tool call away, so nothing is truly lost).
- **`should_stop`** — done (correct + clean, verified by score not by the model's
  say-so), stuck (identical action ×3), idle (no tool call ×2), no-progress
  (N evals without beating best); keeps the best-scoring `lib.rs` and restores it.
- **tools** — added `edit_rust` (targeted fix without a full rewrite), `oracle`
  (check real-`semver` semantics; a dev aid, never called from Rust), and made
  `evaluate` return a compact structured summary instead of a wall of text.
