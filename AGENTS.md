## Commits

For commits:
- Use `feat: <description>` for UX- or user-facing features.
- Use `fix: <description>` for bug fixes and corrections.
- Use `docs: <description>` for documentation updates, including spec files.

## Verification

For future code changes in this repo, run a local CI-equivalent dry run before wrapping up.

Default local gate:

```bash
python3 -m unittest discover -s tests
```

When a change touches packaging, release, updater, or CI behavior, also run the most relevant targeted tests for that area.

## Subagent Policy

For non-trivial tasks, proactively use subagents when the work can be split safely. Only use subagents defined in `/Users/isaacibm/.codex/agents`.

Use subagents for:
- Codebase exploration
- PR review
- Bug investigation
- Test failure triage
- Security / correctness / maintainability review
- Comparing multiple implementation approaches

Default pattern:
- Spawn one read-only explorer subagent to map relevant files and execution paths.
- Spawn one reviewer subagent for correctness, edge cases, and tests.
- Wait for all subagents.
- Summarize findings before editing.

Operational rules:
- When using a named `agent_type`, do not pass `fork_context`; named agents already inherit the appropriate role, model, and effort settings.
- Prefer local named spark agents from `/Users/isaacibm/.codex/agents` over generic `default` or `explorer` agents when a named role fits the task.
- Keep worker fan-out under the configured 8-thread ceiling; for workbook or batch processing, spawn 4-6 workers at a time unless there is clear evidence more will fit safely.
- Explicitly `wait_agent` and `close_agent` for every spawned child before ending the turn, unless the user asks to leave the child running.

Avoid subagents for:
- Tiny edits
- Single-file obvious fixes
- Tasks where multiple agents would edit the same files in parallel
- Anything likely to create merge conflicts

When using subagents, tell the user:
1. Which subagents were spawned
2. What each one checked
3. What conclusions they returned
4. What will happen next
