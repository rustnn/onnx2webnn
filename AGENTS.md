# Agent guidance

- Verify behavior against current source, tests, and tool output before describing it. Do not
  treat documentation as more authoritative than the implementation.
- Keep this file procedural and put user-facing guidance in the README. Do not add API
  inventories, capability counts, model results, CLI options, dependency versions, or snapshots.
- Keep source comments sparse and focused on non-obvious behavior or invariants.
- Use maintained generators for generated files and check for drift when changing their inputs.
- Run checks relevant to the change and report what was and was not verified.
- Preserve unrelated worktree changes. Do not commit or push unless explicitly requested.
- Make sure that operator support state is documented in source and on this projects exectuables