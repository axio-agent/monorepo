# Gas Town Multi-Agent Orchestration

Gas Town is an orchestration system for managing large numbers of concurrent coding agents
(Claude Code instances and compatible CLIs). It treats AI agent work as structured,
git-backed data. Every action is attributed, every agent has a persistent identity, and
every piece of work has provenance tracked through Beads - the universal git-backed data
plane that underpins the entire system.

This document is a comprehensive reference for the Gas Town methodology: its roles, work
units, communication protocols, lifecycle patterns, and key anti-patterns.

---

## 1. Philosophy and When to Use Gas Town

Gas Town is an "industrialized coding factory" - an opinionated system for running 10–30
coding agents simultaneously on sustained workloads. It is designed for developers at
Stage 7+ of the AI-assisted coding evolution: people who already manage multiple concurrent
CLI agent sessions and are pushing the limits of hand-management.

**Core philosophy:**

- **Vibe coding at scale.** Work is fluid. Some bugs get fixed two or three times; the
  winner is picked at merge time. Other fixes get lost. Throughput is the priority,
  not per-task perfection.
- **The human is a Product Manager.** You design features, file implementation plans,
  and sling work to your agents. Gas Town is an Idea Compiler.
- **Graceful degradation.** Every worker can operate independently. You choose which
  parts of Gas Town are running at any time. It works in no-tmux mode and limps along
  on naked sessions.
- **Expensive by design.** Gas Town is a cash guzzler. Multiple subscription accounts
  are typical. Do not use it if token cost is a primary concern.

**When to use Gas Town:**

- You routinely juggle five or more concurrent agent sessions.
- You need structured tracking of parallel work across multiple repositories.
- You want durable, crash-surviving workflows for multi-step agent tasks.
- You are comfortable with work being chaotic and occasionally redundant.

**When NOT to use Gas Town:**

- You work with a single agent at a time.
- You need deterministic, bit-exact workflow replay (use Temporal instead).
- You are cost-sensitive about LLM inference spend.

---

## 2. The GUPP Principle

GUPP - the Gas Town Universal Propulsion Principle - is the single most important rule
in the system:

> **If there is work on your hook, YOU RUN IT.**

This is physics, not politeness. Gas Town is a steam engine. Agents are pistons. The
entire system's throughput depends on one thing: when an agent finds work on its hook,
it executes immediately.

### How GUPP Works

Every Gas Town agent has a **hook** - a special pinned bead where work is hung via
`gt sling`. On startup, an agent checks its hook. If work is present, it begins
execution with no confirmation step, no announcement, no waiting.

```bash
gt hook           # What's on my hook?
gt prime          # Load full context and formula checklist
# Begin working immediately
```

### The Failure Mode GUPP Prevents

```
Agent restarts with work on hook
  -> Agent announces itself
  -> Agent waits for confirmation
  -> Witness assumes work is progressing
  -> Nothing happens
  -> Gas Town stops
```

### Startup Protocol (All Roles)

1. Check hook: `gt hook`
2. Work hooked → EXECUTE immediately (no announcement, no waiting)
3. Hook empty → Check mail: `gt mail inbox`
4. Still nothing → Wait for instructions (crew) or run `gt done` (polecats)

### The GUPP Nudge

Claude Code is "miserably polite" and sometimes waits for user input despite GUPP
prompting. Gas Town works around this with `gt nudge` - a zero-cost tmux notification
sent 30–60 seconds after startup. The nudge content does not matter ("hi", "do your
job", anything). The agent's strict role prompting causes it to ignore the text and
simply check its hook and mail.

Various patrol agents (Boot, Witness, Deacon) propagate nudge signals hierarchically,
ensuring every agent gets kicked within about 5 minutes if the town is running.

### `gt seance`: Talking to Predecessors

Because nudge messages include the Claude Code session ID, agents can use `gt seance`
to revive a predecessor session via `/resume` and ask it for handoff context that
failed to persist.

---

## 3. Roles

Gas Town defines eight roles. Some operate at the **town level** (global across all
repositories), others at the **rig level** (per-project).

| Role | Level | Emoji | Count | Managed By |
|------|-------|-------|-------|------------|
| Overseer | Town | 👤 | 1 (human) | Self |
| Mayor | Town | 🎩 | 1 | Overseer |
| Deacon | Town | 🐺 | 1 | Boot (Dog) |
| Dogs | Town | 🐶 | N | Deacon |
| Witness | Rig | 🦉 | 1 per rig | Deacon |
| Refinery | Rig | 🏭 | 1 per rig | Witness |
| Polecats | Rig | 😺 | N per rig | Witness |
| Crew | Rig | 👷 | N per rig | Overseer |

### 👤 Overseer (Human)

The eighth role. You have a persistent identity in Gas Town, your own inbox, and you
can send and receive town mail. You are the boss.

```bash
gt mail send --human -s "Subject" -m "Message to overseer"
gt mail inbox    # Read your mail
```

### 🎩 Mayor

The main agent you interact with. The Mayor is your concierge and chief-of-staff: it
kicks off convoys, coordinates cross-rig work, and receives notifications when convoys
land. It operates from the town level with visibility across all rigs.

**Responsibilities:**
- Initiate convoys and work distribution
- Coordinate cross-rig operations
- Receive escalations from Deacon and Witness
- Communicate with the Overseer

**Address:** `mayor/`

### 😺 Polecats

Ephemeral per-rig workers that spin up on demand. Polecats are the workhorses of Gas
Town - they swarm issues, produce merge requests, and hand them to the merge queue.
After merge, polecats go idle and their sandboxes are preserved for reuse.

**Key characteristics:**
- Single-task focus: one bead, one branch, one job
- Self-cleaning: `gt done` pushes, submits MR, goes idle
- GUPP-driven: work on hook triggers immediate execution
- Persistent identity but ephemeral sessions

**Address:** `<rig>/polecats/<name>` (e.g. `gastown/polecats/toast`)

### 🦉 Witness

The per-rig monitor. The Witness watches polecats, nudges them toward completion,
verifies clean git state before kills, and escalates stuck workers. Critically, the
Witness is NOT an implementer - it does oversight, not coding.

**Responsibilities:**
- Monitor polecat health and progress
- Pre-kill verification (git state, issue status)
- Send MERGE_READY to Refinery
- Escalate stuck workers to Mayor
- Detect and recover zombie/stalled polecats

**Address:** `<rig>/witness`

### 🏭 Refinery

The per-rig merge queue processor. When polecats complete work, branches go through
the Refinery for sequential, intelligent merging to main. This is a Bors-style
bisecting merge queue: one branch at a time, with conflict detection and rework
requests on failure.

**Responsibilities:**
- Process merge queue entries sequentially
- Rebase and merge to target branch (main)
- Send MERGED / MERGE_FAILED / REWORK_REQUEST to Witness
- Handle conflicts by creating resolution tasks

**Address:** `<rig>/refinery`

### 🐺 Deacon

The daemon beacon - a town-level patrol agent that runs a continuous loop. The Gas
Town daemon pings the Deacon every couple of minutes with a "do your job" signal.
The Deacon propagates this signal downward to other workers, ensuring the town
stays active.

**Responsibilities:**
- Run town-level patrol loop
- Dispatch Dogs for maintenance tasks
- Propagate heartbeat signals to rig-level agents
- Run town-level plugins
- Coordinate `gt handoff` and session recycling protocols

**Address:** `deacon/` (tmux session: `hq-deacon`)

### 🐶 Dogs

The Deacon's personal crew. Dogs handle maintenance tasks that would bog down the
Deacon's patrol loop. They come in two flavors:

**Imperative Dogs** (reliability-critical, implemented in Go):
- **Doctor**: 7 health checks, GC, zombie detection
- **Reaper**: Close/purge stale beads, auto-close, mail purge
- **Compactor**: Flatten + GC when Dolt commits exceed threshold

**Special Dog:**
- **Boot**: Awakened every 5 minutes by the daemon to check on the Deacon. Decides if
  the Deacon needs a heartbeat, a nudge, a restart, or to be left alone. Exists because
  the daemon's heartbeats were interrupting the Deacon's patrol.

**AI Dogs** (Agentic, dispatched by Deacon):
- `stuck-agent-dog`: Context-aware crash/stuck detection. Scope: polecats + deacon only.
  Never touches crew, Mayor, Witness, or Refinery.
- `quality-review`: Analyze merge quality trends per worker.
- `git-hygiene`: Cleanup stale branches.
- `github-sheriff`: GitHub org enforcement.

### 👷 Crew

Long-lived, human-facing workers with persistent identities. Crew members are NOT
managed by the Witness. You choose their names, you direct their work, and they
maintain context across sessions. The Crew are your direct replacements for whatever
workflow you previously used.

| Aspect | Crew | Polecat |
|--------|------|---------|
| Lifecycle | Persistent, user-controlled | Persistent identity, Witness-managed |
| Work assignment | Human-directed | Slung via `gt sling` |
| Git workflow | Can push to main | Branch only, Refinery merges |
| Monitoring | None | Witness watches and nudges |
| Best for | Design, exploration, long projects | Discrete, parallelizable tasks |

**Address:** `<rig>/crew/<name>` (e.g. `gastown/crew/joe`)

---

## 4. The MEOW Stack

MEOW - Molecular Expression of Work - is the layered system for decomposing, tracking,
and executing work in Gas Town.

### Beads

The atomic work unit. A bead is a lightweight issue-tracker issue stored as JSON in a
Dolt database (git-like versioning). Each bead has an ID, title, status, assignee,
priority, type, and dependency links.

```bash
bd create --title="Fix auth bug" --type=bug --priority=2
bd show gt-abc
bd close gt-abc
bd ready          # Show unblocked work
```

### Epics

Beads with children. Children of epics are parallel by default but can have explicit
dependencies to force sequencing. Epics allow "upside-down" plans where the root is
the last thing to complete and leaves are the first.

### Molecules

Durable multi-step workflows, chained with beads. Unlike epics, molecules can have
arbitrary shapes, loops, gates, and are Turing-complete. Each step is executed by an
agent. Molecules survive agent crashes, compactions, restarts, and interruptions.

```
Formula (TOML source) --> bd cook --> Protomolecule (frozen template)
                                          |
                               bd mol pour | bd mol wisp
                                          |
                                    Molecule / Wisp (active instance)
```

### Formulas

TOML templates for molecules. Formulas provide a macro-expansion phase for composing
workflows with loops and gates. They are the "source code" for workflows.

```toml
formula = "mol-polecat-work"
version = 1

[[steps]]
id = "load-context"
title = "Load context and verify assignment"
description = "Initialize session and understand assignment"

[[steps]]
id = "implement"
title = "Implement changes"
needs = ["load-context"]
description = "Do the actual work"

[[steps]]
id = "test"
title = "Run tests"
needs = ["implement"]
description = "Run quality gates"
```

### Wisps

Ephemeral beads - the "vapor phase" of Gas Town work. Wisps exist in the database and
get hash IDs but are NOT written to the JSONL file and NOT persisted to git. At the end
of their run, wisps are "burned" (destroyed). Optionally they can be squashed into a
single-line summary.

Wisps are critical for high-velocity orchestration. All patrol agents create wisp
molecules for every patrol or workflow run, ensuring transactional completion without
polluting git history.

**Two wisp modes:**

| Mode | Storage | Use Case |
|------|---------|----------|
| Root-only | Single root wisp, steps inline | Patrols, polecat work (high frequency, cheap) |
| Poured | Sub-wisps with checkpoints | Releases, expensive workflows (low frequency) |

**Heuristic:** If you would curse losing progress after a crash, use poured mode.

---

## 5. Molecular Workflows

### Step Definitions

Each molecule step is a bead with:
- An ID and title
- A description (the agent's instructions)
- Dependencies (`needs = [...]`) defining execution order
- Optional model constraints (tier, provider, cost limits)

### Dependencies and Parallelism

Steps without dependency links run in parallel. Explicit `needs` fields create
sequencing. The DAG structure supports arbitrary shapes.

### Model Tiers

Steps can specify model constraints for cost/quality optimization:

```toml
[[steps]]
id = "quick-scan"
model = "auto"
max_cost = 0.001       # Cheapest capable model

[[steps]]
id = "deep-work"
model = "auto"
min_mmlu = 85          # High-quality model required
needs = ["quick-scan"]
```

### Session-Per-Step Model

Each molecule step may be executed in a different Claude Code session. When context
fills or a step completes, the agent hands off (`gt handoff`) and a fresh session
picks up from the next step via `gt prime`. The molecule state persists in beads
across all session boundaries.

```
Session 1: Steps 1–2 --> handoff
Session 2: Steps 3–4 --> handoff
Session 3: Step 5     --> gt done
```

---

## 6. Convoys

A convoy is Gas Town's work-order wrapper. It tracks related beads across multiple
rigs and notifies subscribers when all tracked work lands.

```bash
# Create a convoy
gt convoy create "Feature X" gt-abc gt-def --notify overseer

# Check progress
gt convoy status hq-cv-abc

# Dashboard
gt convoy list
```

### Convoy vs Swarm

| Concept | Persistent? | Description |
|---------|-------------|-------------|
| Convoy | Yes (`hq-cv-*`) | Tracking unit you create, monitor, get notified about |
| Swarm | No | Ephemeral - "the polecats currently working on this convoy's issues" |

### Reactive Feeding

Convoys can receive additional issues after creation. Adding issues to a closed convoy
reopens it automatically. Multiple swarms can "attack" a convoy before it finishes -
the Witness keeps recycling polecats and pushing them on open issues.

### Auto-Convoy on Sling

When you sling a single issue, Gas Town auto-creates a convoy so even a "swarm of
one" appears in the dashboard:

```bash
gt sling bd-xyz beads/amber
# Auto-creates convoy "Work: bd-xyz", assigns polecat
```

### Convoy Lifecycle

```
OPEN --(all issues close)--> CLOSED (landed)
  ^                              |
  +---(add more issues)----------+
       (auto-reopens)
```

---

## 7. Communication: Nudge vs Mail

Gas Town has two communication channels with fundamentally different costs.

### `gt nudge` (Ephemeral, Preferred)

- Sends a message directly to an agent's tmux session
- No beads created, no Dolt commits, zero storage cost
- Message appears as a system-reminder in the agent's context
- Lost if the target session is dead

```bash
gt nudge mayor "Status update: PR review complete"
gt nudge gastown/witness "Polecat health check needed"
```

### `gt mail send` (Persistent, Protocol Only)

- Creates a wisp bead in Dolt with a permanent commit
- Persists across session restarts - survives agent death
- Expensive: every mail = permanent Dolt commit

```bash
gt mail send gastown/witness -s "MERGE_READY nux" -m "Branch: feature-xyz
Issue: gp-abc
Verified: clean"
```

### Decision Matrix

| Scenario | Channel | Rationale |
|----------|---------|-----------|
| Wake a sleeping agent | `gt nudge` | Ephemeral, zero cost |
| Health check ping | `gt nudge` | Routine, session-scoped |
| MERGE_READY protocol | `gt mail send` | Must survive session death |
| HELP/escalation | `gt mail send` | Must survive session death |
| Handoff context | `gt mail send` | Successor needs this after restart |
| Status update | `gt nudge` | Informational only |
| Polecat poke | `gt nudge` | Routine monitoring |

**The litmus test:** "If the recipient's session dies and restarts, do they need this
message?" If yes, mail. If no, nudge.

### Role-Specific Mail Budget

| Role | Mail Budget | Mail For | Nudge For |
|------|-------------|----------|-----------|
| Polecat | 0–1 per session | HELP only | Everything else |
| Witness | Protocol only | MERGE_READY, RECOVERY_NEEDED | Health checks, pokes |
| Refinery | Protocol only | MERGED, MERGE_FAILED | Status updates |
| Deacon | Escalations only | Escalations to Mayor | HEALTH_CHECK, pokes |
| Dogs | Zero | Never | Report to Deacon via nudge |

### Key Protocol Messages

| Message | Route | Purpose |
|---------|-------|---------|
| POLECAT_DONE | Polecat → Witness | Signal work completion |
| MERGE_READY | Witness → Refinery | Branch verified, ready to merge |
| MERGED | Refinery → Witness | Merge succeeded |
| MERGE_FAILED | Refinery → Witness | Merge failed (tests, build) |
| REWORK_REQUEST | Refinery → Witness | Rebase needed due to conflicts |
| RECOVERY_NEEDED | Witness → Deacon | Dirty polecat needs manual recovery |
| HELP | Any → Mayor | Stuck, needs intervention |
| HANDOFF | Agent → Self | Session continuity context |

---

## 8. Polecat Lifecycle

Polecats have three distinct lifecycle layers that operate independently.

### Three Layers

| Layer | Component | Lifecycle | Persistence |
|-------|-----------|-----------|-------------|
| Identity | Agent bead, CV chain, work history | Permanent | Never dies |
| Sandbox | Git worktree, branch | Persistent across assignments | Created on first sling |
| Session | Claude instance, context window | Ephemeral | Cycles on handoff/crash/done |

### Four Operating States

| State | Description | How It Happens |
|-------|-------------|----------------|
| Working | Actively executing | Normal after `gt sling` |
| Idle | Completed, sandbox preserved | After `gt done` succeeds |
| Stalled | Session stopped mid-work | Crash, timeout, lost nudge |
| Zombie | Completed but cleanup failed | `gt done` failed |

**Happy path:** IDLE → (gt sling) → WORKING → (gt done) → IDLE

### Startup Protocol

1. Announce: "Polecat `<name>`, checking in."
2. Run: `gt prime && bd prime`
3. Check hook: `gt hook`
4. If formula attached, steps shown inline by `gt prime`
5. Work through checklist, then `gt done`

**If NO work and NO mail:** run `gt done` immediately.

**If assigned bead has nothing to implement** (already done, cannot reproduce):

```bash
bd close <id> --reason="no-changes: <brief explanation>"
gt done
```

### Completion Protocol (MANDATORY)

```
[ ] 1. Run quality gates (lint, format, tests - ALL must pass)
[ ] 2. Stage changes:     git add <files>
[ ] 3. Commit changes:    git commit -m "msg (issue-id)"
[ ] 4. Self-clean:        gt done   ← MANDATORY FINAL STEP
```

`gt done` pushes the branch, creates an MR bead in the merge queue, sets the agent
to idle, and kills the session. The polecat is gone after `gt done`.

**The Landing Rule:** Work is NOT landed until it is in the Refinery MQ.

```
Local branch --> gt done --> MR in queue --> Refinery merges --> LANDED
```

### The Idle Polecat Heresy

The most critical failure mode: a polecat that completed work but sits idle instead of
running `gt done`. There is no approval step. If you have finished your implementation
work, your ONLY next action is `gt done`.

Do NOT:
- Sit idle waiting for more work (there is no more work - you're done)
- Say "work complete" without running `gt done`
- Wait for confirmation or approval
- Try `gt unsling` or other commands (only `gt done` signals completion)

### Spawn Storms

When a polecat exits without closing its bead (no `gt done`, no `bd close`), the
Witness zombie patrol resets the bead to `open` and dispatches it to a new polecat.
If this repeats, 6–7 polecats can be assigned the same bead.

**Prevention:** Every session must end with either a branch push via `gt done` OR an
explicit `bd close` on the hook bead.

### Persist Findings Early

Sessions can die at any time. Code survives in git, but analysis, findings, and
decisions exist ONLY in the context window. Persist to the bead as you work:

```bash
bd update <issue-id> --notes "Findings: <what you discovered>"
bd update <issue-id> --design "<structured findings>"
```

**Do this early and often.** If your session dies before persisting, the work is lost.

---

## 9. Patrol System

Patrols are ephemeral (wisp) workflows that patrol agents run in a loop. They have
exponential backoff: agents gradually sleep longer when no work is found.

### Witness Patrol

Steps in sequence:
1. **inbox-check** - Process POLECAT_DONE, MERGED, HELP, escalations
2. **process-cleanups** - Handle cleanup wisps from dead polecats
3. **check-refinery** - Verify refinery is alive
4. **survey-workers** - Check all active polecats for health/progress
5. **check-timer-gates** - Evaluate elapsed timer gates
6. **check-swarm** - Track convoy completion
7. **patrol-cleanup** - Close completed patrol wisps
8. **context-check** - If context full, handoff for fresh session
9. **loop-or-exit** - Report and spawn next cycle

### Deacon Patrol

Steps in sequence:
1. inbox-check, trigger-pending-spawns, gate-evaluation
2. dispatch-gated-molecules, check-convoy-completion
3. health-scan (Dolt status), zombie-scan
4. plugin-run, dog-pool-maintenance, orphan-check, session-gc
5. patrol-cleanup, context-check, loop-or-exit

### Refinery Patrol

Steps in sequence:
1. inbox-check (MERGE_READY from Witness)
2. queue-scan, process-branch, run-tests
3. handle-failures (bisect if needed), merge-push
4. notify Witness (MERGED), cleanup, context-check, loop-or-exit

### Patrol Lifecycle

```bash
gt patrol new                         # Create root-only patrol wisp
gt prime                              # Shows patrol checklist inline
# Work through each step in sequence
gt patrol report --summary "..."      # Close current patrol + start next cycle
```

`gt patrol report` atomically closes the current patrol root and spawns a new one
for the next cycle.

### Exponential Backoff

When patrols find no work, the wait between cycles increases. Any mutating `gt` or
`bd` command wakes the town, or you can wake workers manually with `gt nudge`.

---

## 10. Handoff and Session Cycling

Session cycling is normal operation, not failure. The agent continues working - only
the Claude context window refreshes.

### When to Handoff

- **Context filling** - slow responses, forgetting earlier context
- **Logical chunk done** - good checkpoint between molecule steps
- **Stuck** - need a fresh perspective

### How Handoff Works

```bash
gt handoff -s "Session cycling" -m "Issue: gt-abc
Current step: 3 of 5
Progress: tests passing, docs remaining"
```

The `gt handoff` command:
1. Optionally sends a handoff mail to self with context
2. Restarts the session in tmux
3. The new session auto-primes via the SessionStart hook (`gt prime`)
4. GUPP kicks in: finds work on hook, continues execution

### Context Recovery

After compaction, clear, or new session:

```bash
gt prime          # Full role context reload
gt hook           # Check for assigned work
gt mail inbox     # Check for handoff messages
```

The molecule state, hook, and agent identity all persist in beads across session
boundaries. Only the Claude context window is lost.

---

## 11. Nondeterministic Idempotence (NDI)

Gas Town achieves durable execution guarantees through a principle called
Nondeterministic Idempotence. This is conceptually similar to Temporal's deterministic
durable replay but uses completely different machinery.

### How NDI Works

All work is expressed as molecules. Each component is persistent:

- **Agent**: a bead backed by git. Sessions come and go; agents stay.
- **Hook**: a pinned bead backed by git.
- **Molecule**: a chain of beads, also in git.

If a session crashes mid-step, the next session finds the molecule on the hook,
determines which step it was on, figures out the right fix, and moves on. The path
is fully nondeterministic - the agent might take different actions each time - but
the outcome eventually converges on the workflow's acceptance criteria.

### Comparison to Temporal

| Aspect | Temporal | Gas Town |
|--------|----------|----------|
| Replay model | Deterministic replay from event log | Nondeterministic re-execution |
| State persistence | Event sourcing | Git-backed beads |
| Step execution | Exactly-once with replay | At-least-once with self-correction |
| Worker model | Language SDK activities | Superintelligent AI agents |
| Guarantee | Deterministic completion | Eventual completion |
| Failure recovery | Replay from last checkpoint | Agent re-examines state, self-corrects |

**Key insight:** Because each step is executed by a superintelligent AI that can reason
about partial state, the system does not need deterministic replay. The agent inspects
what happened, corrects any issues, and continues. Mistakes along the way can be
self-corrected because the molecule's acceptance criteria are well-specified by whoever
designed the formula.

---

## 12. Key Anti-Patterns

### The Idle Polecat Heresy

Completing work without running `gt done`. The polecat sits idle, the Witness assumes
it is still working, and the system stalls.

### Pushing to Main

Polecats NEVER push directly to main. All polecat work goes through the merge queue.
The Refinery is the only role that writes to main. Do NOT create GitHub PRs either -
the merge queue handles everything.

### Closing Foreign Wisps (Swim Lane Rule)

You may ONLY close wisps that YOU created. Wisp lifecycle for other agents' wisps is
the reaper Dog's responsibility. Closing a foreign wisp kills active polecat work
molecules.

### Mailing for Routine Communication

Using `gt mail send` where `gt nudge` would suffice. Every mail creates a permanent
Dolt commit. Four agents doing 15 patrol cycles with 2 mails each generates 120
permanent commits per day for routine chatter alone.

### Working Outside Your Assigned Bead

Polecats must work on their assigned bead only. Discovered work should be filed with
`bd create`, not fixed in-place. Do not get distracted by tangential discoveries.

### Spawn Storms

Caused by polecats exiting without closing their bead. The Witness resets the bead to
open, dispatches a new polecat, which also fails to close it, creating an exponentially
growing number of polecats assigned to the same bead. Prevention: always `gt done` or
`bd close` before exit.

### Health Check Responses via Mail

When Deacon sends a health check nudge, do NOT respond with mail. The Deacon tracks
health via session status, not mail responses. Responding with mail generates one
permanent commit per health check per agent per patrol cycle.

### Duplicate Escalations

Witnesses sending multiple mails about the same issue minutes apart. Check your inbox
before sending: if you already mailed about this topic, do not send again.

---

## 13. Dolt Health

Dolt is the git-like SQL database backing all Gas Town data. One Dolt server per town
serves all rig databases via MySQL protocol on port 3307. The critical thing to
understand: **Dolt is git, not Postgres.** Every `bd` write command and every
`gt mail send` generates a permanent Dolt commit that lives in the history forever.

### Agent Responsibilities

- **Nudge, don't mail.** `gt nudge` costs zero. `gt mail send` costs one permanent commit.
- **Don't create unnecessary beads.** File real work, not scratchpads.
- **Close your beads promptly.** Open beads that linger become pollution.
- **Don't retry `bd` commands in a loop** when Dolt is slow or down. Check `gt health`
  and nudge the Deacon.
- **Don't file beads about Dolt trouble** - someone is already handling it.

### Cost Model

| Operation | Dolt Cost |
|-----------|-----------|
| `gt nudge` | Zero (tmux only) |
| `bd show`, `bd ready` | Read-only, negligible |
| `bd create`, `bd update`, `bd close` | 1 permanent commit each |
| `gt mail send` | 1 permanent commit (wisp bead) |
| `gt patrol report` | 1 commit (wisp lifecycle) |

---

## 14. Comparison to Kubernetes

Gas Town's architecture bears a structural resemblance to Kubernetes, though the
systems optimize for fundamentally different things.

| Kubernetes | Gas Town | Role |
|------------|----------|------|
| kube-scheduler | Mayor / Deacon | Control plane, work distribution |
| Nodes | Rigs | Execution environments |
| kubelet | Witness | Per-node/rig agent monitoring workers |
| Pods | Polecats | Ephemeral workers executing tasks |
| etcd | Beads (Dolt) | Source of truth the system reconciles against |
| CronJobs | Patrols | Scheduled recurring work |
| DaemonSet | Deacon + Dogs | System-level background tasks |

### The Key Difference

Kubernetes asks: **"Is it running?"**
Gas Town asks: **"Is it done?"**

Kubernetes optimizes for uptime - keep N replicas alive, restart crashed pods,
maintain the desired state indefinitely. Gas Town optimizes for completion - finish
the work, land the convoy, nuke the worker, and move on.

Kubernetes pods are anonymous cattle. Gas Town polecats are credited workers whose
completions accumulate into CV chains; the sessions are the cattle. Kubernetes
reconciles toward a continuous desired state. Gas Town proceeds toward a terminal goal.

Same engine shape. Radically different destination.
