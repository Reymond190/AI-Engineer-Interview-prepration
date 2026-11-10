# Multi-agent orchestration with LangGraph: interview guide

Short answers you should be able to give from memory. Lesson numbers point to
the runnable script that demonstrates each one.

---

## 1. Fundamentals

**What is LangGraph and why use it over a plain LLM chain?**
A low-level orchestration framework that models an agent as a stateful graph:
typed state, nodes (functions), edges (control flow), including cycles. It gives
you persistence, human-in-the-loop, streaming, and fine control over the loop,
none of which a linear chain has. LangChain 1.x's `create_agent` is built on it.

**Workflow vs agent?** (Anthropic, "Building effective agents")
In a workflow, code defines the control flow and LLMs fill in the steps (prompt
chaining, routing, parallelisation, orchestrator-workers, evaluator-optimizer).
In an agent, the LLM decides the control flow dynamically (a tool loop).
Start with the simplest thing that works; add agency only when the path can't be
predicted in advance.

**Explain state and reducers.** (01)
State is a TypedDict or Pydantic model. Nodes return *partial* updates. Each key
merges by its reducer: overwrite by default, `operator.add` appends, and
`add_messages` appends and upserts by message id (and deletes on `RemoveMessage`).
Parallel nodes writing the same key without a reducer raise `InvalidUpdateError`.

**How does execution work?** (01, 08)
Pregel-style *super-steps*: every node triggered in a step runs in parallel, then
their updates are applied together, then the next step starts. A checkpoint is
saved after every super-step.

**Conditional edges vs `Command`?** (02)
A conditional edge puts the routing function on the edge and keeps nodes pure.
`Command(update=..., goto=...)` updates state and routes in one return value,
which suits handoffs where the routing decision comes out of the node's own work.
`Command(graph=Command.PARENT)` jumps out of a subgraph.

**Graph API vs Functional API?** (15)
They share the same runtime. The Functional API (`@entrypoint`/`@task`) uses
plain Python control flow; task results are checkpointed, so a resume replays
them instead of re-running. The Graph API gives explicit state/reducers and a
diagram you can visualise.

## 2. Tools

**How does tool calling work end to end?** (03)
Tool schemas (name, description, JSON schema) go in the request; the model
returns `AIMessage.tool_calls`; `ToolNode` executes them (in parallel) and returns
`ToolMessage`s with matching `tool_call_id`s; the loop repeats until there are no
tool calls. Every tool call must get a ToolMessage back.

**The model doesn't support function calling. What now?** (03b)
1) Switch to a model or server that does. 2) Use constrained decoding (JSON
schema / grammar). 3) Use a prompt protocol plus a lenient parser, restricted to
known tool names, with a nudge/retry when the model describes a call instead of
making it.

**What makes a good tool?**
Clear name and docstring, few typed params, actionable error strings ("unknown
product; known: a, b"), idempotent writes, bounded output size. Fewer tools per
agent gives better tool selection, which is a real reason to split into multiple agents.

**What is MCP?** (19)
The Model Context Protocol, an open standard for connecting agents to tool
servers over JSON-RPC. It defines tools (model-controlled), resources
(app-controlled context) and prompts (user templates), over stdio or
Streamable HTTP with OAuth. `langchain-mcp-adapters` turns MCP tools into
LangChain tools. Risks: prompt injection via tool descriptions or results, and
over-broad permissions. A2A is the agent-to-agent counterpart.

## 3. Multi-agent architectures

| Pattern | Shape | Use when | Watch out |
|---|---|---|---|
| Supervisor (05) | star | clear specialist roles, central control | extra hop per step; supervisor is a bottleneck |
| Agents-as-tools | star via tool calls | default in recent LangChain docs | same as supervisor |
| Handoffs / swarm (06) | peer to peer | conversational routing between specialists | ping-pong loops; cap the handoffs |
| Hierarchical (07) | tree of subgraphs | many agents, teams | latency, debugging depth |
| Network / debate (18) | everyone sees everything | reasoning/factuality via critique | cost = agents × rounds |
| Orchestrator-workers / map-reduce (08) | dynamic fan-out | N independent subtasks (deep research) | aggregation quality, concurrency limits |
| Plan-and-execute (14) | planner + executor loop | long-horizon tasks | rigid plans; needs a replanner |

**When should you NOT use multi-agent?**
When one agent with good tools works. Multi-agent costs more tokens and latency,
and errors compound across handoffs. Split when there are too many tools for one
prompt, when contexts conflict, or when you need parallelism or context isolation.

**How do agents share context?**
Through a shared message list (simple, grows fast), or private state plus compact
summaries passed back (05, 07, 20). Context isolation (sub-agents with fresh
contexts that return only a summary) is a key technique in deep research systems.

**How do you stop infinite loops?**
`recursion_limit`, explicit counters in state (max rounds/handoffs/retries), and
termination conditions enforced in code, not in prompts (05's FINISH guardrail).

## 4. State, persistence, memory

**What does a checkpointer give you?** (09)
Multi-turn memory per `thread_id`, pause/resume for HITL, fault tolerance (resume
from the last super-step; finished parallel branches are kept as pending writes),
and time travel (`get_state_history`, replay from a `checkpoint_id`, fork with
`update_state`). In production use Postgres; SQLite or in-memory for development.
Durability modes are `exit`, `async` (default) and `sync`.

**Short-term vs long-term memory?** (11)
Short-term is the thread's messages in the checkpoint; manage it with trimming
and summaries. Long-term is a `Store` shared across threads, namespaced per
user, optionally with semantic search. Memory types: semantic (facts), episodic
(past experiences, few-shot examples), procedural (instructions/prompt).
Memories can be written in the hot path or in the background.

**What is context engineering?**
Deciding what enters the window at each step. Four moves: *write* (scratchpads,
files, memory), *select* (retrieval, relevant memories/tools), *compress*
(summaries, trimming), *isolate* (sub-agents, sandboxes, separate state keys).
Re-injecting the todo list each turn (20) is one example.

## 5. Human-in-the-loop (10)

`interrupt(payload)` pauses inside a node and persists state; resume with
`invoke(Command(resume=value), config)`. Gotcha: **the node re-runs from the top
on resume**, so code before `interrupt()` must be side-effect free. Patterns:
approve/reject, edit tool args, give feedback, ask a clarifying question. Gate
writes, payments and external communications; let reads run freely.

## 6. Streaming (12)

Stream modes: `updates` (per-node deltas), `values` (full state), `messages`
(LLM tokens + node metadata), `custom` (`get_stream_writer()` progress events),
plus `debug`/`tasks`/`checkpoints`. Pass a list to combine them;
`subgraphs=True` adds namespaces. Filter tokens by `langgraph_node` or by tags.

## 7. Reliability and production (16)

- Node policies: `RetryPolicy`, `CachePolicy` (+ cache backend), `error_handler`
  (receives `NodeError`, returns a `Command` to a fallback), `TimeoutPolicy`.
  `set_node_defaults` applies them graph-wide.
- Model-level: `with_retry`, `with_fallbacks`, request timeouts.
- Validate structured outputs and feed errors back (04); enforce invariants in code.
- Observability: trace every run (LangSmith / OpenTelemetry); track tokens,
  latency, steps, tool error rate.
- Safety: least-privilege tools, human approval for side effects, prompt
  injection defences (treat tool output as data), sandboxed code execution.
- Deployment: LangGraph Server / Platform (assistants, threads, runs, cron,
  streaming over SSE), or your own FastAPI around `graph.astream`.

## 8. Patterns for quality

- **Evaluator-optimizer / reflection** (13): prefer *grounded* feedback (tests,
  validators, retrieval checks) over pure self-critique; cap iterations; keep
  the best attempt.
- **Agentic RAG** (17): route, retrieve, grade, rewrite, generate with
  citations, then check groundedness, or abstain.
- **Debate / self-consistency / best-of-n** (18): sample several answers, critique, aggregate.
- **Reasoning budgets**: use thinking mode where it pays (planning, judging) and
  fast mode for routing and tool calls.

## 9. Gotchas found while building this course (good war stories)

- **Model can't call tools natively** (DeepSeek-R1-Distill): it narrated plans,
  dropped the call format and claimed actions it never took. Fix: switch models,
  or use a parser + nudge + verification (03b).
- **Constrained JSON without the schema in the prompt** (vLLM `json_schema`): the
  grammar enforces shape but the model never sees field descriptions, so a judge
  scored every answer 1/5. Fix: put the schema/rubric in the prompt; put `reasoning`
  before scores (21). But HOW you show it matters: a raw JSON-schema dump flipped
  a router from 3/3 correct to 3/3 wrong, while a plain field list fixed it. Measure
  prompt changes with an eval harness (`run_all.py`), don't eyeball them.
- **Whitespace runaway in guided decoding**: after a long `<think>`, the model
  emitted newlines until it hit max_tokens. Fix: `disable_any_whitespace` (server/serve.sh).
- **Thinking budget exhausted**: all 4,096 tokens spent reasoning, so there was no answer.
  Fix: a larger budget, and thinking only for the planner/judge (13, 14).
- **Supervisor shortcuts**: it skipped the workers and asked the writer to "know"
  facts. Fix: describe worker limits, preconditions in code, reasoning for routing (05).
- **Handoff amnesia**: the receiving agent assumed the work was done. Fix: put the
  open request into the handoff message (06).
- **Debaters fabricate evidence, and the judge rewards it.** Fix: instruct against it
  and penalise unverifiable claims (18).
- **Critic's suggestion broke working code (8/8 to 0/8).** Fix: keep the best
  attempt and gate revisions on tests (13).
- **Router misroutes on vocabulary mismatch.** Fix: describe what each destination
  covers, and add few-shot examples (17).

## 10. Evaluation (21)

1. Unit tests with a fake LLM (scripted `AIMessage`s with tool_calls) for wiring and error paths.
2. Trajectory evals: compare tool-call sequences with a reference (strict / unordered / superset).
3. Output evals: exact or programmatic checks first, then LLM-as-judge with a rubric
   and structured scores, calibrated against human labels.
Run them offline on datasets before every prompt or model change, and online on sampled traces.

## 11. Questions to practise out loud

1. Design a customer-support system with billing, tech and refunds agents. Which topology, and why?
2. Your agent loops forever calling the same tool. How do you debug it and prevent it?
3. How would you add human approval to refunds over $100 without blocking other work?
4. The conversation exceeds the context window after 30 turns. What are your options?
5. How do you make a 10-step agent resumable after a crash halfway through?
6. A sub-agent's output keeps confusing the supervisor. What do you change?
7. How do you evaluate an agent whose answers are open-ended?
8. When would you pick the Functional API over the Graph API?
9. How do you stream partial results from a node deep inside a subgraph to a UI?
10. How do you protect an MCP-connected agent against prompt injection from tool results?
