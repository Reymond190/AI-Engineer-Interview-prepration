# LangGraph multi-agent course (local Qwen3.5-9B)

Runnable scripts covering what interviewers ask about LangGraph and multi-agent
orchestration. Every script has an "Interview topics" docstring at the top:
read that first, run it, then change things and break them.

## Setup

```bash
# Terminal 1 (WSL): start the model server (Qwen3.5-9B, native tool calling)
bash server/serve.sh

# Terminal 2 (Windows, project root): run a lesson
.venv\Scripts\python.exe langgraph_course\common.py           # connectivity check
.venv\Scripts\python.exe langgraph_course\03_tool_calling_react.py
```

Packages (already in `.venv`): `langgraph langchain langchain-openai
langgraph-checkpoint-sqlite langchain-mcp-adapters mcp python-dotenv`.

`common.py` points `ChatOpenAI` at the local vLLM server. Swap `get_llm()` for
any hosted model and every lesson still works.

## Lessons

| # | Script | Core idea |
|---|--------|-----------|
| 01 | `01_state_graph_basics.py` | State, reducers, nodes, edges, super-steps, input/output schemas, runtime context |
| 02 | `02_routing_conditional_edges.py` | Router pattern; `add_conditional_edges` vs `Command(goto=...)` |
| 03 | `03_tool_calling_react.py` | ReAct loop by hand: `bind_tools`, `ToolNode`, `tools_condition`, parallel calls, `create_agent` |
| 03b | `03b_tool_calling_without_native_support.py` | Emulating tool calls for models without function calling |
| 04 | `04_structured_output.py` | `with_structured_output`, constrained decoding, validate-and-retry loop |
| 05 | `05_supervisor_multi_agent.py` | Supervisor / orchestrator-worker, compact work log, guardrails in code |
| 06 | `06_handoffs_swarm.py` | Decentralised handoffs (swarm), `active_agent`, multi-turn with checkpointer |
| 07 | `07_hierarchical_subgraphs.py` | Subgraphs (shared vs private state), hierarchical teams, streaming namespaces |
| 08 | `08_parallel_map_reduce_send.py` | Fan-out/fan-in, `Send` map-reduce, `defer=True`, `max_concurrency` |
| 09 | `09_persistence_time_travel.py` | Checkpointers, threads, durability, state history, replay, fork |
| 10 | `10_human_in_the_loop.py` | `interrupt()` and `Command(resume=...)`: approve / edit / reject a tool call |
| 11 | `11_memory_short_and_long_term.py` | Trimming, summarisation + `RemoveMessage`, Store namespaces, semantic memory |
| 12 | `12_streaming.py` | Stream modes: updates, values, messages (tokens), custom; filtering by node/tag |
| 13 | `13_reflection_evaluator_optimizer.py` | Generate, then run tests (grounded), then LLM critique, then revise |
| 14 | `14_plan_and_execute.py` | Planner / executor / replanner |
| 15 | `15_functional_api.py` | `@entrypoint` / `@task`, parallel futures, resume without recompute |
| 16 | `16_reliability_production.py` | `RetryPolicy`, `CachePolicy`, `error_handler`, `TimeoutPolicy`, recursion limit, fallbacks |
| 17 | `17_agentic_rag.py` | Route, retrieve, grade, rewrite, generate, groundedness check (CRAG / Self-RAG) |
| 18 | `18_debate_network.py` | Network topology, multi-agent debate, judge with rubric |
| 19 | `19_mcp_tools.py` | MCP server (tools / resources / prompts) and the LangGraph client |
| 20 | `20_deep_agent.py` | Deep agents: todo planning tool, virtual filesystem, sub-agents, tools that update state |
| 21 | `21_evaluation_testing.py` | Fake-LLM unit tests, trajectory evals, LLM-as-judge |

See `INTERVIEW_GUIDE.md` for the concepts and likely questions, with short answers.

## Scoring / comparing models

`run_all.py` runs every lesson and checks outcomes (right number, right action,
right route), not just exit codes:

```bash
.venv\Scripts\python.exe langgraph_course\run_all.py --save results_a.json
.venv\Scripts\python.exe langgraph_course\run_all.py --repeat 3 05 06 14   # flaky lessons
.venv\Scripts\python.exe langgraph_course\run_all.py --compare results_a.json results_b.json
```

Switch models with `bash server/serve.sh qwen3 | qwen35 | deepseek`; `common.py`
detects the served model and applies its recommended sampling.

## Notes on the local model

- Qwen3 is a hybrid reasoning model. `get_llm()` is fast mode (no thinking);
  `get_llm(thinking=True)` is used for planners, critics and judges.
- An 8B model is less reliable than frontier models. When a lesson misbehaves,
  treat it as material: figure out which guardrail (validation, retries, caps,
  checks in code) would have caught it. That discussion goes over well in interviews.
- Model ids, URL: `.env` `AGENT_BASE_URL`, `AGENT_MODEL`.
