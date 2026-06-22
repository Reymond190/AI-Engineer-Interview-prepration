Langgraph and Multiagents

Why langgraph and why not create agents from scratch?
-> missout on integrations on vector dbs and llms from differnt providers, etc.
-> langgraph solves the cyclic problem, uses nodes and edges instead of nested loops, when agent acts and checks results and does another thing.
-> You must manually track conversation history, tool outputs, and variables across complex loops. Langgraph provides statemanagement.
-> Multi-Agent Coordination: Langgraph can easily define a network of agents where one agent transfers control to another via edges.

To really understand why LangGraph is so powerful compared to writing custom loops, it helps to see code examples of its most crucial features: **State Management**, **Cyclic Workflows (Loops)**, and **Persistence (Time-Travel / Human-in-the-Loop)**.

Here are minimal, runnable examples for each using the current LangGraph API.

---

## 1. State Management & Cyclic Workflows (Loops)

In a standard script, passing data back and forth between functions while keeping track of what tools have been called gets messy. LangGraph handles this by passing a centralized `State` dictionary to every node.

This example creates a simple loop: a **Router** checks a value in the state and decides whether to loop back or finish.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. Define the State that will be passed between nodes
class AgentState(TypedDict):
    counter: int
    message: str

# 2. Define the Nodes (the processing steps)
def increment_node(state: AgentState):
    print(f"--- Node Processing --- Current Counter: {state['counter']}")
    return {"counter": state["counter"] + 1, "message": "Incremented!"}

# 3. Define Conditional Edges (The Router)
def router_condition(state: AgentState):
    if state["counter"] < 3:
        print("-> Condition Met: Looping back!")
        return "loop_back"
    print("-> Condition Met: Finishing!")
    return "finish"

# 4. Build the Graph
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("incrementor", increment_node)

# Add edges
builder.add_edge(START, "incrementor")

# Add conditional routing out of the incrementor node
builder.add_conditional_edges(
    "incrementor",
    router_condition,
    {
        "loop_back": "incrementor", # If router returns "loop_back", go to incrementor
        "finish": END               # If router returns "finish", go to END
    }
)

# Compile the graph
graph = builder.compile()

# 5. Run it!
initial_state = {"counter": 0, "message": "Starting"}
graph.invoke(initial_state)

```

---

## 2. Persistence & Checkpointing (Time-Travel)

LangGraph automatically saves a snapshot of the graph's state after *every single node execution* if you provide a memory checkpointer. This allows you to inspect history or resume a crashed graph.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver  # In-memory checkpointer

class SimpleState(TypedDict):
    text: str

def add_hello(state: SimpleState):
    return {"text": state["text"] + " Hello"}

def add_world(state: SimpleState):
    return {"text": state["text"] + " World"}

# Build graph with persistence
builder = StateGraph(SimpleState)
builder.add_node("hello_node", add_hello)
builder.add_node("world_node", add_world)

builder.add_edge(START, "hello_node")
builder.add_edge("hello_node", "world_node")
builder.add_edge("world_node", END)

# Crucial: Compile with a checkpointer
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Thread config identifies this specific conversation run
config = {"configurable": {"thread_id": "user_session_42"}}

# Run the graph
final_state = graph.invoke({"text": "Init:"}, config=config)
print("Final Output:", final_state)

# --- THE MAGIC: Time Travel ---
# We can look up the history of this exact thread ID
print("\n--- History Inspection ---")
for state_history in graph.get_state_history(config):
    print(f"Step: {state_history.next} | Current State Data: {state_history.values}")

```

---

## 3. Human-in-the-Loop (Interrupts)

Imagine your agent wants to execute a dangerous task (like deleting a file or spending money via an API). You want the graph to run, *pause right before the dangerous node*, wait for a human to type "yes", and then resume.

LangGraph handles this natively via `interrupt_before`.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class ApprovalState(TypedDict):
    action: str
    approved: bool

def prep_action(state: ApprovalState):
    return {"action": "DELETE_DATABASE"}

def execute_action(state: ApprovalState):
    print(f"🚀 EXECUTING DANGEROUS ACTION: {state['action']}!")
    return state

builder = StateGraph(ApprovalState)
builder.add_node("prep", prep_action)
builder.add_node("execute", execute_action)

builder.add_edge(START, "prep")
builder.add_edge("prep", "execute")
builder.add_edge("execute", END)

# Tell the compiler to PAUSE right before running the 'execute' node
memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["execute"])

config = {"configurable": {"thread_id": "secure_run_101"}}

# 1. Start the run
print("--- Starting Agent ---")
graph.invoke({"action": "None", "approved": False}, config=config)

# Check where the graph is
current_state = graph.get_state(config)
print(f"Graph Paused! Next node to run: {current_state.next}")

# 2. Simulate human input
human_response = input("Do you approve this action? (yes/no): ")

if human_response.lower() == "yes":
    # Update the graph state with human approval, then resume
    graph.update_state(config, {"approved": True}, as_node="prep")
    print("--- Resuming Agent ---")
    graph.invoke(None, config=config) # Passing None tells it to pick up where it left off
else:
    print("Action denied. Terminating.")

```

---

### How to run these:

You will need to install the core packages:

```bash
pip install langgraph

```

*(Note: These examples focus entirely on LangGraph's architecture, so you don't even need an OpenAI/Anthropic API key to run them and see how the logic flows!)*

Which of these architectural patterns fits closest to what you are trying to build?
