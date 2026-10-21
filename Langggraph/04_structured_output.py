"""
04 - Structured output and validation loops
===========================================

Interview topics
----------------
- Agents talk to code through structured output: router decisions, plans,
  extracted records, grades. Define the shape with Pydantic and validate it.
- `llm.with_structured_output(Schema, method=...)`:
    "json_schema"     provider-native constrained decoding (OpenAI strict mode,
                      vLLM guided decoding). Output always parses.
    "function_calling" schema exposed as a forced tool call.
    "json_mode"       "return JSON" mode; shape not enforced.
  Without any of these: prompt + parse + validate-and-retry (lesson 03b).
- Constrained decoding guarantees *syntax*, not *semantics*. Business rules
  (totals add up, dates in range) still need validators, and the error should
  be fed back to the model so it can fix it. Here that loop is drawn as a
  graph: extract -> (valid? done : retry with error).
- include_raw=True returns {"raw", "parsed", "parsing_error"} instead of
  raising, which is what you want inside a graph node.
- Reasoning models: think first, then constrain only the final answer. vLLM's
  reasoning parser applies the JSON grammar after </think>, so thinking=True
  still works with structured output.

Run:  python langgraph_course/04_structured_output.py
"""

from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from common import banner, get_llm, show


class LineItem(BaseModel):
    description: str
    quantity: int = Field(gt=0)
    unit_price: float = Field(ge=0)


class Invoice(BaseModel):
    vendor: str
    currency: Literal["USD", "EUR", "GBP"]
    items: list[LineItem]
    total: float = Field(description="The amount due as printed on the invoice")

    @field_validator("total")
    @classmethod
    def total_matches_items(cls, v: float, info: ValidationInfo) -> float:
        items = info.data.get("items") or []
        expected = round(sum(i.quantity * i.unit_price for i in items), 2)
        if items and abs(expected - v) > 0.01:
            raise ValueError(f"total {v} != sum of line items {expected}. Use the computed sum.")
        return v


# The printed "Amount due" is wrong on purpose (items sum to 644.00). A model that
# copies 650 fails the business-rule validator and has to correct itself.
INVOICE_TEXT = """
ACME Office Supply - Invoice #4471
3 x Ergonomic chair @ 189.00 EUR
10 x Notebook pack @ 4.50 EUR
1 x Desk lamp @ 32.00 EUR
Amount due: 650.00 EUR
"""

# Pass the JSON *schema* (not the class): decoding is constrained to the shape, we get
# a dict back, and the business rules are checked by Pydantic in our own node.
# (Passing the class makes the OpenAI SDK validate inside the API call and raise.)
extractor = get_llm().with_structured_output(Invoice.model_json_schema(), method="json_schema", include_raw=True)


class State(TypedDict, total=False):
    text: str
    history: list[BaseMessage]  # extraction conversation, including error feedback
    attempts: int
    error: str
    invoice: Invoice


def extract(state: State) -> dict:
    history = state.get("history") or [HumanMessage(f"Extract this invoice:\n{state['text']}")]
    result = extractor.invoke(history)
    attempts = state.get("attempts", 0) + 1
    try:
        if result["parsing_error"]:
            raise result["parsing_error"]
        return {"invoice": Invoice.model_validate(result["parsed"]), "attempts": attempts, "error": ""}
    except Exception as e:  # noqa: BLE001  (ValidationError or parsing error)
        error = " ".join(str(e).split())[:300]
    show(f"validation failed (attempt {attempts})", error)
    feedback = [AIMessage(result["raw"].content), HumanMessage(f"Validation failed: {error}\nReturn corrected JSON.")]
    return {"history": history + feedback, "attempts": attempts, "error": error}


def should_retry(state: State) -> Literal["extract", "__end__"]:
    return "extract" if state.get("error") and state["attempts"] < 3 else END


builder = StateGraph(State)
builder.add_node("extract", extract)
builder.add_edge(START, "extract")
builder.add_conditional_edges("extract", should_retry)
graph = builder.compile()


if __name__ == "__main__":
    banner("Structured output -> business validation -> retry with feedback")
    out = graph.invoke({"text": INVOICE_TEXT})
    show("attempts", out["attempts"])
    show("invoice", out.get("invoice") or out.get("error"), width=1200)

    banner("Thinking + structured output (reasoning first, JSON after </think>)")
    class Answer(BaseModel):
        reasoning_summary: str
        answer: int

    llm = get_llm(thinking=True).with_structured_output(Answer, method="json_schema")
    show("answer", llm.invoke("A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How many cents is the ball?"), width=2000)
