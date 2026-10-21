"""
A tiny MCP server (stdio transport) exposing a ticket tracker.
Used by lesson 19. Run standalone to inspect with the MCP Inspector:
    npx @modelcontextprotocol/inspector python langgraph_course/mcp_servers/tickets_server.py
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("tickets", instructions="Ticket tracker for the platform team.", log_level="WARNING")

TICKETS = {
    "PLAT-101": {"title": "Checkout latency p99 > 2s", "status": "open", "priority": "high", "owner": "dana"},
    "PLAT-102": {"title": "Rotate staging DB credentials", "status": "open", "priority": "medium", "owner": None},
    "PLAT-103": {"title": "Upgrade Python 3.11 -> 3.12", "status": "done", "priority": "low", "owner": "li"},
}


@mcp.tool()
def list_tickets(status: str = "open") -> list[dict]:
    """List tickets with the given status ('open' or 'done')."""
    return [{"id": k, **v} for k, v in TICKETS.items() if v["status"] == status]


@mcp.tool()
def assign_ticket(ticket_id: str, owner: str) -> str:
    """Assign a ticket to an owner."""
    if ticket_id not in TICKETS:
        return f"No ticket {ticket_id}"
    TICKETS[ticket_id]["owner"] = owner
    return f"{ticket_id} assigned to {owner}"


@mcp.resource("tickets://runbook")
def runbook() -> str:
    """Team runbook (an MCP *resource*: read-only context, not an action)."""
    return "Every open ticket must have an owner. Assign unowned tickets to the on-call engineer: sam."


@mcp.prompt()
def triage_prompt(team: str) -> str:
    """A reusable prompt template served by the MCP server."""
    return f"You are the triage assistant for the {team} team. Make sure every open ticket has an owner."


if __name__ == "__main__":
    mcp.run(transport="stdio")
