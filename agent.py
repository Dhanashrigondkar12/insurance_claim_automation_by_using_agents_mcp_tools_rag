import asyncio
from pathlib import Path
from typing import TypedDict

from fastmcp import Client
from langgraph.graph import StateGraph, START, END

from config import GEMINI_API_KEY
# from config import HUGGINGFACE_aPI_KEY


# ============================================================
# CLAIM STATE
# ============================================================

class ClaimState(TypedDict):
    claim_id: str
    claim: dict
    document: dict
    validation: dict
    policy: dict
    fraud: dict
    recommendation: dict


# ============================================================
# MCP CONNECTION
# ============================================================

async def call_mcp_tool(tool_name: str, arguments: dict):

    # Find server.py in the same folder as agent.py
    server_path = Path(__file__).parent / "server.py"

    client = Client(server_path)

    async with client:

        result = await client.call_tool(
            tool_name,
            arguments
        )

        return result


# ============================================================
# 1. INTAKE NODE
# ============================================================

async def intake_node(state: ClaimState):

    print("\n[1] Calling MCP Intake Tool...")

    result = await call_mcp_tool(
        "intake_claim",
        {
            "claim_id": state["claim_id"]
        }
    )

    return {
        "claim": result.data
    }


# ============================================================
# 2. DOCUMENT EXTRACTION NODE
# ============================================================

async def document_node(state: ClaimState):

    print("\n[2] Calling MCP Document Extraction Tool...")

    document_text = """
    Accident Report

    Vehicle: BMW
    Accident Date: 2026-09-20
    Damage Amount: 85000
    Registration: MH12AB123
    """

    result = await call_mcp_tool(
        "extract_document",
        {
            "document_text": document_text
        }
    )

    return {
        "document": result.data
    }


# ============================================================
# 3. VALIDATION NODE
# ============================================================

async def validation_node(state: ClaimState):

    print("\n[3] Calling MCP Validation Tool...")

    claim = state["claim"]
    document = state["document"]

    result = await call_mcp_tool(
        "validate_claim",
        {
            "claim_amount": claim["claim_amount"],
            "damage_amount": document["damage_amount"]
        }
    )

    return {
        "validation": result.data
    }


# ============================================================
# 4. POLICY NODE
# ============================================================

async def policy_node(state: ClaimState):

    print("\n[4] Calling MCP Policy Check Tool...")

    claim = state["claim"]

    result = await call_mcp_tool(
        "check_policy",
        {
            "policy_number": claim["policy_number"],
            "claim_amount": claim["claim_amount"]
        }
    )

    return {
        "policy": result.data
    }


# ============================================================
# 5. FRAUD NODE
# ============================================================

async def fraud_node(state: ClaimState):

    print("\n[5] Calling MCP Fraud Risk Tool...")

    claim = state["claim"]

    result = await call_mcp_tool(
        "check_fraud_risk",
        {
            "claim_amount": claim["claim_amount"]
        }
    )

    return {
        "fraud": result.data
    }


# ============================================================
# 6. RECOMMENDATION NODE
# ============================================================

async def recommendation_node(state: ClaimState):

    print("\n[6] Calling MCP Recommendation Tool...")

    validation = state["validation"]
    policy = state["policy"]
    fraud = state["fraud"]

    result = await call_mcp_tool(
        "generate_recommendation",
        {
            "validation_passed": validation["valid"],
            "policy_covered": policy["covered"],
            "fraud_risk_level": fraud["risk_level"]
        }
    )

    return {
        "recommendation": result.data
    }


# ============================================================
# LANGGRAPH WORKFLOW
# ============================================================

builder = StateGraph(ClaimState)


builder.add_node("intake", intake_node)
builder.add_node("document", document_node)
builder.add_node("validation", validation_node)
builder.add_node("policy", policy_node)
builder.add_node("fraud", fraud_node)
builder.add_node("recommendation", recommendation_node)


# Workflow

builder.add_edge(START, "intake")

builder.add_edge("intake", "document")

builder.add_edge("document", "validation")

builder.add_edge("validation", "policy")

builder.add_edge("policy", "fraud")

builder.add_edge("fraud", "recommendation")

builder.add_edge("recommendation", END)


# Compile graph

graph = builder.compile()


# ============================================================
# MAIN
# ============================================================

async def main():

    initial_state = {

        "claim_id": "CLM1001",

        "claim": {},

        "document": {},

        "validation": {},

        "policy": {},

        "fraud": {},

        "recommendation": {}
    }


    result = await graph.ainvoke(initial_state)


    print("\n")
    print("==============================================")
    print("       ERGO CLAIM MANAGEMENT RESULT")
    print("==============================================")


    print("\n1. CLAIM INFORMATION")
    print(result["claim"])


    print("\n2. DOCUMENT EXTRACTION")
    print(result["document"])


    print("\n3. VALIDATION")
    print(result["validation"])


    print("\n4. POLICY CHECK")
    print(result["policy"])


    print("\n5. FRAUD RISK")
    print(result["fraud"])


    print("\n6. FINAL RECOMMENDATION")
    print(result["recommendation"])


    print("\n==============================================")
    print("             WORKFLOW COMPLETED")
    print("==============================================")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())