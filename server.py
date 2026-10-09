from fastmcp import FastMCP
from rag import rag_search

# Create MCP server
mcp = FastMCP("ERGO Claim Management Server")


# ============================================================
# 1. INTAKE TOOL
# ============================================================

@mcp.tool
def intake_claim(claim_id: str) -> dict:
    """Retrieve basic claim information."""

    return {
        "claim_id": claim_id,
        "policy_number": "PLO1001",
        "customer": "Shri Gondkar",
        "claim_type": "motor",
        "claim_amount": 85000,
        "status": "Under review"
    }


# ============================================================
# 2. DOCUMENT EXTRACTION TOOL
# ============================================================

@mcp.tool
def extract_document(document_text: str) -> dict:
    """Extract important information from a claim document."""

    # In real project:
    # Azure AI Document Intelligence would extract
    # structured information from PDF/TIFF/images.

    return {
        "document_type": "accident_report",
        "accident_date": "2026-09-20",
        "vehicle": "BMW",
        "damage_amount": 85000,
        "registration_number": "MH12AB123"
    }


# ============================================================
# 3. CLAIM VALIDATION TOOL
# ============================================================

@mcp.tool
def validate_claim(claim_amount: float, damage_amount: float) -> dict:
    """Check whether claim amount and document amount are consistent."""

    difference = abs(claim_amount - damage_amount)

    if difference <= 5000:
        return {
            "valid": True,
            "message": "Claim amount is consistent with document amount."
        }

    return {
        "valid": False,
        "message": "Claim amount differs significantly from document amount."
    }


# ============================================================
# 4. POLICY CHECK TOOL
# ============================================================

@mcp.tool
def check_policy(policy_number: str, claim_amount: float) -> dict:
    """Check policy coverage and calculate eligible amount."""

    policy_limit = 100000
    deductible = 5000

    if claim_amount <= policy_limit:

        eligible_amount = claim_amount - deductible

        return {
            "policy_number": policy_number,
            "covered": True,
            "policy_limit": policy_limit,
            "deductible": deductible,
            "eligible_amount": eligible_amount
        }

    return {
        "policy_number": policy_number,
        "covered": False,
        "policy_limit": policy_limit,
        "message": "Claim exceeds policy limit."
    }




# ============================================================
# 5. FRAUD RISK TOOL
# ============================================================

@mcp.tool
def check_fraud_risk(claim_amount: float) -> dict:
    """Simulate a fraud risk model."""

    # In the real ERGO project:
    # XGBoost / ML model would be called here.

    if claim_amount > 900000:
        risk = 0.75
    else:
        risk = 0.18

    if risk >= 0.7:
        risk_level = "HIGH"

    elif risk >= 0.4:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    return {
        "fraud_probability": risk,
        "risk_level": risk_level
    }


# ============================================================
# 6. RECOMMENDATION TOOL
# ============================================================

@mcp.tool
def generate_recommendation(
    validation_passed: bool,
    policy_covered: bool,
    fraud_risk_level: str
) -> dict:
    """Generate next step recommendation."""

    if not validation_passed:
        return {
            "recommendation": "Manual document review required."
        }

    if not policy_covered:
        return {
            "recommendation": "Policy coverage review required."
        }

    if fraud_risk_level == "HIGH":
        return {
            "recommendation": "Fraud investigation required."
        }

    return {
        "recommendation": "Claim can proceed to human review."
    }
###########################################################################
##RAG TOOL
############################################################################

@mcp.tool()
def search_claim_documents(
    query: str,
    top_k: int = 15,
    top_n: int = 5
):
    """
    Search ERGO insurance documents using
    Pinecone retrieval and reranking.
    """

    try:

        result = rag_search(
            query=query,
            top_k=top_k,
            top_n=top_n
        )

        return result

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }
    
# ============================================================
# START MCP SERVER
# ============================================================

if __name__ == "__main__":
    mcp.run()