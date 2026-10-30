import os
import json
import httpx
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="PayGuard Agent")

# PayPal Sandbox config
PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID", "")
PAYPAL_CLIENT_SECRET = os.getenv("PAYPAL_CLIENT_SECRET", "")
PAYPAL_BASE_URL = "https://api-m.sandbox.paypal.com"

# In-memory transaction history for fraud detection
transaction_history = []


class PaymentIntent(BaseModel):
    text: str  # Natural language payment request


class PaymentRequest(BaseModel):
    amount: float
    recipient: str
    purpose: Optional[str] = None
    confirmed: bool = False


class FraudResult(BaseModel):
    risk_score: int  # 0-100
    risk_level: str  # low, medium, high
    flags: list[str]
    explanation: str
    recommendation: str  # approve, review, block


async def get_paypal_access_token() -> str:
    """Get PayPal OAuth2 access token."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{PAYPAL_BASE_URL}/v1/oauth2/token",
            auth=(PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET),
            data={"grant_type": "client_credentials"},
            headers={"Accept": "application/json"},
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail="PayPal auth failed")
        return resp.json()["access_token"]


async def create_paypal_payment(amount: float, recipient: str, purpose: str = "") -> dict:
    """Create a PayPal payment (sandbox)."""
    token = await get_paypal_access_token()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{PAYPAL_BASE_URL}/v2/checkout/orders",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={
                "intent": "CAPTURE",
                "purchase_units": [{
                    "amount": {"currency_code": "USD", "value": str(amount)},
                    "payee": {"email_address": recipient},
                    "description": purpose or "PayGuard payment",
                }],
            },
        )
        if resp.status_code not in (200, 201):
            raise HTTPException(status_code=500, detail=f"PayPal payment failed: {resp.text}")
        return resp.json()


def analyze_fraud(amount: float, recipient: str, purpose: str = "") -> FraudResult:
    """Run fraud detection heuristics on a payment."""
    flags = []
    risk_score = 0

    # 1. Velocity check: too many payments in last hour
    one_hour_ago = datetime.now() - timedelta(hours=1)
    recent = [t for t in transaction_history if t["timestamp"] > one_hour_ago]
    if len(recent) >= 5:
        flags.append("High velocity: 5+ payments in last hour")
        risk_score += 30
    elif len(recent) >= 3:
        flags.append("Moderate velocity: 3+ payments in last hour")
        risk_score += 15

    # 2. New recipient check
    known_recipients = {t["recipient"] for t in transaction_history}
    if recipient not in known_recipients:
        flags.append("New recipient: first payment to this address")
        risk_score += 20

    # 3. Amount anomaly: unusually large amount
    if transaction_history:
        avg_amount = sum(t["amount"] for t in transaction_history) / len(transaction_history)
        if amount > avg_amount * 3 and amount > 100:
            flags.append(f"Amount anomaly: ${amount} is 3x+ the average (${avg_amount:.2f})")
            risk_score += 25
    elif amount > 500:
        flags.append(f"Large first payment: ${amount}")
        risk_score += 20

    # 4. Structuring: multiple small payments to same recipient
    same_recipient_recent = [t for t in recent if t["recipient"] == recipient]
    if len(same_recipient_recent) >= 2:
        flags.append("Possible structuring: multiple payments to same recipient")
        risk_score += 25

    # 5. Suspicious recipient patterns
    suspicious_domains = ["tempmail", "guerrillamail", "mailinator", "throwaway"]
    if any(d in recipient.lower() for d in suspicious_domains):
        flags.append("Suspicious recipient domain")
        risk_score += 40

    # Determine risk level and recommendation
    if risk_score >= 60:
        risk_level = "high"
        recommendation = "block"
    elif risk_score >= 30:
        risk_level = "medium"
        recommendation = "review"
    else:
        risk_level = "low"
        recommendation = "approve"

    explanation = f"Risk score: {risk_score}/100. " + ("; ".join(flags) if flags else "No fraud signals detected.")

    return FraudResult(
        risk_score=risk_score,
        risk_level=risk_level,
        flags=flags,
        explanation=explanation,
        recommendation=recommendation,
    )


@app.post("/api/parse-intent")
async def parse_intent(intent: PaymentIntent):
    """Parse natural language into structured payment request using AI."""
    # Simple parsing (replace with LLM call in production)
    text = intent.text.lower()
    # Extract amount
    import re
    amount_match = re.search(r'\$?(\d+(?:\.\d{2})?)', text)
    amount = float(amount_match.group(1)) if amount_match else 0

    # Extract recipient (email pattern)
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    recipient = email_match.group(0) if email_match else ""

    # Extract purpose
    purpose = ""
    if "for" in text:
        purpose = text.split("for", 1)[-1].strip()

    return PaymentRequest(amount=amount, recipient=recipient, purpose=purpose)


@app.post("/api/analyze")
async def analyze_payment(req: PaymentRequest):
    """Analyze a payment for fraud before executing."""
    fraud = analyze_fraud(req.amount, req.recipient, req.purpose or "")
    return fraud


@app.post("/api/execute")
async def execute_payment(req: PaymentRequest):
    """Execute a PayPal payment after fraud check."""
    fraud = analyze_fraud(req.amount, req.recipient, req.purpose or "")

    if fraud.recommendation == "block":
        raise HTTPException(status_code=403, detail=f"Payment blocked: {fraud.explanation}")

    if fraud.recommendation == "review" and not req.confirmed:
        return {"status": "review_required", "fraud": fraud.dict()}

    # Execute payment
    result = await create_paypal_payment(req.amount, req.recipient, req.purpose or "")

    # Record in history
    transaction_history.append({
        "amount": req.amount,
        "recipient": req.recipient,
        "purpose": req.purpose,
        "timestamp": datetime.now(),
        "order_id": result.get("id"),
    })

    return {"status": "executed", "order": result, "fraud": fraud.dict()}


@app.get("/api/history")
async def get_history():
    """Get transaction history."""
    return transaction_history


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)