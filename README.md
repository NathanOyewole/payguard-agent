# PayGuard Agent

AI-powered fraud-aware payment agent. Handles PayPal payments via natural language with real-time fraud detection.

## Features
- Natural language payment parsing ("Send $200 to vendor@example.com")
- Real-time fraud risk scoring (velocity, recipient novelty, amount anomaly, structuring)
- PayPal Sandbox integration for actual payment execution
- Risk levels: low (auto-approve), medium (review), high (block)

## Quick Start
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env  # Add PayPal credentials
python main.py
```

Open http://localhost:8000/../frontend/index.html

## Environment Variables
- `PAYPAL_CLIENT_ID` - PayPal sandbox client ID
- `PAYPAL_CLIENT_SECRET` - PayPal sandbox client secret

## Prize Targets
- Best Use of Agentic Commerce ($5,000)
- Best Use of PayPal + AI ($5,000)
- Most Impactful ($5,000)
- Best Demo Delivery ($5,000)

## License
MIT
