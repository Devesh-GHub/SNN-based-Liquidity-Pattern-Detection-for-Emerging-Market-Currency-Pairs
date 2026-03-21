# FastAPI Notes — BRICS SNN Project


---

## What is an endpoint?

An endpoint is a specific URL path on a server that accepts requests
and returns a response. It is defined by two things: a **path**
(like `/predict`) and an **HTTP method** (like `GET` or `POST`).

In our API:
- `GET /hello` → browser or client asks "are you there?"
  → server replies with a JSON message
- `POST /predict` → client sends price data in the request body
  → server runs the SNN model and returns a prediction

The analogy: an endpoint is like a specific counter at a bank.
Counter 1 handles deposits (POST data), Counter 2 handles balance
checks (GET data). You go to the right counter for the right task.

---

## What does uvicorn do?

Uvicorn is an **ASGI server** — it runs your FastAPI application
and listens for incoming HTTP requests on a port (default: 8000).

When you run:
```bash
uvicorn src.fastapi_hello:app --reload
```

- `src.fastapi_hello` = the Python module path to your file
- `app` = the FastAPI object inside that file
- `--reload` = automatically restart when you change the code
  (development mode only — remove in production)

FastAPI defines *what* to do with a request.
Uvicorn handles *how* to receive and serve it over the network.
They work together: FastAPI is the restaurant menu and kitchen,
uvicorn is the front door and waiter.

---

## What will your REAL API have? (Month 2)

### Endpoint 1: `GET /health`
Called by the dashboard every 30 seconds to confirm the model
server is alive and the SNN model is loaded in memory.
```
Response:
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "snn_v1",
  "timestamp": "2024-..."
}
```

### Endpoint 2: `POST /predict`
The core endpoint. The dashboard sends a 10-day window of INR/BRL
features, the SNN model runs inference, and the API returns a
settlement recommendation.
```
Input (JSON body):
{
  "price_window": [16.2, 16.3, 16.1, 16.4, 16.5,
                   16.3, 16.6, 16.4, 16.7, 16.5],
  "currency_pair": "INR/BRL",
  "threshold": 0.003
}

Output:
{
  "direction": 1,
  "confidence": 0.73,
  "spike_rate": 0.144,
  "settlement_signal": "SETTLE_NOW",
  "latency_ms": 2.3
}
```

### Endpoint 3: `POST /encode` (optional)
Accepts a raw price array and returns the spike-encoded binary
sequence — useful for the dashboard's spike visualisation panel.

---

## What input will /predict accept?

A JSON body with a `price_window` field:
- **Type:** array of 10 floats
- **Content:** the last 10 daily closing INR/BRL rates
- **Example:** `[16.2, 16.3, 16.1, 16.4, 16.5, 16.3, 16.6, 16.4, 16.7, 16.5]`

The API will internally:
1. Convert the 10 prices into returns (pct_change)
2. Build all 9 features using `src/feature_engineering.py`
3. Normalise using the train-fitted scaler (loaded from disk)
4. Run through the SpikingJelly SNN model
5. Return the prediction

The client (dashboard) only needs to send 10 numbers.
All the complexity is hidden inside the API.

---

## What output will /predict return?
```json
{
  "direction": 1,
  "confidence": 0.73,
  "spike_rate": 0.144,
  "settlement_signal": "SETTLE_NOW",
  "latency_ms": 2.3
}
```

| Field | Type | Meaning |
|---|---|---|
| `direction` | 0 or 1 | 1 = model predicts INR/BRL goes UP tomorrow |
| `confidence` | float [0,1] | SNN output probability (0.5 = uncertain) |
| `spike_rate` | float [0,1] | Fraction of input timesteps that fired a spike |
| `settlement_signal` | string | "SETTLE_NOW" if confident, "WAIT" if uncertain |
| `latency_ms` | float | Time taken for inference in milliseconds |

### Settlement signal logic
```
confidence >= 0.65  AND  direction == 1  →  "SETTLE_NOW"
confidence >= 0.65  AND  direction == 0  →  "WAIT"
confidence <  0.65                       →  "UNCERTAIN — monitor"
```

---

## How to test the API (Month 2)
```bash
# Start the server
uvicorn src.fastapi_hello:app --reload

# Test /health with curl
curl http://localhost:8000/health

# Test /predict with curl (Month 2)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"price_window": [16.2,16.3,16.1,16.4,16.5,16.3,16.6,16.4,16.7,16.5]}'

# Or just open the browser:
# http://localhost:8000/docs   ← click "Try it out" on any endpoint
```

---

## Key FastAPI concepts to remember

| Concept | What it means |
|---|---|
| `@app.get("/path")` | Register a function as a GET endpoint |
| `@app.post("/path")` | Register a function as a POST endpoint |
| `BaseModel` (Pydantic) | Define the shape of request/response JSON |
| `response_model=` | FastAPI auto-validates and documents the output |
| `/docs` | Auto-generated Swagger UI — free, always up to date |
| `--reload` | Dev mode — restart on code changes |
| HTTP 200 | Success |
| HTTP 422 | Validation error (wrong input shape/type) |
| HTTP 500 | Server error (bug in your endpoint function) |