# Bank Churn Prediction — Demo

Three-layer project: **ML training → API server → React UI**

```
ml/               ← train model, export to server/model/churn_model.pkl
server/           ← FastAPI: loads model, serves POST /predict
frontend/         ← React + Vite: form UI → calls /predict
```

---

## Step 1 — Train the model

```bash
cd ml
pip install -r requirements.txt
python train_model.py
```

What it does:
- Loads the CSV dataset
- Encodes Gender, Contract, PaymentMethod
- Engineers `ChargesPerMonth` and `ChargesTenureInteract` features
- Trains Logistic Regression, Random Forest, and Gradient Boosting (all with class balancing)
- Picks the best model by F1 score
- Exports `server/model/churn_model.pkl` (model + scaler + label maps bundled together)

Custom CSV path:
```bash
python train_model.py --data "C:/path/to/your/dataset.csv"
```

---

## Step 2 — Start the API server

```bash
cd server
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Server imports the pickle bundle at startup. Open http://localhost:8000/docs for the
interactive Swagger UI where you can test the API directly.

Key endpoints:
| Method | Path          | Description                        |
|--------|---------------|------------------------------------|
| POST   | /predict      | Returns churn prediction + factors |
| GET    | /model-info   | Shows which model is loaded        |
| GET    | /health       | Health check                       |

Example request body:
```json
{
  "gender": "Male",
  "senior_citizen": 0,
  "tenure": 6,
  "monthly_charges": 85.0,
  "total_charges": 510.0,
  "contract": "Month-to-month",
  "payment_method": "Electronic check"
}
```

---

## Step 3 — Start the React frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — fill in the customer form and click **Predict Churn**.
