import { useState } from "react";

const API = "/predict";

const palette = {
  navy:    "#1B3A6B",
  blue:    "#2563EB",
  green:   "#059669",
  red:     "#E11D48",
  amber:   "#D97706",
  slate:   "#64748B",
  light:   "#EFF6FF",
  white:   "#FFFFFF",
  bg:      "#F8FAFC",
};

const INITIAL = {
  gender:          "Male",
  senior_citizen:  0,
  tenure:          24,
  monthly_charges: 65,
  total_charges:   1560,
  contract:        "Month-to-month",
  payment_method:  "Electronic check",
};

// ── Small UI primitives ────────────────────────────────────────────────────────
const Label = ({ children }) => (
  <label style={{ display: "block", fontSize: 12, fontWeight: 700,
    color: palette.slate, textTransform: "uppercase", letterSpacing: 0.8,
    marginBottom: 4 }}>
    {children}
  </label>
);

const inputStyle = {
  width: "100%", padding: "9px 12px", borderRadius: 8,
  border: "1.5px solid #CBD5E1", fontSize: 14, background: palette.white,
  boxSizing: "border-box", outline: "none",
  transition: "border-color .15s",
};

const Select = ({ value, onChange, options }) => (
  <select value={value} onChange={onChange} style={inputStyle}>
    {options.map(o => <option key={o}>{o}</option>)}
  </select>
);

// ── Risk gauge ─────────────────────────────────────────────────────────────────
const Gauge = ({ pct }) => {
  const color = pct < 35 ? palette.green : pct < 60 ? palette.amber : palette.red;
  return (
    <div style={{ margin: "12px 0" }}>
      <div style={{ display: "flex", justifyContent: "space-between",
        fontSize: 12, color: palette.slate, marginBottom: 4 }}>
        <span>0%</span><span>50%</span><span>100%</span>
      </div>
      <div style={{ background: "#E2E8F0", borderRadius: 99, height: 14, overflow: "hidden" }}>
        <div style={{
          width: `${pct}%`, height: "100%", background: color,
          borderRadius: 99, transition: "width .5s ease",
        }} />
      </div>
      <div style={{ textAlign: "center", marginTop: 6, fontSize: 26,
        fontWeight: 900, color, lineHeight: 1 }}>
        {pct.toFixed(1)}%
      </div>
      <div style={{ textAlign: "center", fontSize: 11, color: palette.slate }}>
        churn probability
      </div>
    </div>
  );
};

// ── Main App ───────────────────────────────────────────────────────────────────
export default function App() {
  const [form,    setForm]    = useState(INITIAL);
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  const set = (key) => (e) => {
    const val = e.target.type === "number" ? Number(e.target.value) : e.target.value;
    setForm(prev => ({ ...prev, [key]: val }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(API, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(form),
      });
      if (!res.ok) {
        const body = await res.json();
        throw new Error(body.detail || `HTTP ${res.status}`);
      }
      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const riskColor = result
    ? (result.risk_level === "Low" ? palette.green
      : result.risk_level === "Medium" ? palette.amber
      : palette.red)
    : palette.slate;

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", background: palette.bg,
      minHeight: "100vh", color: "#1E293B" }}>

      {/* Header */}
      <div style={{ background: palette.navy, color: palette.white,
        padding: "18px 32px" }}>
        <div style={{ fontSize: 11, letterSpacing: 2, opacity: 0.65,
          textTransform: "uppercase", fontFamily: "monospace" }}>
          Big Data Analytics — Assignment Demo
        </div>
        <div style={{ fontSize: 22, fontWeight: 800, marginTop: 4 }}>
          Bank Customer Churn Predictor
        </div>
        <div style={{ fontSize: 13, opacity: 0.75, marginTop: 2 }}>
          Trained on 10,000 customers · Logistic Regression / RF / Gradient Boosting
        </div>
      </div>

      <div style={{ maxWidth: 960, margin: "0 auto", padding: "28px 20px",
        display: "grid", gridTemplateColumns: "1fr 1fr", gap: 28 }}>

        {/* ── Form ── */}
        <div style={{ background: palette.white, borderRadius: 16,
          padding: 28, boxShadow: "0 2px 16px #0001" }}>
          <h2 style={{ margin: "0 0 20px", fontSize: 16, color: palette.navy }}>
            Customer Information
          </h2>

          <form onSubmit={handleSubmit}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

              <div>
                <Label>Gender</Label>
                <Select value={form.gender} onChange={set("gender")}
                  options={["Male", "Female"]} />
              </div>

              <div>
                <Label>Senior Citizen</Label>
                <Select value={form.senior_citizen} onChange={set("senior_citizen")}
                  options={[0, 1]} />
              </div>

              <div>
                <Label>Tenure (months)</Label>
                <input type="number" min={1} max={120} value={form.tenure}
                  onChange={set("tenure")} style={inputStyle} />
              </div>

              <div>
                <Label>Monthly Charges ($)</Label>
                <input type="number" min={1} step={0.01} value={form.monthly_charges}
                  onChange={set("monthly_charges")} style={inputStyle} />
              </div>

              <div>
                <Label>Total Charges ($)</Label>
                <input type="number" min={1} step={0.01} value={form.total_charges}
                  onChange={set("total_charges")} style={inputStyle} />
              </div>

              <div>
                <Label>Contract</Label>
                <Select value={form.contract} onChange={set("contract")}
                  options={["Month-to-month", "One year", "Two year"]} />
              </div>

              <div style={{ gridColumn: "span 2" }}>
                <Label>Payment Method</Label>
                <Select value={form.payment_method} onChange={set("payment_method")}
                  options={["Electronic check", "Mailed check",
                    "Bank transfer", "Credit card"]} />
              </div>

            </div>

            <button type="submit" disabled={loading} style={{
              marginTop: 24, width: "100%", padding: "12px 0",
              background: loading ? palette.slate : palette.blue,
              color: palette.white, border: "none", borderRadius: 10,
              fontSize: 15, fontWeight: 700, cursor: loading ? "default" : "pointer",
              transition: "background .2s",
            }}>
              {loading ? "Predicting …" : "Predict Churn"}
            </button>
          </form>

          {error && (
            <div style={{ marginTop: 16, padding: "12px 16px",
              background: "#FEF2F2", border: "1px solid #FECACA",
              borderRadius: 8, color: palette.red, fontSize: 13 }}>
              {error}
            </div>
          )}
        </div>

        {/* ── Result panel ── */}
        <div style={{ background: palette.white, borderRadius: 16,
          padding: 28, boxShadow: "0 2px 16px #0001",
          display: "flex", flexDirection: "column", gap: 16 }}>

          <h2 style={{ margin: 0, fontSize: 16, color: palette.navy }}>
            Prediction Result
          </h2>

          {!result && !loading && (
            <div style={{ flex: 1, display: "flex", alignItems: "center",
              justifyContent: "center", color: palette.slate,
              fontSize: 14, textAlign: "center", minHeight: 220 }}>
              Fill in the form and click<br /><b>Predict Churn</b> to see results.
            </div>
          )}

          {loading && (
            <div style={{ flex: 1, display: "flex", alignItems: "center",
              justifyContent: "center", color: palette.blue,
              fontSize: 14, minHeight: 220 }}>
              Running model…
            </div>
          )}

          {result && (
            <>
              {/* Verdict badge */}
              <div style={{
                background: result.churn ? "#FEF2F2" : "#F0FDF4",
                border: `2px solid ${result.churn ? palette.red : palette.green}`,
                borderRadius: 12, padding: "14px 20px", textAlign: "center",
              }}>
                <div style={{ fontSize: 26, fontWeight: 900,
                  color: result.churn ? palette.red : palette.green }}>
                  {result.churn_label}
                </div>
                <div style={{ fontSize: 12, color: palette.slate, marginTop: 4 }}>
                  Risk level: <b style={{ color: riskColor }}>{result.risk_level}</b>
                  &nbsp;·&nbsp;Model: {result.model_used}
                </div>
              </div>

              {/* Gauge */}
              <Gauge pct={result.confidence_pct} />

              {/* Key factors */}
              <div>
                <div style={{ fontSize: 12, fontWeight: 700,
                  color: palette.slate, textTransform: "uppercase",
                  letterSpacing: 0.8, marginBottom: 8 }}>
                  Top Risk Factors
                </div>
                {result.top_factors.map((f, i) => (
                  <div key={i} style={{
                    display: "flex", alignItems: "center", gap: 8,
                    padding: "6px 0", borderBottom: "1px solid #F1F5F9",
                    fontSize: 13,
                  }}>
                    <span style={{ width: 6, height: 6, borderRadius: "50%",
                      background: riskColor, flexShrink: 0 }} />
                    {f}
                  </div>
                ))}
              </div>

              {/* Raw probability */}
              <div style={{ fontSize: 12, color: palette.slate,
                background: "#F8FAFC", borderRadius: 8, padding: "8px 12px" }}>
                Raw probability: <b>{result.probability}</b>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Footer note */}
      <div style={{ textAlign: "center", padding: "0 0 24px",
        fontSize: 12, color: palette.slate }}>
        POST /predict → FastAPI → scikit-learn model → churn probability
      </div>
    </div>
  );
}
