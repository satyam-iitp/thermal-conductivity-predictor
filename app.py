"""
app.py  –  Flask web app for Thermal Conductivity Prediction
Run:  python app.py
Then open:  http://localhost:5000
"""
import os, json
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ── Load model artifacts ──────────────────────────────────────────────────────
MODEL_PATH    = "kl_predictor_lgbm.pkl"
FEATURES_PATH = "kl_top10_features.pkl"
STATS_PATH    = "feature_stats.pkl"

model    = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)
stats    = joblib.load(STATS_PATH) if os.path.exists(STATS_PATH) else {}

# Human-readable labels & descriptions for the top-10 features
FEATURE_META = {
    "Egap_fit": {
        "label": "Band Gap (Egap_fit)",
        "unit": "eV",
        "desc": "DFT-fitted electronic band gap. Zero for metals; positive for insulators/semiconductors.",
        "icon": "⚡"
    },
    "energy_atom": {
        "label": "DFT Energy per Atom",
        "unit": "eV/atom",
        "desc": "Total DFT ground-state energy normalized per atom.",
        "icon": "🔋"
    },
    "enthalpy_atom_x": {
        "label": "Enthalpy per Atom",
        "unit": "eV/atom",
        "desc": "Formation enthalpy per atom from DFT calculations.",
        "icon": "🌡️"
    },
    "volume_atom": {
        "label": "Volume per Atom",
        "unit": "Å³/atom",
        "desc": "Unit cell volume divided by number of atoms. Related to atomic packing density.",
        "icon": "📦"
    },
    "density_x": {
        "label": "Mass Density",
        "unit": "g/cm³",
        "desc": "Calculated crystallographic mass density of the compound.",
        "icon": "⚖️"
    },
    "natoms_x": {
        "label": "Atoms per Unit Cell",
        "unit": "atoms",
        "desc": "Total number of atoms in the primitive or conventional unit cell.",
        "icon": "🔵"
    },
    "nspecies_x": {
        "label": "Number of Species",
        "unit": "count",
        "desc": "Number of distinct chemical element types in the compound.",
        "icon": "🧪"
    },
    "valence_cell_iupac": {
        "label": "Valence Electrons (IUPAC)",
        "unit": "electrons",
        "desc": "Total number of valence electrons per unit cell (IUPAC counting).",
        "icon": "🌀"
    },
    "valence_cell_std": {
        "label": "Valence Electrons (Standard)",
        "unit": "electrons",
        "desc": "Total valence electron count using standard chemical convention.",
        "icon": "🌀"
    },
    "eentropy_atom": {
        "label": "Electronic Entropy per Atom",
        "unit": "eV/atom",
        "desc": "Smearing-induced electronic entropy from DFT Fermi–Dirac occupation.",
        "icon": "📊"
    },
    "Egap_x": {
        "label": "Band Gap (Egap_x)",
        "unit": "eV",
        "desc": "Cross-validated band gap estimate.",
        "icon": "⚡"
    },
    "spinF": {
        "label": "Spin at Fermi Level",
        "unit": "",
        "desc": "Spin polarization at the Fermi energy. Zero for non-magnetic materials.",
        "icon": "🧲"
    },
    "spin_atom_x": {
        "label": "Magnetic Moment per Atom",
        "unit": "μB/atom",
        "desc": "Net magnetic moment per atom in Bohr magnetons.",
        "icon": "🧲"
    },
    "point_group_order": {
        "label": "Point Group Order",
        "unit": "",
        "desc": "Number of symmetry operations in the crystallographic point group.",
        "icon": "🔷"
    },
    "PV_atom": {
        "label": "PV per Atom",
        "unit": "eV/atom",
        "desc": "Pressure × Volume contribution per atom (pV term in enthalpy).",
        "icon": "💠"
    },
}

def get_meta(feat):
    """Return metadata for a feature, with fallback defaults."""
    if feat in FEATURE_META:
        return FEATURE_META[feat]
    label = feat.replace("_", " ").title()
    return {"label": label, "unit": "", "desc": feat, "icon": "📌"}


# ── HTML Template ─────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>κ_L Predictor — Lattice Thermal Conductivity</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:        #0a0e14;
    --surface:   #111720;
    --card:      #161d28;
    --border:    #1e2d3d;
    --accent:    #00d4ff;
    --accent2:   #7b61ff;
    --gold:      #f5a623;
    --text:      #e2eaf5;
    --muted:     #6b7f96;
    --success:   #22d3a5;
    --error:     #ff4d6a;
    --glow:      0 0 30px rgba(0, 212, 255, 0.15);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── Background grid ── */
  body::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background-image:
      linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
  }

  /* ── Layout ── */
  .container { max-width: 980px; margin: 0 auto; padding: 0 24px; position: relative; z-index: 1; }

  /* ── Header ── */
  header {
    padding: 52px 0 40px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 48px;
  }
  .header-tag {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.2em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px;
  }
  .header-tag::before { content: '//'; opacity: 0.5; }
  h1 {
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #e2eaf5 30%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 16px;
  }
  .subtitle {
    font-size: 15px;
    color: var(--muted);
    max-width: 540px;
    line-height: 1.6;
    font-weight: 400;
  }
  .badge-row {
    display: flex; flex-wrap: wrap; gap: 10px; margin-top: 24px;
  }
  .badge {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    padding: 5px 12px;
    border-radius: 20px;
    border: 1px solid var(--border);
    color: var(--muted);
    letter-spacing: 0.05em;
  }
  .badge.accent { border-color: var(--accent); color: var(--accent); background: rgba(0,212,255,0.06); }

  /* ── Form grid ── */
  .form-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }
  .field {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    transition: border-color 0.2s, box-shadow 0.2s;
    position: relative;
    overflow: hidden;
  }
  .field::before {
    content: attr(data-icon);
    position: absolute; top: 16px; right: 18px;
    font-size: 18px; opacity: 0.25;
  }
  .field:focus-within {
    border-color: var(--accent);
    box-shadow: var(--glow);
  }
  .field-label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 4px;
  }
  .field-desc {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    margin-bottom: 14px;
    line-height: 1.5;
  }
  .input-wrap { position: relative; display: flex; align-items: center; }
  input[type="number"] {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: 'DM Mono', monospace;
    font-size: 14px;
    padding: 10px 14px;
    outline: none;
    transition: border-color 0.2s;
    appearance: textfield;
    -moz-appearance: textfield;
  }
  input[type="number"]::-webkit-inner-spin-button { display: none; }
  input[type="number"]:focus { border-color: var(--accent); }
  .unit-badge {
    position: absolute; right: 10px;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    pointer-events: none;
    letter-spacing: 0.05em;
  }
  .range-hint {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    margin-top: 6px;
    opacity: 0.7;
  }

  /* ── Actions ── */
  .actions {
    display: flex; gap: 12px; flex-wrap: wrap;
    margin-bottom: 40px;
  }
  .btn {
    padding: 14px 32px;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.04em;
    cursor: pointer;
    transition: all 0.2s;
    border: none;
  }
  .btn-primary {
    background: var(--accent);
    color: #000;
  }
  .btn-primary:hover { background: #33ddff; transform: translateY(-1px); box-shadow: 0 8px 24px rgba(0,212,255,0.3); }
  .btn-secondary {
    background: transparent;
    color: var(--muted);
    border: 1px solid var(--border);
  }
  .btn-secondary:hover { border-color: var(--muted); color: var(--text); }

  /* ── Result panel ── */
  #result {
    display: none;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 36px;
    margin-bottom: 48px;
    position: relative;
    overflow: hidden;
  }
  #result.show { display: block; animation: slideUp 0.4s ease; }
  #result.success { border-color: var(--success); }
  #result.error   { border-color: var(--error); }
  #result::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
  }
  #result.success::before { background: var(--success); }
  #result.error::before   { background: var(--error); }

  .result-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.15em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 12px;
  }
  .result-value {
    font-size: clamp(2.4rem, 6vw, 4.5rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--success);
    line-height: 1;
    margin-bottom: 8px;
  }
  .result-unit {
    font-family: 'DM Mono', monospace;
    font-size: 15px;
    color: var(--muted);
    margin-bottom: 20px;
  }
  .result-meta {
    display: flex; gap: 24px; flex-wrap: wrap;
    border-top: 1px solid var(--border);
    padding-top: 20px;
    margin-top: 20px;
  }
  .meta-item { display: flex; flex-direction: column; gap: 4px; }
  .meta-key {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }
  .meta-val {
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
  }
  .conductivity-bar {
    margin-top: 20px;
    background: var(--surface);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
  }
  .conductivity-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent2), var(--accent), var(--gold));
    transition: width 1s cubic-bezier(0.16, 1, 0.3, 1);
    width: 0%;
  }
  .bar-labels {
    display: flex; justify-content: space-between;
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    color: var(--muted);
    margin-top: 4px;
  }
  .error-msg { color: var(--error); font-size: 15px; font-weight: 600; }

  /* ── Spinner ── */
  .spinner {
    display: inline-block; width: 16px; height: 16px;
    border: 2px solid rgba(0,0,0,0.3);
    border-top-color: #000;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    margin-right: 8px; vertical-align: middle;
  }

  /* ── Footer ── */
  footer {
    border-top: 1px solid var(--border);
    padding: 32px 0;
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 12px;
  }
  .footer-text {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--muted);
  }

  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes slideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
  }
</style>
</head>
<body>
<div class="container">

  <header>
    <div class="header-tag">Materials ML · AFLOW Database</div>
    <h1>Lattice Thermal<br>Conductivity Predictor</h1>
    <p class="subtitle">
      Predict κ_L for crystalline compounds using a LightGBM model trained on
      5,530 DFT-computed materials from the AFLOW database.
    </p>
    <div class="badge-row">
      <span class="badge accent">LightGBM</span>
      <span class="badge accent">Top-10 SHAP Features</span>
      <span class="badge">log(1+κ_L) target</span>
      <span class="badge">AFLOW · 5,530 compounds</span>
    </div>
  </header>

  <form id="predict-form">
    <div class="form-grid" id="fields-container">
      <!-- Injected by JS -->
    </div>

    <div class="actions">
      <button type="submit" class="btn btn-primary" id="submit-btn">
        Predict κ_L
      </button>
      <button type="button" class="btn btn-secondary" onclick="fillExample()">
        Load Example
      </button>
      <button type="button" class="btn btn-secondary" onclick="clearForm()">
        Clear
      </button>
    </div>
  </form>

  <div id="result">
    <div id="result-content"></div>
  </div>

  <footer>
    <span class="footer-text">κ_L Predictor · LightGBM Lean Model</span>
    <span class="footer-text">Target: log(1 + κ_L) → expm1 → W/mK</span>
  </footer>

</div>

<script>
const FEATURES = {{ features_json | safe }};
const STATS    = {{ stats_json | safe }};

// Example compound values (diamond-like semiconductor, e.g. Si-like)
const EXAMPLES = {
  "Egap_fit":          1.12,
  "energy_atom":      -5.43,
  "enthalpy_atom_x":  -5.43,
  "volume_atom":       20.0,
  "density_x":         2.33,
  "natoms_x":          2,
  "nspecies_x":        1,
  "valence_cell_iupac": 8,
  "valence_cell_std":   8,
  "eentropy_atom":      0.001,
  "Egap_x":            1.12,
  "spinF":              0.0,
  "spin_atom_x":        0.0,
  "point_group_order":  48,
  "PV_atom":            0.0,
};

const META = {{ meta_json | safe }};

function buildForm() {
  const container = document.getElementById('fields-container');
  FEATURES.forEach(feat => {
    const m = META[feat] || { label: feat, unit: '', desc: feat, icon: '📌' };
    const s = STATS[feat] || {};
    const rangeText = s.min !== undefined
      ? `range: ${s.min.toFixed(2)} – ${s.max.toFixed(2)}  |  mean: ${s.mean.toFixed(2)}`
      : '';

    const div = document.createElement('div');
    div.className = 'field';
    div.dataset.icon = m.icon;
    div.innerHTML = `
      <div class="field-label">${m.label}</div>
      <div class="field-desc">${m.desc}</div>
      <div class="input-wrap">
        <input type="number" step="any" name="${feat}" id="f_${feat}"
               placeholder="e.g. ${s.mean !== undefined ? s.mean.toFixed(3) : '0'}"
               required>
        ${m.unit ? `<span class="unit-badge">${m.unit}</span>` : ''}
      </div>
      ${rangeText ? `<div class="range-hint">${rangeText}</div>` : ''}
    `;
    container.appendChild(div);
  });
}

function fillExample() {
  FEATURES.forEach(feat => {
    const el = document.getElementById('f_' + feat);
    if (el && EXAMPLES[feat] !== undefined) el.value = EXAMPLES[feat];
    else if (el) {
      const s = STATS[feat] || {};
      if (s.mean !== undefined) el.value = s.mean.toFixed(4);
    }
  });
}

function clearForm() {
  FEATURES.forEach(feat => {
    const el = document.getElementById('f_' + feat);
    if (el) el.value = '';
  });
  const res = document.getElementById('result');
  res.className = '';
  res.style.display = 'none';
}

document.getElementById('predict-form').addEventListener('submit', async e => {
  e.preventDefault();
  const btn = document.getElementById('submit-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Computing…';

  const data = {};
  FEATURES.forEach(feat => {
    const el = document.getElementById('f_' + feat);
    data[feat] = el ? parseFloat(el.value) : 0;
  });

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    const json = await resp.json();
    showResult(json);
  } catch (err) {
    showResult({ error: 'Network error: ' + err.message });
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Predict κ_L';
  }
});

function classify(kl) {
  if (kl < 1)   return { label: 'Very Low', color: '#6b7f96' };
  if (kl < 5)   return { label: 'Low', color: '#7b61ff' };
  if (kl < 20)  return { label: 'Moderate', color: '#00d4ff' };
  if (kl < 100) return { label: 'High', color: '#22d3a5' };
  return { label: 'Very High', color: '#f5a623' };
}

function showResult(json) {
  const res = document.getElementById('result');
  const content = document.getElementById('result-content');

  if (json.error) {
    res.className = 'show error';
    content.innerHTML = `<div class="error-msg">⚠ ${json.error}</div>`;
    res.style.display = 'block';
    return;
  }

  const kl  = json.kl_wpmk;
  const cls = classify(kl);
  const pct = Math.min(100, (Math.log1p(kl) / Math.log1p(500)) * 100);

  content.innerHTML = `
    <div class="result-label">Predicted Lattice Thermal Conductivity</div>
    <div class="result-value" style="color:${cls.color}">${kl.toFixed(3)}</div>
    <div class="result-unit">W · m⁻¹ · K⁻¹</div>
    <div class="conductivity-bar">
      <div class="conductivity-fill" id="kl-bar"></div>
    </div>
    <div class="bar-labels"><span>Insulator (~0)</span><span>Diamond (~2000)</span></div>
    <div class="result-meta">
      <div class="meta-item">
        <span class="meta-key">Classification</span>
        <span class="meta-val" style="color:${cls.color}">${cls.label}</span>
      </div>
      <div class="meta-item">
        <span class="meta-key">log(1 + κ_L)</span>
        <span class="meta-val">${json.log_kl.toFixed(4)}</span>
      </div>
      <div class="meta-item">
        <span class="meta-key">Model</span>
        <span class="meta-val">LightGBM Top-10</span>
      </div>
    </div>
  `;

  res.className = 'show success';
  res.style.display = 'block';
  res.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  // Animate bar
  setTimeout(() => {
    const bar = document.getElementById('kl-bar');
    if (bar) bar.style.width = pct + '%';
  }, 100);
}

buildForm();
</script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────
import json

@app.route("/")
def index():
    meta_dict = {f: get_meta(f) for f in features}
    return render_template_string(
        HTML,
        features_json=json.dumps(features),
        stats_json=json.dumps(stats),
        meta_json=json.dumps(meta_dict),
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        import pandas as pd
        row = pd.DataFrame([{f: float(data.get(f, 0)) for f in features}])
        log_kl = float(model.predict(row)[0])
        kl     = float(np.expm1(log_kl))
        return jsonify({"kl_wpmk": round(kl, 6), "log_kl": round(log_kl, 6)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/health")
def health():
    return jsonify({"status": "ok", "features": features})

if __name__ == "__main__":
    print("🚀 Starting κ_L Predictor at http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
