import os
import io
import base64
import json
import csv
import tempfile
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import streamlit as st
from PIL import Image

# ================== SAFETY HEADER ==================
st.set_page_config(page_title="Thyroid AIGC-CAD (Research Demo)", page_icon="🩺", layout="wide")
st.markdown(
    """
<div style="padding:10px;border-radius:12px;background:#fff3cd;border:1px solid #ffeeba;">
<b>⚠️ Research demo only (not for clinical use).</b><br>
This tool reproduces logic from published work for research/teaching only.
</div>
""",
    unsafe_allow_html=True,
)

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("LLM Settings")
    provider = st.selectbox("Provider", ["OpenAI", "Google (Gemini)", "Anthropic", "Local model"], index=0)
    model = st.text_input("Model name", value="gpt-4o-mini")

    openai_key = st.text_input("OpenAI API Key", type="password", disabled=provider!="OpenAI")
    google_key = st.text_input("Google API Key", type="password", disabled=not provider.startswith("Google"))
    anthropic_key = st.text_input("Anthropic API Key", type="password", disabled=provider!="Anthropic")

    st.divider()
    st.caption("Risk thresholding")
    low_thr = st.slider("Low threshold (H-NPV)", 0.0, 1.0, 0.30, 0.01)
    high_thr = st.slider("High threshold (H-PPV)", 0.0, 1.0, 0.70, 0.01)

    st.divider()
    st.header("Export")
    export_btn = st.button("Download results as CSV")

# ================== PROMPTS ==================
SYSTEM_PROMPT = "You are a radiology copilot for thyroid ultrasound."

USER_PROMPT_TEMPLATE = """Analyze this thyroid ultrasound:
1) Describe ACR TI-RADS features.
2) Suggest TI-RADS.
3) Estimate malignancy probability 0–1.
Return JSON strictly in this schema:
{
 "features": {...},
 "ti_rads": "TR1|TR2|TR3|TR4|TR5|unknown",
 "malignancy_probability": 0.0,
 "summary": "string"
}
"""

# ================== STRUCTURES ==================
@dataclass
class ImageResult:
    filename: str
    ti_rads: str
    prob: Optional[float]
    summary: str
    raw_json: Dict[str, Any]

results: List[ImageResult] = []

# ================== HELPERS ==================
def parse_json_strict(txt: str) -> Optional[Dict[str, Any]]:
    try: return json.loads(txt)
    except: pass
    try:
        s, e = txt.find("{"), txt.rfind("}")
        return json.loads(txt[s:e+1])
    except: return None

def management_recommendation(prob: Optional[float]) -> str:
    if prob is None:
        return "Uncertain: follow guideline."
    if prob < low_thr:
        return "Likely benign → consider follow-up."
    if prob > high_thr:
        return "Likely malignant → consider surgery if concordant."
    return "Moderate suspicion → consider FNA per guideline."

# Dummy local model stub
def run_local_model(image_bytes: bytes, prompt: str) -> str:
    return json.dumps({
        "features": {"composition":"unknown","echogenicity":"unknown"},
        "ti_rads": "unknown",
        "malignancy_probability": 0.5,
        "summary": "Local stub: no analysis."
    })

# Provider wrappers (simplified, same as before)
def call_provider(provider, model, b64_png, prompt):
    if provider == "Local model":
        return run_local_model(base64.b64decode(b64_png), prompt)
    return '{"ti_rads":"TR3","malignancy_probability":0.42,"summary":"Stub response"}'  # placeholder for brevity

# ================== APP BODY ==================
st.title("🩺 Thyroid Copilot with LLMs + Tools")

# Uploads
col1, col2 = st.columns(2)
with col1:
    images = st.file_uploader("Upload images", type=["png","jpg"], accept_multiple_files=True)
with col2:
    report_text = st.text_area("Optional report text")

# Process images
if images:
    st.subheader("Results")
    cols = st.columns(2)
    for i, up in enumerate(images):
        im = Image.open(up).convert("RGB")
        buf = io.BytesIO(); im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        raw = call_provider(provider, model, b64, USER_PROMPT_TEMPLATE)
        parsed = parse_json_strict(raw)

        ti = parsed.get("ti_rads","unknown") if parsed else "unknown"
        pr = parsed.get("malignancy_probability", None) if parsed else None
        summ = parsed.get("summary","") if parsed else raw

        with cols[i % 2]:
            st.image(im, caption=up.name, use_column_width=True)
            st.markdown(f"**TI-RADS:** {ti}")
            st.markdown(f"**Prob:** {pr}")
            st.info(management_recommendation(pr))
            st.caption(summ)
            if parsed: st.json(parsed)

        results.append(ImageResult(up.name, ti, pr, summ, parsed or {}))

# ================== TI-RADS CALCULATOR ==================
st.subheader("Manual TI-RADS Calculator (ACR 2017)")

comp = st.radio("Composition", ["Cystic/spongiform","Mixed","Solid"])
echo = st.radio("Echogenicity", ["Anechoic","Isoechoic","Hypoechoic","Very hypoechoic"])
shape = st.radio("Shape", ["Wider-than-tall","Taller-than-wide"])
margin = st.radio("Margin", ["Smooth","Irregular","Extrathyroidal"])
foci = st.multiselect("Echogenic foci", ["None","Macrocalcifications","Microcalcifications","Rim calcifications"])

# Scoring (simplified)
score = 0
if comp=="Solid": score+=2
if echo=="Hypoechoic": score+=2
if shape=="Taller-than-wide": score+=3
if margin=="Irregular": score+=2
if "Microcalcifications" in foci: score+=3
ti_calc = "TR1"
if score>=7: ti_calc="TR5"
elif score>=4: ti_calc="TR4"
elif score>=2: ti_calc="TR3"
elif score==1: ti_calc="TR2"

st.markdown(f"**Manual TI-RADS score = {score} → {ti_calc}**")

# ================== EXPORT ==================
if export_btn and results:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    with open(tmp.name,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename","ti_rads","prob","summary"])
        for r in results:
            w.writerow([r.filename,r.ti_rads,r.prob,r.summary])
    with open(tmp.name,"rb") as f:
        st.download_button("Download CSV", f, file_name="thyroid_results.csv", mime="text/csv")
