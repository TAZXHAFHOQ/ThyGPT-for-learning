import os
import io
import csv
import json
import base64
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import streamlit as st
from PIL import Image

# ================== APP SETUP ==================
st.set_page_config(page_title="Thyroid AIGC-CAD (Research Demo)", page_icon="🩺", layout="wide")
st.markdown(
    """
<div style="padding:10px;border-radius:12px;background:#fff3cd;border:1px solid #ffeeba;">
<b>⚠️ Research demo only (not for clinical use).</b><br>
This app sends data to the selected LLM provider. Validate findings with a radiologist.
</div>
""",
    unsafe_allow_html=True,
)

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("Provider")
    provider = st.selectbox(
        "Choose model provider",
        [
            "OpenAI",
            "Google (Gemini)",
            "Anthropic",
            "Vertex AI • Llama 4 Maverick (multimodal)",
            "Vertex AI • Llama 4 Scout (long-context)",
            "Local model",
        ],
        index=0,
    )

    # Suggested defaults
    default_model = (
        "gpt-4o-mini" if provider == "OpenAI"
        else "gemini-2.0-flash" if provider.startswith("Google")
        else "claude-3-5-sonnet-20240620" if provider == "Anthropic"
        else "llama-4-maverick-17b-128e-instruct-maas" if "Maverick" in provider
        else "llama-4-scout-405b-instruct-maas" if "Scout" in provider
        else "local-stub"
    )
    model = st.text_input("Model name / ID", value=default_model,
                          help="For Vertex options this is pre-filled to the recommended ID.")

    st.divider()
    st.header("API Keys (non-Vertex providers)")
    openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", disabled=provider!="OpenAI")
    google_key = st.text_input("Google (Gemini) API Key", type="password", placeholder="AIza...", disabled=not provider.startswith("Google"))
    anthropic_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...", disabled=provider!="Anthropic")

    # Vertex credentials
    if provider.startswith("Vertex AI"):
        st.divider()
        st.header("Vertex AI Settings")
        gcp_project = st.text_input("GCP Project ID", placeholder="your-gcp-project")
        gcp_location = st.text_input("Location", value="us-east5")

        use_vertex_sa = st.checkbox(
            "Use Service Account JSON file",
            value=True,
            help="If unchecked, the app relies on Application Default Credentials on the host."
        )
        sa_file = st.file_uploader("Upload Service Account JSON", type=["json"], disabled=not use_vertex_sa)

    st.divider()
    st.header("Risk Thresholds")
    low_thr = st.slider("Low (H-NPV)", 0.0, 1.0, 0.30, 0.01)
    high_thr = st.slider("High (H-PPV)", 0.0, 1.0, 0.70, 0.01)

    st.divider()
    st.header("Export")
    enable_export = st.checkbox("Enable CSV export control", value=True)

# ================== PROMPTS ==================
SYSTEM_PROMPT = """You are a radiology copilot assisting with thyroid ultrasound.
Extract clear, structured findings and avoid hallucinations. If image quality is insufficient, say so."""

USER_PROMPT_TEMPLATE = """You are given a thyroid ultrasound image (B-mode typical).
1) Describe key sonographic features relevant to ACR TI-RADS (composition, echogenicity, shape, margin, echogenic foci).
2) Suggest an ACR TI-RADS category (TR1–TR5) with brief rationale.
3) Estimate malignancy probability (0.0–1.0). If unsure, return null.
4) Provide a short natural-language summary suitable for a radiologist.

Return STRICT JSON with this schema (no extra keys, no trailing text):
{
  "features": {
    "composition": "solid|mixed|spongiform/cystic|unknown",
    "echogenicity": "anechoic|hyperechoic|isoechoic|hypoechoic|markedly hypoechoic|unknown",
    "shape": "wider-than-tall|taller-than-wide|round|unknown",
    "margin": "smooth|ill-defined|lobulated/irregular|extrathyroidal extension|unknown",
    "echogenic_foci": "none|macrocalcifications|microcalcifications|peripheral (rim)|comet-tail artifacts|unknown"
  },
  "ti_rads": "TR1|TR2|TR3|TR4|TR5|unknown",
  "malignancy_probability": 0.0,
  "summary": "string"
}
"""

ERROR_CHECK_TEMPLATE = """You are checking an ultrasound report for internal consistency vs. the image.
- Identify missing, inconsistent, or side-confusion details (e.g., left/right mismatch).
- If you cannot verify from the image, say so.
Return JSON:
{
  "issues": [
    { "type": "omission|inconsistency|side_confusion|insertion|other", "detail": "string", "suggested_fix": "string" }
  ],
  "overall_comment": "string"
}
"""

CHAT_PROMPT_WRAP = """You are continuing a discussion about the last analyzed image.
Here is the structured context you previously produced:
{context_json}

User question:
{user_q}

Answer concisely and reference the structured features when relevant.
"""

# ================== DATA STRUCTURES ==================
@dataclass
class ImageResult:
    filename: str
    ti_rads: str
    prob: Optional[float]
    summary: str
    raw_json: Dict[str, Any]
    provider_raw: str

# ================== HELPERS ==================
def ensure_env_keys(provider_name: str) -> bool:
    if provider_name == "OpenAI":
        if not openai_key:
            st.error("Please provide OpenAI API key.")
            return False
        os.environ["OPENAI_API_KEY"] = openai_key
    elif provider_name.startswith("Google (Gemini)"):
        if not google_key:
            st.error("Please provide Google (Gemini) API key.")
            return False
        os.environ["GOOGLE_API_KEY"] = google_key
    elif provider_name == "Anthropic":
        if not anthropic_key:
            st.error("Please provide Anthropic API key.")
            return False
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    return True

def parse_json_strict(txt: str) -> Optional[Dict[str, Any]]:
    if not txt:
        return None
    try:
        return json.loads(txt)
    except Exception:
        pass
    try:
        s, e = txt.find("{"), txt.rfind("}")
        if s != -1 and e != -1:
            return json.loads(txt[s:e+1])
    except Exception:
        return None
    return None

def management_recommendation(prob: Optional[float], low: float, high: float) -> str:
    if prob is None:
        return "Uncertain risk: follow ACR TI-RADS guideline and clinical context."
    if prob < low:
        return "H-NPV bucket: if radiologist concurs, consider follow-up rather than FNA."
    if prob > high:
        return "H-PPV bucket: if radiologist concurs, consider escalating (e.g., surgery) without FNA."
    return "Moderately suspicious: consider FNA per ACR TI-RADS and clinical factors."

# ---------- Service Account JSON handling (upload file) ----------
def _load_sa_json_from_upload(sa_file) -> (Optional[Dict[str, Any]], Optional[str]):
    if sa_file is None:
        return None, "No service account JSON file uploaded."
    try:
        # Reset pointer if the file-like object was read earlier
        sa_file.seek(0)
        sa_dict = json.load(sa_file)
        return sa_dict, None
    except Exception as e:
        return None, f"Failed to parse service account JSON: {e}"

def _build_vertex_credentials_from_dict(sa_dict: Optional[Dict[str, Any]]):
    try:
        from google.oauth2 import service_account
    except Exception as e:
        return None, f"google-auth not available: {e}"
    if not sa_dict:
        return None, "No service account JSON provided."
    try:
        creds = service_account.Credentials.from_service_account_info(
            sa_dict,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return creds, None
    except Exception as e:
        return None, f"Failed to build credentials: {e}"

# ================== PROVIDER CALLS ==================
def call_openai_vision(model: str, b64_png: str, user_prompt: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_png}"}}
        ]
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":content}],
            temperature=0.1
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"__ERROR__ {e}"

def call_gemini_vision(model: str, b64_png: str, user_prompt: str) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model_obj = genai.GenerativeModel(model)
        img = {"mime_type":"image/png", "data": base64.b64decode(b64_png)}
        resp = model_obj.generate_content([user_prompt, img], safety_settings=None)
        return (resp.text or "").strip()
    except Exception as e:
        return f"__ERROR__ {e}"

def call_anthropic_vision(model: str, b64_png: str, user_prompt: str) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model=model,
            temperature=0.1,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type":"text","text":user_prompt},
                    {"type":"image","source":{"type":"base64","media_type":"image/png","data":b64_png}}
                ]
            }]
        )
        return msg.content[0].text.strip()
    except Exception as e:
        return f"__ERROR__ {e}"

def call_vertex_llama_maverick(model_id: str, b64_png: str, user_prompt: str,
                               project: str, location: str,
                               sa_dict: Optional[Dict[str, Any]], use_sa: bool) -> str:
    """
    Vertex AI Llama 4 Maverick (multimodal): image+text.
    Uses explicit SA credentials (uploaded file) to avoid metadata lookups.
    """
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel, Part, GenerationConfig

        creds = None
        if use_sa:
            creds, err = _build_vertex_credentials_from_dict(sa_dict)
            if err:
                return f"__ERROR__ {err}"

        vertexai.init(project=project, location=location, credentials=creds)
        model = GenerativeModel(model_id or "llama-4-maverick-17b-128e-instruct-maas")

        img_part = Part.from_data(mime_type="image/png", data=base64.b64decode(b64_png))
        gen_cfg = GenerationConfig(temperature=0.1)

        response = model.generate_content([user_prompt, img_part],
                                          generation_config=gen_cfg,
                                          safety_settings=None)
        return (getattr(response, "text", None) or str(response)).strip()

    except Exception as e:
        return f"__ERROR__ {e}"

def call_vertex_llama_scout(model_id: str, text_payload: str,
                            project: str, location: str,
                            sa_dict: Optional[Dict[str, Any]], use_sa: bool) -> str:
    """
    Vertex AI Llama 4 Scout (text-only long-context).
    Uses explicit SA credentials (uploaded file) to avoid metadata lookups.
    """
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel, GenerationConfig

        creds = None
        if use_sa:
            creds, err = _build_vertex_credentials_from_dict(sa_dict)
            if err:
                return f"__ERROR__ {err}"

        vertexai.init(project=project, location=location, credentials=creds)
        model = GenerativeModel(model_id or "llama-4-scout-405b-instruct-maas")
        gen_cfg = GenerationConfig(temperature=0.1)

        response = model.generate_content([text_payload],
                                          generation_config=gen_cfg,
                                          safety_settings=None)
        return (getattr(response, "text", None) or str(response)).strip()

    except Exception as e:
        return f"__ERROR__ {e}"

def run_local_model(image_bytes: bytes, prompt: str) -> str:
    # Minimal placeholder for offline testing
    return json.dumps({
        "features": {"composition":"unknown","echogenicity":"unknown"},
        "ti_rads": "unknown",
        "malignancy_probability": 0.5,
        "summary": "Local stub: no analysis."
    })

def call_provider(provider_name: str, model_id: str, b64_png: Optional[str], prompt: str,
                  # Vertex args (optional)
                  gcp_project: Optional[str] = None,
                  gcp_location: Optional[str] = None,
                  sa_dict: Optional[Dict[str, Any]] = None,
                  use_sa: bool = False) -> str:
    """
    Unified provider dispatcher.
    For Scout (text-only), pass b64_png=None and prompt should contain the textual payload.
    """
    if provider_name == "OpenAI":
        return call_openai_vision(model_id, b64_png or "", prompt)
    elif provider_name.startswith("Google (Gemini)"):
        return call_gemini_vision(model_id, b64_png or "", prompt)
    elif provider_name == "Anthropic":
        return call_anthropic_vision(model_id, b64_png or "", prompt)
    elif provider_name == "Vertex AI • Llama 4 Maverick (multimodal)":
        return call_vertex_llama_maverick(
            model_id=model_id or "llama-4-maverick-17b-128e-instruct-maas",
            b64_png=b64_png or "",
            user_prompt=prompt,
            project=gcp_project or "",
            location=gcp_location or "us-east5",
            sa_dict=sa_dict,
            use_sa=use_sa
        )
    elif provider_name == "Vertex AI • Llama 4 Scout (long-context)":
        return call_vertex_llama_scout(
            model_id=model_id or "llama-4-scout-405b-instruct-maas",
            text_payload=prompt,
            project=gcp_project or "",
            location=gcp_location or "us-east5",
            sa_dict=sa_dict,
            use_sa=use_sa
        )
    else:
        return run_local_model(base64.b64decode(b64_png) if b64_png else b"", prompt)

# ================== UI: INPUTS ==================
st.title("🩺 Multimodal LLM Copilot for Thyroid Nodules (Demo)")

left, right = st.columns([1,1])
with left:
    images = st.file_uploader("Upload ultrasound image(s) (PNG/JPG)", type=["png","jpg","jpeg"], accept_multiple_files=True)
with right:
    report_text = st.text_area(
        "Optional: Paste ultrasound report text (for consistency check / long-context Q&A)",
        placeholder="e.g., Right lobe nodule 12×9×10 mm, hypoechoic, microcalcifications...",
        height=150
    )

st.divider()

# ================== ANALYSIS ==================
results: List[ImageResult] = []

def analyze_single_image(upfile) -> None:
    """Analyze a single uploaded image with the chosen provider and render the UI card."""
    im = Image.open(upfile).convert("RGB")
    buf = io.BytesIO(); im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    if provider in ("OpenAI", "Google (Gemini)", "Anthropic"):
        if not ensure_env_keys(provider):
            return

    if provider == "Vertex AI • Llama 4 Scout (long-context)":
        # TEXT-only route: will output conservative JSON with 'unknown' as needed
        text_payload = (
            "You are a radiology copilot. The user provided a thyroid ultrasound image "
            "but this model endpoint is optimized for long-text reading and may not parse images. "
            "Using domain knowledge and the following generic instruction, provide conservative, "
            "uncertain-aware reasoning and return the required JSON schema with 'unknown' where appropriate.\n\n"
            + USER_PROMPT_TEMPLATE
        )
        raw = call_provider(
            provider, model, None, text_payload,
            gcp_project=gcp_project if provider.startswith("Vertex") else None,
            gcp_location=gcp_location if provider.startswith("Vertex") else None,
            sa_dict=_load_sa_json_from_upload(sa_file)[0] if use_vertex_sa and provider.startswith("Vertex") else None,
            use_sa=bool(use_vertex_sa and provider.startswith("Vertex"))
        )
    else:
        raw = call_provider(
            provider, model, b64, USER_PROMPT_TEMPLATE,
            gcp_project=gcp_project if provider.startswith("Vertex") else None,
            gcp_location=gcp_location if provider.startswith("Vertex") else None,
            sa_dict=_load_sa_json_from_upload(sa_file)[0] if use_vertex_sa and provider.startswith("Vertex") else None,
            use_sa=bool(use_vertex_sa and provider.startswith("Vertex"))
        )

    parsed = parse_json_strict(raw)
    ti = parsed.get("ti_rads","unknown") if parsed else "unknown"
    pr_val = parsed.get("malignancy_probability", None) if parsed else None
    if not isinstance(pr_val, (int, float)):
        pr_val = None
    summ = parsed.get("summary","") if parsed else ""

    # UI card
    col = st.container()
    with col:
        st.image(im, caption=upfile.name, use_column_width=True)
        st.markdown(f"**Model TI-RADS**: {ti}")
        if pr_val is not None:
            st.markdown(f"**Malignancy probability**: `{pr_val:.3f}`")
            st.progress(min(max(pr_val, 0.0), 1.0))
        else:
            st.markdown("**Malignancy probability**: `unknown`")
        st.markdown(f"**Summary**: {summ or '—'}")
        st.info(management_recommendation(pr_val, low_thr, high_thr))

        with st.expander("Show structured JSON"):
            st.json(parsed if parsed else {"error": "Unable to parse JSON.", "raw": raw[:1200]})
        with st.expander("Show raw model text"):
            st.code(raw[:4000])

    results.append(ImageResult(
        filename=upfile.name, ti_rads=ti, prob=pr_val, summary=summ, raw_json=parsed or {}, provider_raw=raw)
    )

# Process images (if any)
if images:
    st.subheader("Results")
    grid_cols = st.columns(2)
    for i, up in enumerate(images):
        with grid_cols[i % 2]:
            with st.spinner(f"Analyzing {up.name} with {provider} • {model} ..."):
                analyze_single_image(up)

# ================== REPORT CONSISTENCY CHECK ==================
if report_text and images:
    st.subheader("Report Error Check (optional)")
    ref_up = images[0]
    im = Image.open(ref_up).convert("RGB")
    buf = io.BytesIO(); im.save(buf, format="PNG")
    b64_first = base64.b64encode(buf.getvalue()).decode("utf-8")
    templ = ERROR_CHECK_TEMPLATE + f'\n\nReport text:\n"""\n{report_text}\n"""'

    if provider in ("OpenAI", "Google (Gemini)", "Anthropic"):
        if not ensure_env_keys(provider):
            st.stop()

    if provider == "Vertex AI • Llama 4 Scout (long-context)":
        payload = "Long-context report consistency check:\n" + templ
        with st.spinner("Checking report for consistency (Vertex Llama 4 Scout)..."):
            raw_err = call_provider(
                provider, model, None, payload,
                gcp_project=gcp_project if provider.startswith("Vertex") else None,
                gcp_location=gcp_location if provider.startswith("Vertex") else None,
                sa_dict=_load_sa_json_from_upload(sa_file)[0] if use_vertex_sa and provider.startswith("Vertex") else None,
                use_sa=bool(use_vertex_sa and provider.startswith("Vertex"))
            )
    else:
        with st.spinner(f"Checking report vs. image {ref_up.name} ..."):
            raw_err = call_provider(
                provider, model, b64_first, templ,
                gcp_project=gcp_project if provider.startswith("Vertex") else None,
                gcp_location=gcp_location if provider.startswith("Vertex") else None,
                sa_dict=_load_sa_json_from_upload(sa_file)[0] if use_vertex_sa and provider.startswith("Vertex") else None,
                use_sa=bool(use_vertex_sa and provider.startswith("Vertex"))
            )

    parsed_err = parse_json_strict(raw_err)
    if parsed_err:
        issues = parsed_err.get("issues", [])
        if issues:
            for i, it in enumerate(issues, 1):
                st.warning(f"#{i} • **{it.get('type','issue')}** — {it.get('detail','')}\n\nSuggested fix: {it.get('suggested_fix','')}")
        else:
            st.success("No specific issues detected from the provided text; still use human review.")
        st.caption(parsed_err.get("overall_comment",""))
        with st.expander("Show JSON"):
            st.json(parsed_err)
    else:
        st.error("Could not parse JSON from the error-checking response.")
        st.code(raw_err[:2000])

# ================== SIMPLE CHAT ==================
st.divider()
st.subheader("Follow-up Q&A with the Copilot")
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

user_q = st.text_input("Ask about features, TI-RADS rationale, or guidelines")
ask_btn = st.button("Ask")

if ask_btn and user_q.strip():
    if not results and not report_text:
        st.error("Analyze at least one image or provide a report text first.")
    else:
        context_json = json.dumps(results[-1].raw_json if results else {"note":"no-image-json"},
                                  ensure_ascii=False)
        context_prompt = CHAT_PROMPT_WRAP.format(context_json=context_json, user_q=user_q)

        if provider in ("OpenAI", "Google (Gemini)", "Anthropic"):
            if not ensure_env_keys(provider):
                st.stop()

        # For Scout: include report_text to exploit long-context
        if provider == "Vertex AI • Llama 4 Scout (long-context)":
            payload = "Long-context Q&A.\n\n" + (f"Report text:\n{report_text}\n\n" if report_text else "") + context_prompt
            raw_ans = call_provider(
                provider, model, None, payload,
                gcp_project=gcp_project if provider.startswith("Vertex") else None,
                gcp_location=gcp_location if provider.startswith("Vertex") else None,
                sa_dict=_load_sa_json_from_upload(sa_file)[0] if use_vertex_sa and provider.startswith("Vertex") else None,
                use_sa=bool(use_vertex_sa and provider.startswith("Vertex"))
            )
        else:
            # Use last image for grounding if available
            if images:
                im_last = Image.open(images[-1]).convert("RGB")
                buf = io.BytesIO(); im_last.save(buf, format="PNG")
                b64_last = base64.b64encode(buf.getvalue()).decode("utf-8")
            else:
                b64_last = None

            raw_ans = call_provider(
                provider, model, b64_last, context_prompt,
                gcp_project=gcp_project if provider.startswith("Vertex") else None,
                gcp_location=gcp_location if provider.startswith("Vertex") else None,
                sa_dict=_load_sa_json_from_upload(sa_file)[0] if use_vertex_sa and provider.startswith("Vertex") else None,
                use_sa=bool(use_vertex_sa and provider.startswith("Vertex"))
            )

        st.session_state.chat_log.append(("you", user_q))
        st.session_state.chat_log.append(("copilot", raw_ans))

for role, text in st.session_state.chat_log[-10:]:
    if role == "you":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Copilot:** {text}")

# ================== TI-RADS CALCULATOR (Manual) ==================
st.divider()
st.subheader("Manual TI-RADS Calculator (ACR 2017, simplified)")

comp = st.radio("Composition", ["Cystic/spongiform","Mixed","Solid"], horizontal=True)
echo = st.radio("Echogenicity", ["Anechoic/Isoechoic","Hypoechoic","Very hypoechoic"], horizontal=True)
shape = st.radio("Shape", ["Wider-than-tall","Taller-than-wide"], horizontal=True)
margin = st.radio("Margin", ["Smooth","Irregular","Extrathyroidal"], horizontal=True)
foci = st.multiselect("Echogenic foci", ["None","Macrocalcifications","Microcalcifications","Rim calcifications"])

score = 0
if comp=="Solid": score+=2
if echo=="Hypoechoic": score+=2
if echo=="Very hypoechoic": score+=3
if shape=="Taller-than-wide": score+=3
if margin=="Irregular": score+=2
if margin=="Extrathyroidal": score+=3
if "Microcalcifications" in foci: score+=3
if "Macrocalcifications" in foci: score+=1
if "Rim calcifications" in foci: score+=2

ti_calc = "TR1"
if score>=7: ti_calc="TR5"
elif score>=4: ti_calc="TR4"
elif score>=2: ti_calc="TR3"
elif score==1: ti_calc="TR2"

st.markdown(f"**Manual TI-RADS score = {score} → {ti_calc}**")

# ================== EXPORT CSV ==================
if enable_export and results:
    rows = [["filename","ti_rads","malignancy_probability","summary"]]
    for r in results:
        rows.append([r.filename, r.ti_rads, "" if r.prob is None else r.prob, r.summary])

    csv_buf = io.StringIO()
    w = csv.writer(csv_buf)
    w.writerows(rows)
    st.download_button("Download CSV", csv_buf.getvalue(), file_name="thyroid_results.csv", mime="text/csv")

st.caption("Based on an AIGC-CAD concept for thyroid US: image → LLM features → TI-RADS & malignancy probability → thresholded management. Research demo only.")
