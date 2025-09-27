import os
import io
import base64
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import streamlit as st
from PIL import Image

# ---- Safety banner (research demo only) --------------------------------------
st.set_page_config(page_title="Thyroid AIGC-CAD (Research Demo)", page_icon="🩺", layout="wide")
st.markdown(
    """
<div style="padding:10px;border-radius:12px;background:#fff3cd;border:1px solid #ffeeba;">
<b>⚠️ Research demo only (not for clinical use).</b><br>
Do not use for diagnosis or patient management. This app sends inputs to an LLM provider you select.
</div>
""",
    unsafe_allow_html=True,
)

# ---- Side panel: provider/model/keys -----------------------------------------
with st.sidebar:
    st.header("LLM Settings")
    provider = st.selectbox("Provider", ["OpenAI", "Google (Gemini)", "Anthropic"], index=0)
    model = st.text_input(
        "Model name",
        value=(
            "gpt-4o-mini" if provider == "OpenAI"
            else "gemini-2.0-flash" if provider.startswith("Google")
            else "claude-3-5-sonnet-20240620"
        ),
        help="Change to a vision-capable model for images."
    )
    st.divider()
    openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", disabled=provider!="OpenAI")
    google_key = st.text_input("Google (Gemini) API Key", type="password", placeholder="AIza...", disabled=not provider.startswith("Google"))
    anthropic_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...", disabled=provider!="Anthropic")

    st.divider()
    st.caption("Risk thresholding (inspired by paper’s H-NPV / H-PPV rules)")
    low_thr = st.slider("Low threshold (H-NPV)", 0.0, 1.0, 0.30, 0.01)
    high_thr = st.slider("High threshold (H-PPV)", 0.0, 1.0, 0.70, 0.01)
    st.caption("• score < Low → high NPV bucket (likely benign, follow-up)\n"
               "• Low ≤ score ≤ High → moderate suspicion (consider ACR rules / FNA by guideline)\n"
               "• score > High → high PPV bucket (consider surgery if concordant)")

# ---- App header ---------------------------------------------------------------
st.title("🩺 Multimodal LLM Copilot for Thyroid Nodules (Demo)")
st.write(
    "Upload ultrasound image(s) and (optionally) a draft report. The LLM will "
    "summarize salient features, suggest ACR TI-RADS category, and produce a malignancy probability. "
    "A management recommendation is then derived using threshold rules inspired by the paper. "
)

# ---- Input widgets ------------------------------------------------------------
col_l, col_r = st.columns([1,1])
with col_l:
    images = st.file_uploader(
        "Upload thyroid ultrasound image(s) (PNG/JPG)", type=["png","jpg","jpeg"], accept_multiple_files=True
    )
with col_r:
    report_text = st.text_area(
        "Optional: Paste the ultrasound report text for error checking",
        placeholder="e.g., Right lobe nodule 12×9×10 mm, hypoechoic, microcalcifications...",
        height=150
    )

st.divider()

# ---- Prompt templates ---------------------------------------------------------
SYSTEM_PROMPT = """You are a radiology copilot assisting with thyroid ultrasound.
Extract clear, structured findings and avoid hallucinations. If image quality is insufficient, say so."""

# We request a strict JSON to make post-processing robust.
USER_PROMPT_TEMPLATE = """You are given a thyroid ultrasound image (grayscale B-mode typical).
1) Describe key sonographic features relevant to ACR TI-RADS (composition, echogenicity, shape, margin, echogenic foci).
2) Suggest an ACR TI-RADS category (TR1–TR5) with brief rationale.
3) Estimate malignancy probability (0.0–1.0). If unsure, return null.
4) Provide a short natural-language summary suitable for a radiologist.

Return STRICT JSON with this schema (no extra keys, no trailing text):

{{
  "features": {{
    "composition": "solid|mixed|spongiform/cystic|unknown",
    "echogenicity": "anechoic|hyperechoic|isoechoic|hypoechoic|markedly hypoechoic|unknown",
    "shape": "wider-than-tall|taller-than-wide|round|unknown",
    "margin": "smooth|ill-defined|lobulated/irregular|extrathyroidal extension|unknown",
    "echogenic_foci": "none|macrocalcifications|microcalcifications|peripheral (rim)|comet-tail artifacts|unknown"
  }},
  "ti_rads": "TR1|TR2|TR3|TR4|TR5|unknown",
  "malignancy_probability": 0.0,
  "summary": "string"
}}
"""

ERROR_CHECK_TEMPLATE = """You are checking an ultrasound report for internal consistency vs. the image.
- Identify missing, inconsistent, or side-confusion details (e.g., left/right mismatch).
- If you cannot verify from the image, say so.
Return JSON:
{{
  "issues": [
    {{ "type": "omission|inconsistency|side_confusion|insertion|other", "detail": "string", "suggested_fix": "string" }}
  ],
  "overall_comment": "string"
}}
"""

# ---- Provider clients (lazy import to avoid hard dependency explosions) -------
def b64_image(file) -> str:
    return base64.b64encode(file.read()).decode("utf-8")

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
        return f'__ERROR__ {e}'

def call_gemini_vision(model: str, b64_png: str, user_prompt: str) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model_obj = genai.GenerativeModel(model)
        img = {"mime_type":"image/png", "data": base64.b64decode(b64_png)}
        resp = model_obj.generate_content([user_prompt, img], safety_settings=None)
        return resp.text.strip()
    except Exception as e:
        return f'__ERROR__ {e}'

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
        return f'__ERROR__ {e}'

def call_provider(provider: str, model: str, b64_png: str, prompt: str) -> str:
    if provider == "OpenAI":
        return call_openai_vision(model, b64_png, prompt)
    elif provider.startswith("Google"):
        return call_gemini_vision(model, b64_png, prompt)
    else:
        return call_anthropic_vision(model, b64_png, prompt)

# ---- Parsing helpers ----------------------------------------------------------
import json

@dataclass
class ImageResult:
    ti_rads: str
    prob: Optional[float]
    summary: str
    raw_json: Dict[str, Any]
    provider_raw: str

def parse_json_strict(txt: str) -> Optional[Dict[str, Any]]:
    # Many LLMs prepend/append prose; try to extract JSON block.
    try:
        # direct parse
        return json.loads(txt)
    except Exception:
        pass
    # fallback: find first/last braces
    try:
        start = txt.find("{")
        end = txt.rfind("}")
        if start != -1 and end != -1:
            return json.loads(txt[start:end+1])
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

def ensure_env_keys(provider: str, openai_key: str, google_key: str, anthropic_key: str) -> bool:
    if provider == "OpenAI":
        if not openai_key:
            st.error("Please provide OpenAI API key.")
            return False
        os.environ["OPENAI_API_KEY"] = openai_key
    elif provider.startswith("Google"):
        if not google_key:
            st.error("Please provide Google (Gemini) API key.")
            return False
        os.environ["GOOGLE_API_KEY"] = google_key
    else:
        if not anthropic_key:
            st.error("Please provide Anthropic API key.")
            return False
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    return True

# ---- Run analysis -------------------------------------------------------------
results: List[ImageResult] = []

if images:
    if ensure_env_keys(provider, openai_key, google_key, anthropic_key):
        st.subheader("Results")
        grid = st.columns(2)
        for idx, up in enumerate(images):
            try:
                # Convert to consistent PNG bytes for base64
                im = Image.open(up).convert("RGB")
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                with st.spinner(f"Analyzing image {idx+1}/{len(images)} with {provider} • {model} ..."):
                    raw = call_provider(provider, model, b64, USER_PROMPT_TEMPLATE)

                parsed = parse_json_strict(raw)
                if not parsed:
                    st.error(f"Could not parse JSON for image {up.name}. Raw response shown below.")
                    st.code(raw)
                    continue

                ti = parsed.get("ti_rads","unknown")
                pr = parsed.get("malignancy_probability", None)
                if isinstance(pr, (int,float)):
                    pr = float(pr)
                else:
                    pr = None
                summ = parsed.get("summary","")

                # ---- UI card
                with grid[idx % 2]:
                    st.image(im, caption=up.name, use_column_width=True)
                    st.markdown(f"**Model TI-RADS**: {ti}")
                    if pr is not None:
                        st.markdown(f"**Malignancy probability**: `{pr:.3f}`")
                        st.progress(min(max(pr,0.0),1.0))
                    else:
                        st.markdown("**Malignancy probability**: `unknown`")
                    st.markdown(f"**Summary**: {summ}")
                    st.info(management_recommendation(pr, low_thr, high_thr))

                    with st.expander("Show structured JSON"):
                        st.json(parsed)
                    with st.expander("Show raw LLM text"):
                        st.code(raw)

                results.append(ImageResult(ti_rads=ti, prob=pr, summary=summ, raw_json=parsed, provider_raw=raw))

            except Exception as e:
                st.exception(e)

# ---- Optional: error checking of report vs image -----------------------------
if report_text and images:
    st.subheader("Report Error Check (optional)")
    if ensure_env_keys(provider, openai_key, google_key, anthropic_key):
        # Use the first image as reference for a quick demo
        ref_up = images[0]
        im = Image.open(ref_up).convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        templ = ERROR_CHECK_TEMPLATE + f"\n\nReport text:\n\"\"\"\n{report_text}\n\"\"\"\n"
        with st.spinner(f"Checking report against image {ref_up.name} ..."):
            raw_err = call_provider(provider, model, b64, templ)

        parsed_err = parse_json_strict(raw_err)
        if parsed_err:
            issues = parsed_err.get("issues", [])
            if issues:
                for i, it in enumerate(issues, 1):
                    st.warning(f"#{i} • **{it.get('type','issue')}** — {it.get('detail','')}\n\nSuggested fix: {it.get('suggested_fix','')}")
            else:
                st.success("No concrete issues detected from the image; still use human review.")
            st.caption(parsed_err.get("overall_comment",""))
            with st.expander("Show JSON"):
                st.json(parsed_err)
        else:
            st.error("Could not parse JSON from the error-checking response.")
            st.code(raw_err)

# ---- Chat mode (simple) ------------------------------------------------------
st.divider()
st.subheader("Follow-up Q&A with the Copilot")
st.caption("Ask about a specific image’s features, TI-RADS rationale, or guideline references.")
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

q = st.text_input("Your question")
ask_btn = st.button("Ask")
if ask_btn and q.strip():
    if not images:
        st.error("Upload at least one image first.")
    elif ensure_env_keys(provider, openai_key, google_key, anthropic_key):
        # Use the last analyzed image context (if any)
        if results:
            context_json = json.dumps(results[-1].raw_json, ensure_ascii=False)
            context_prompt = f"""You are continuing a discussion about the last analyzed image.
Here is the structured context you previously produced:
{context_json}

User question:
{q}

Answer concisely and reference the structured features when relevant."""
        else:
            context_prompt = f"User question (no prior image JSON available): {q}"

        # Reuse first image for visual grounding
        im = Image.open(images[-1]).convert("RGB")
        buf = io.BytesIO(); im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        with st.spinner("Generating answer..."):
            ans = call_provider(provider, model, b64, context_prompt)

        st.session_state.chat_log.append(("user", q))
        st.session_state.chat_log.append(("assistant", ans))

# Render chat
for role, text in st.session_state.chat_log[-10:]:
    if role == "user":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Copilot:** {text}")

st.divider()
st.caption("Based on the multimodal AIGC-CAD concept and threshold logic described in the paper on a Thyroid GPT copilot. (Research demo only.)")
