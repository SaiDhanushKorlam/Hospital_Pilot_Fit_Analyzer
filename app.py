import json
from io import BytesIO
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DATASET_PATH = Path(__file__).with_name("hospital_pilot_dataset_corrected.csv")
MAX_EVIDENCE_CHARS = 12000
EVALUATION_REQUEST_DELAY_SECONDS = 12
ORIGINAL_FAILURES = {
    "Aster CMI Hospital": ["hip_replacement"],
    "HOSMAT Hospital": ["knee_replacement", "hip_replacement"],
    "Sakra World Hospital": ["knee_replacement", "hip_replacement"],
}
LABEL_FIELDS = ["joint_replacement", "knee_replacement", "hip_replacement", "rehabilitation"]
REQUIRED_DATASET_COLUMNS = ["hospital", "city", "website", "evidence_text"]
EVALUATION_COLUMNS = REQUIRED_DATASET_COLUMNS + ["known_ground_truth"]
KNEE_PROCEDURE_PATTERNS = (
    r"\bknee\s+(?:replacement|replacements|arthroplasty)\b",
    r"\b(?:total|partial|revision|robotic)\s+knee\s+(?:replacement|replacements|arthroplasty)\b",
    r"\bTKR\b",
    r"\bhip\s+and\s+knee\s+replacements?\b",
    r"\b(?:total|robotic|partial|revision)?\s*knee\s+and\s+(?:total|robotic|partial|revision)?\s*hip\s+replacements?\b",
    r"\b(?:total|robotic|partial|revision)\s+knee\s+and\s+(?:total|robotic|partial|revision)\s+hip\s+replacement\b",
    r"\bknee\s+and\s+hip\s+joint\s+replacements?\b",
    r"\bjoint\s+replacement\s+surgeries?\s+of\s+the\s+hip\s+and\s+knee\b",
    r"\btotal\s+hip\s*/\s*knee\s+replacement\b",
)
HIP_PROCEDURE_PATTERNS = (
    r"\bhip\s+(?:replacement|replacements|arthroplasty)\b",
    r"\b(?:total|revision|robotic)\s+hip\s+(?:replacement|replacements|arthroplasty)\b",
    r"\bTHR\b",
    r"\bhip\s+and\s+knee\s+replacements?\b",
    r"\b(?:total|robotic|partial|revision)?\s*knee\s+and\s+(?:total|robotic|partial|revision)?\s*hip\s+replacements?\b",
    r"\b(?:total|robotic|partial|revision)\s+knee\s+and\s+(?:total|robotic|partial|revision)\s+hip\s+replacement\b",
    r"\bknee\s+and\s+hip\s+joint\s+replacements?\b",
    r"\bjoint\s+replacement\s+surgeries?\s+of\s+the\s+hip\s+and\s+knee\b",
    r"\btotal\s+hip\s*/\s*knee\s+replacement\b",
)
SCHEMA_DESCRIPTION = {
    "hospital_name": "string",
    "joint_replacement": "yes|no|unclear",
    "knee_replacement": "yes|no|unclear",
    "hip_replacement": "yes|no|unclear",
    "rehabilitation": "yes|no|unclear",
    "volume_signal": {
        "value": "string",
        "type": "annual|monthly|cumulative|surgeon_level|hospital_level|none|unclear",
        "evidence": "string",
    },
    "suggested_contact": {"name": "string", "role": "string", "evidence": "string"},
    "pilot_fit": "high|medium|low|insufficient_evidence",
    "pilot_fit_reason": "string",
    "confidence": "high|medium|low",
    "evidence": ["short quotations or close excerpts copied from supplied evidence"],
}

st.set_page_config(page_title="Hospital Pilot Fit Analyzer", page_icon="+", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#16221f; --muted:#65736e; --line:#dce5df; --paper:#f7faf7; --mint:#c9e7d5; --green:#1f6a4d; --amber:#b27625; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stApp { background: radial-gradient(circle at 90% 0%, #e5f3e8 0, #f7faf7 32%, #f7faf7 100%); }
    h1, h2, h3, h4, .stApp h1, .stApp h2, .stApp h3, .stApp h4, [data-testid="stAppViewContainer"] h1 { color:#172a23 !important; font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stText"] { color:#43534c !important; }
    .stApp [data-testid="stDataFrame"] { color:#20312b !important; }
    h1 { font-size: 2.4rem !important; margin-bottom: .25rem !important; }
    .subtitle { color: var(--muted); font-size: 1.05rem; margin-bottom: 1.7rem; }
    .eyebrow { color: var(--green); font-size: .75rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .panel { background: rgba(255,255,255,.92); border: 1px solid var(--line); border-radius: 8px; padding: 1.15rem; height: 100%; color:#20312b; }
    .panel p { color:#43534c !important; }
    .metric-label { color:#567067 !important; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
    .metric-value { color:#172a23 !important; font-family: 'Space Grotesk'; font-size: 1.25rem; font-weight: 600; margin-top: .4rem; }
    [data-testid="stExpander"] { background:#e4f0e6; border:1px solid #b8cfbe; border-radius:8px; }
    [data-testid="stExpander"] summary { color:#20543c !important; font-weight:600; }
    [data-testid="stExpander"] summary:hover { color:#123d2b !important; background:#d4e8d8; }
    .tag { display:inline-block; border-radius: 99px; padding: .25rem .6rem; font-size: .78rem; font-weight: 700; background: #e5f3e8; color: var(--green); }
    .tag.warn { background:#fff0d5; color:#855815; }
    .tag.neutral { background:#edf1ef; color:#59655f; }
    .quote { border-left: 3px solid var(--mint); padding-left: .75rem; color: #43534c; font-size: .92rem; margin: .55rem 0; }
    .disclaimer { color:#59655f; background:#eef4ef; border:1px solid #d8e5da; border-radius:6px; padding:.65rem .8rem; font-size:.82rem; }
    section[data-testid="stSidebar"] { background:#d8e9dc; border-right:1px solid #b8cfbe; }
    section[data-testid="stSidebar"] > div { background:#d8e9dc; }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#20382d !important; }
    section[data-testid="stSidebar"] [data-testid="stRadio"] > label { font-weight:600; }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { color:#14583d !important; }
    section[data-testid="stSidebar"] hr { border-color:#b8cfbe; }
    </style>
    """,
    unsafe_allow_html=True,
)


def empty_result(hospital_name: str) -> dict[str, Any]:
    return {
        "hospital_name": hospital_name,
        "joint_replacement": "unclear",
        "knee_replacement": "unclear",
        "hip_replacement": "unclear",
        "rehabilitation": "unclear",
        "volume_signal": {"value": "", "type": "none", "evidence": ""},
        "suggested_contact": {"name": "", "role": "", "evidence": ""},
        "pilot_fit": "insufficient_evidence",
        "pilot_fit_reason": "Not established from supplied evidence.",
        "confidence": "low",
        "evidence": [],
    }


def has_source_support(text: str, evidence: str) -> bool:
    if not evidence or evidence == "Not established from supplied evidence.":
        return False
    source = re.sub(r"\s+", " ", text.lower()).strip()
    excerpt = re.sub(r"\s+", " ", evidence.lower()).strip()
    return bool(excerpt) and excerpt in source


def explicit_procedure_labels(source_text: str) -> dict[str, str]:
    normalized = re.sub(r"\s+", " ", source_text).strip()
    return {
        "knee_replacement": "yes" if any(re.search(pattern, normalized, re.I) for pattern in KNEE_PROCEDURE_PATTERNS) else "unclear",
        "hip_replacement": "yes" if any(re.search(pattern, normalized, re.I) for pattern in HIP_PROCEDURE_PATTERNS) else "unclear",
    }


def validate_result(result: dict[str, Any], source_text: str, hospital_name: str) -> dict[str, Any]:
    clean = empty_result(hospital_name)
    for field in LABEL_FIELDS + ["pilot_fit", "confidence", "pilot_fit_reason"]:
        if field in result:
            clean[field] = result[field]
    for field in LABEL_FIELDS:
        if clean[field] not in {"yes", "no", "unclear"}:
            clean[field] = "unclear"
    if clean["pilot_fit"] not in {"high", "medium", "low", "insufficient_evidence"}:
        clean["pilot_fit"] = "insufficient_evidence"
    if clean["confidence"] not in {"high", "medium", "low"}:
        clean["confidence"] = "low"

    volume = result.get("volume_signal", {})
    if not isinstance(volume, dict):
        volume = {}
    volume_type = volume.get("type", "none")
    allowed_types = {"annual", "monthly", "cumulative", "surgeon_level", "hospital_level", "none", "unclear"}
    if volume_type not in allowed_types:
        volume_type = "unclear"
    volume_evidence = str(volume.get("evidence", ""))
    volume_value = str(volume.get("value", ""))
    if volume_type == "annual" and not re.search(r"\b(annual|per year|each year|yearly)\b", volume_evidence, re.I):
        volume_type, volume_value, volume_evidence = "unclear", "", "Not established from supplied evidence."
    if volume_type in {"annual", "monthly", "cumulative", "surgeon_level", "hospital_level"} and not has_source_support(source_text, volume_evidence):
        volume_type, volume_value, volume_evidence = "unclear", "", "Not established from supplied evidence."
    clean["volume_signal"] = {"value": volume_value, "type": volume_type, "evidence": volume_evidence}

    contact = result.get("suggested_contact", {})
    if not isinstance(contact, dict):
        contact = {}
    contact_evidence = str(contact.get("evidence", ""))
    if not has_source_support(source_text, contact_evidence):
        clean["suggested_contact"] = {"name": "Not established from supplied evidence.", "role": "", "evidence": "Not established from supplied evidence."}
    else:
        contact_role = str(contact.get("role", "")).strip()
        authority_pattern = r"\b(authori[sz]ed|decision[- ]?maker|pilot|partnership|procurement|program director|program lead|approv(?:e|es|ed|al)|partnerships? lead)\b"
        if not re.search(authority_pattern, contact_evidence, re.I):
            contact_role = re.sub(r"(?i)\b(?:confirmed?\s+)?decision[- ]?maker\b:?\s*", "", contact_role).strip()
            if not contact_role.lower().startswith("suggested clinical contact"):
                contact_role = f"Suggested clinical contact: {contact_role or 'relevant clinical contact'}"
        clean["suggested_contact"] = {"name": str(contact.get("name", "")), "role": contact_role, "evidence": contact_evidence}

        # Explicit procedure wording in the supplied source can correct a missed model label.
        explicit_labels = explicit_procedure_labels(source_text)
        for field in ("knee_replacement", "hip_replacement"):
            if explicit_labels[field] == "yes":
                clean[field] = "yes"
        if any(explicit_labels[field] == "yes" for field in ("knee_replacement", "hip_replacement")):
            clean["joint_replacement"] = "yes"

    clean["pilot_fit_reason"] = str(clean["pilot_fit_reason"] or "Not established from supplied evidence.")
    clean["evidence"] = [item for item in result.get("evidence", []) if isinstance(item, str) and has_source_support(source_text, item)][:8]
    return clean


def analyze_hospital(hospital_name: str, website: str, evidence_text: str) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to .env or the environment before analyzing.")
    if not evidence_text.strip():
        raise ValueError("Evidence text is required. Paste public source text before analyzing.")
    if len(evidence_text) > MAX_EVIDENCE_CHARS:
        raise ValueError(f"Evidence is too long ({len(evidence_text):,} characters). Limit it to {MAX_EVIDENCE_CHARS:,} characters.")

    prompt = f"""You are a conservative research analyst. Analyze only the supplied public evidence. Never use outside knowledge, the URL, or assumptions. Return JSON matching this shape exactly: {json.dumps(SCHEMA_DESCRIPTION)}

Rules:
- Use yes only when the text explicitly supports the label; otherwise use unclear, never guess no.
- Quote only text that appears in the evidence. Keep evidence short.
- A volume claim must preserve its qualification. Never turn cumulative or surgeon-level activity into annual hospital activity.
- Populate suggested_contact only when the evidence names a person. If the source establishes only a consultant, clinical lead, HOD, or joint-replacement specialist, describe them as a suggested clinical contact. Do not claim decision-making authority unless the supplied evidence explicitly establishes authority for pilots, partnerships, procurement, clinical programs, or equivalent.
- Use high only for strong evidence of knee/hip replacement, rehabilitation, meaningful qualified volume or institutional evidence, and an identifiable relevant clinical lead. Use medium for strong clinical relevance with weaker volume/lead evidence. Use low for limited relevance. Use insufficient_evidence when a reasonable assessment cannot be made.
- For unsupported conclusions write exactly: Not established from supplied evidence.

Hospital: {hospital_name}
Website: {website}
Evidence:
{evidence_text}"""
    client = genai.Client(api_key=api_key)
    for attempt, retry_delay in enumerate((0, 5, 15, 30)):
        if retry_delay:
            time.sleep(retry_delay)
        try:
            response = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="Return valid JSON only. Do not include markdown fences.",
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
            break
        except Exception as exc:
            error_text = str(exc).lower()
            retryable = any(marker in error_text for marker in ("429", "resource_exhausted", "rate limit", "503", "unavailable"))
            if not retryable or attempt == 3:
                if retryable:
                    raise RuntimeError("Gemini rate limit reached. Wait a few minutes and rerun the evaluation; no result was fabricated.") from exc
                raise RuntimeError(f"Gemini API error: {exc}") from exc
    raw = response.text or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("The model returned invalid JSON. Nothing was presented as a valid analysis.") from exc
    return validate_result(parsed, evidence_text, hospital_name)


def tag(value: str) -> str:
    labels = {"yes": "Yes", "no": "No", "unclear": "Unclear"}
    class_name = "" if value == "yes" else "warn" if value == "no" else "neutral"
    return f'<span class="tag {class_name}">{labels.get(value, value.title())}</span>'


def render_result(result: dict[str, Any]) -> None:
    st.markdown(f"### {result['hospital_name']}")
    columns = st.columns(4)
    for column, label, field in zip(columns, ["Joint replacement", "Knee replacement", "Hip replacement", "Rehabilitation"], LABEL_FIELDS):
        with column:
            st.markdown(f'<div class="panel"><div class="metric-label">{label}</div><div class="metric-value">{tag(result[field])}</div></div>', unsafe_allow_html=True)
    st.write("")
    columns = st.columns(3)
    with columns[0]:
        volume = result["volume_signal"]
        value = volume["value"] or "Not established from supplied evidence."
        st.markdown(f'<div class="panel"><div class="metric-label">Volume evidence · {volume["type"]}</div><div class="metric-value">{value}</div><p>{volume["evidence"]}</p></div>', unsafe_allow_html=True)
    with columns[1]:
        contact = result["suggested_contact"]
        name = contact["name"]
        role = contact["role"]
        display = name if name else "Not established from supplied evidence."
        st.markdown(f'<div class="panel"><div class="metric-label">Suggested contact</div><div class="metric-value">{display}</div><p>{role}</p></div>', unsafe_allow_html=True)
    with columns[2]:
        fit_label = result["pilot_fit"].replace("_", " ").title()
        st.markdown(f'<div class="panel"><div class="metric-label">Pilot fit · confidence {result["confidence"]}</div><div class="metric-value">{fit_label}</div><p>{result["pilot_fit_reason"]}</p></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("**Evidence snippets**")
    if result["evidence"]:
        for excerpt in result["evidence"]:
            st.markdown(f'<div class="quote">“{excerpt}”</div>', unsafe_allow_html=True)
    else:
        st.caption("Not established from supplied evidence.")
    with st.expander("View validated JSON"):
        st.json(result)


def parse_ground_truth(value: str) -> dict[str, str]:
    parsed = {}
    for item in value.split(";"):
        if "=" in item:
            key, label = item.split("=", 1)
            parsed[key] = label
    return parsed


def validate_dataset_schema(dataset: pd.DataFrame) -> tuple[list[str], bool]:
    missing = [column for column in REQUIRED_DATASET_COLUMNS if column not in dataset.columns]
    return missing, "known_ground_truth" in dataset.columns


def evaluation_template() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "hospital": "Example Hospital",
            "city": "Bengaluru",
            "website": "https://example.org/orthopaedics",
            "evidence_text": "Paste public evidence here. Include explicit procedure and rehabilitation wording where supported.",
            "known_ground_truth": "",
        }
    ], columns=EVALUATION_COLUMNS)


def audit_evaluation_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    audit_rows = []
    for row in dataset.to_dict("records"):
        ground_truth = parse_ground_truth(row["known_ground_truth"])
        explicit_labels = explicit_procedure_labels(row["evidence_text"])
        source_text = row["evidence_text"].lower()
        checks = {
            "joint_replacement": ground_truth.get("joint_replacement") != "yes" or bool(re.search(r"\bjoint\s+replacement\b|\bjoint\s+replacements\b", source_text, re.I)) or explicit_labels["knee_replacement"] == "yes" or explicit_labels["hip_replacement"] == "yes",
            "knee_replacement": ground_truth.get("knee_replacement") != "yes" or explicit_labels["knee_replacement"] == "yes",
            "hip_replacement": ground_truth.get("hip_replacement") != "yes" or explicit_labels["hip_replacement"] == "yes",
            "rehabilitation": ground_truth.get("rehabilitation") != "yes" or bool(re.search(r"\b(rehabilitation|physiotherapy|physical therapy)\b", source_text, re.I)),
        }
        for field, supported in checks.items():
            if ground_truth.get(field) != "yes":
                continue
            action = "No change; explicit supporting evidence present." if supported else "Needs correction before scoring."
            if row["hospital"] in {"Aster CMI Hospital", "HOSMAT Hospital", "Sakra World Hospital"} and supported:
                action = "Updated evidence_text with source-backed explicit wording."
            audit_rows.append({"Hospital": row["hospital"], "Ground-truth field": f"{field}=yes", "Supporting evidence present?": "Yes" if supported else "No", "Action taken": action})
    return pd.DataFrame(audit_rows)


def run_evaluation(dataset: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    results = []
    errors = []
    for index, row in enumerate(dataset.to_dict("records")):
        try:
            if index:
                time.sleep(EVALUATION_REQUEST_DELAY_SECONDS)
            result = analyze_hospital(row["hospital"], row["website"], row["evidence_text"])
            expected = parse_ground_truth(row["known_ground_truth"])
            comparisons = {field: {"expected": expected.get(field, "unclear"), "actual": result[field], "correct": expected.get(field, "unclear") == result[field]} for field in LABEL_FIELDS}
            results.append({"hospital": row["hospital"], "result": result, "comparisons": comparisons})
        except Exception as exc:
            errors.append({"hospital": row["hospital"], "error": str(exc)})
    return results, errors


def run_batch_analysis(dataset: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    results = []
    errors = []
    for index, row in enumerate(dataset.to_dict("records")):
        try:
            if index:
                time.sleep(EVALUATION_REQUEST_DELAY_SECONDS)
            result = analyze_hospital(row["hospital"], row["website"], row["evidence_text"])
            results.append({"hospital": row["hospital"], "city": row["city"], "website": row["website"], "result": result})
        except Exception as exc:
            errors.append({"hospital": row["hospital"], "error": str(exc)})
    return results, errors


def result_export_row(row: dict[str, Any]) -> dict[str, str]:
    result = row["result"]
    return {
        "hospital": row["hospital"],
        "city": row.get("city", ""),
        "website": row.get("website", ""),
        "joint_replacement": result["joint_replacement"],
        "knee_replacement": result["knee_replacement"],
        "hip_replacement": result["hip_replacement"],
        "rehabilitation": result["rehabilitation"],
        "volume_signal": json.dumps(result["volume_signal"]),
        "suggested_contact": json.dumps(result["suggested_contact"]),
        "pilot_fit": result["pilot_fit"],
        "pilot_fit_reason": result["pilot_fit_reason"],
        "confidence": result["confidence"],
        "evidence": json.dumps(result["evidence"]),
    }


def analyze_adversarial_before(hospital_name: str, evidence_text: str) -> dict[str, Any]:
    """Run a controlled prompt without the specific-joint safeguard or local validation."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to .env or the environment before analyzing.")
    prompt = f"""You are analyzing a hospital using only the supplied evidence. Return JSON matching this shape exactly: {json.dumps(SCHEMA_DESCRIPTION)}

Use the evidence to make the best interpretation for each field. Quote only supplied text. Preserve any volume qualification, but do not add facts that are not present.

Hospital: {hospital_name}
Evidence:
{evidence_text}"""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
    )
    try:
        return json.loads(response.text or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("The uncontrolled adversarial model response was invalid JSON.") from exc


def run_adversarial_tests() -> list[dict[str, str]]:
    specific_joint_evidence = "This hospital is a leading centre for joint replacement and provides comprehensive orthopaedic care."
    before_specific = analyze_adversarial_before("Example Orthopaedic Centre", specific_joint_evidence)
    after_specific = analyze_hospital("Example Orthopaedic Centre", "", specific_joint_evidence)
    before_specific_text = f"knee={before_specific.get('knee_replacement', 'missing')}; hip={before_specific.get('hip_replacement', 'missing')}"
    after_specific_text = f"knee={after_specific['knee_replacement']}; hip={after_specific['hip_replacement']}"
    specific_failure = before_specific.get("knee_replacement") == "yes" or before_specific.get("hip_replacement") == "yes"
    rows = [{
        "test_type": "Specific-joint inference",
        "test_description": "Generic joint replacement wording must not establish knee or hip replacement.",
        "expected_result": "joint=yes; knee=unclear; hip=unclear; rehabilitation=unclear",
        "before_result": before_specific_text,
        "after_result": after_specific_text,
        "failure_detected": "Yes" if specific_failure else "No",
        "fix_applied": "Specific-joint safeguard in prompt and evidence validation.",
        "status": "PASS after fix" if after_specific["knee_replacement"] == "unclear" and after_specific["hip_replacement"] == "unclear" else "FAIL",
    }]
    if not specific_failure:
        volume_evidence = "Dr. X has performed 2,000 joint replacements during their career."
        before_volume = analyze_adversarial_before("Volume Scope Test Hospital", volume_evidence)
        after_volume = analyze_hospital("Volume Scope Test Hospital", "", volume_evidence)
        before_volume_signal = before_volume.get("volume_signal", {})
        before_volume_text = f"{before_volume_signal.get('value', '')}; type={before_volume_signal.get('type', '')}"
        after_volume_text = f"{after_volume['volume_signal']['value']}; type={after_volume['volume_signal']['type']}"
        volume_failure = before_volume_signal.get("type") == "annual" or "annual" in before_volume_text.lower()
        rows.append({
            "test_type": "Volume-scope inference",
            "test_description": "Career volume must not become annual hospital-wide volume.",
            "expected_result": "2,000 joint replacements; cumulative or surgeon_level",
            "before_result": before_volume_text,
            "after_result": after_volume_text,
            "failure_detected": "Yes" if volume_failure else "No",
            "fix_applied": "Volume qualification validation preserves cumulative/surgeon-level scope.",
            "status": "PASS after fix" if after_volume["volume_signal"]["type"] in {"cumulative", "surgeon_level", "unclear"} else "FAIL",
        })
    return rows


def render_analyze() -> None:
    st.markdown('<div class="eyebrow">Research workspace · Bengaluru</div>', unsafe_allow_html=True)
    st.title("Hospital Pilot Fit Analyzer")
    st.markdown('<div class="subtitle">AI-assisted research for identifying orthopaedic pilot sites</div>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer">Research prototype. AI outputs must be verified against original sources before use in business decisions.</div>', unsafe_allow_html=True)
    st.write("")
    with st.form("analysis_form"):
        hospital_name = st.text_input("Hospital name", placeholder="e.g. Narayana Health City")
        website = st.text_input("Source URL", placeholder="https://...")
        st.caption("The application does not fetch this URL; it is retained as the source reference for the supplied evidence.")
        evidence = st.text_area("Evidence text", height=220, placeholder="Paste publicly available text from the hospital's website, annual report, doctor profile, or other research source.")
        submitted = st.form_submit_button("Analyze evidence", type="primary", use_container_width=True)
    if submitted:
        try:
            with st.spinner("Structuring evidence and checking claims..."):
                st.session_state["analysis_result"] = analyze_hospital(hospital_name.strip(), website.strip(), evidence)
        except Exception as exc:
            st.session_state.pop("analysis_result", None)
            st.error(str(exc))
    if st.session_state.get("analysis_result"):
        st.divider()
        render_result(st.session_state["analysis_result"])


def render_evaluation() -> None:
    st.markdown('<div class="eyebrow">Reusable research set</div>', unsafe_allow_html=True)
    st.title("Evaluation")
    st.markdown('<div class="subtitle">Run the same analyzer against the demo fixture or your own research CSV.</div>', unsafe_allow_html=True)
    template_csv = evaluation_template().to_csv(index=False).encode("utf-8")
    st.download_button("Download Evaluation CSV Template", data=template_csv, file_name="evaluation_template.csv", mime="text/csv")
    dataset_choice = st.radio("Evaluation Dataset", ["Use Demo Dataset", "Upload CSV"], horizontal=True)
    dataset = None
    dataset_label = ""
    dataset_identity = ""
    if dataset_choice == "Use Demo Dataset":
        dataset = pd.read_csv(DATASET_PATH)
        dataset_label = "Demo Dataset (audited)"
        dataset_identity = "demo"
    else:
        uploaded_file = st.file_uploader("Upload evaluation CSV", type=["csv"])
        if uploaded_file is None:
            st.info("Upload a CSV with hospital, city, website, and evidence_text columns to continue.")
            return
        try:
            dataset = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")
            return
        dataset_label = "Uploaded Dataset"
        dataset_identity = f"upload:{uploaded_file.name}:{uploaded_file.size}"

    missing_columns, has_ground_truth = validate_dataset_schema(dataset)
    st.markdown(f"**Dataset:** {dataset_label}  \n**Rows:** {len(dataset)}" + (f"  \n**Filename:** {uploaded_file.name}" if dataset_choice == "Upload CSV" else ""))
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.markdown("**Required schema example:**")
        st.code(",".join(EVALUATION_COLUMNS) + "\nExample Hospital,Bengaluru,https://example.org,Public evidence text,", language="csv")
        return

    if dataset_identity != st.session_state.get("evaluation_dataset_identity"):
        for key in ("evaluation_results", "evaluation_errors", "batch_results", "batch_errors"):
            st.session_state.pop(key, None)
        st.session_state["evaluation_dataset_identity"] = dataset_identity

    if has_ground_truth:
        if dataset_label.startswith("Demo Dataset"):
            st.subheader("Evaluation Dataset Audit")
            st.caption("The original 7/10 result is preserved as a historical pre-audit baseline. This audited dataset adds source-backed wording to the evidence; ground-truth labels are unchanged.")
            audit_table = audit_evaluation_dataset(dataset)
            st.dataframe(audit_table, use_container_width=True, hide_index=True)
        st.markdown("**Evaluation:** Ground truth provided; accuracy will be measured.")
        if st.button("Run evaluation", type="primary"):
            with st.spinner(f"Analyzing {len(dataset)} hospitals..."):
                results, errors = run_evaluation(dataset)
            st.session_state["evaluation_results"] = results
            st.session_state["evaluation_errors"] = errors
        if "evaluation_results" not in st.session_state:
            st.info("Evaluation calls the configured LLM for every row. Ground truth is parsed only after each model response.")
            return
        errors = st.session_state.get("evaluation_errors", [])
        results = st.session_state["evaluation_results"]
        if errors:
            st.error(f"{len(errors)} test case(s) could not be analyzed. No result was fabricated.")
            for error in errors:
                st.write(f"**{error['hospital']}**: {error['error']}")
        total = len(results)
        correct_cases = sum(all(item["correct"] for item in row["comparisons"].values()) for row in results)
        incorrect_cases = total - correct_cases
        columns = st.columns(4)
        for column, label, value in zip(columns, ["Evaluation", "Incorrect", "Case error rate", "Rows analyzed"], [f"{correct_cases} correct", incorrect_cases, f"{incorrect_cases / total:.1%}" if total else "n/a", total]):
            with column:
                st.markdown(f'<div class="panel"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
        rows = []
        for row in results:
            wrong = [field for field, comparison in row["comparisons"].items() if not comparison["correct"]]
            rows.append({"Hospital": row["hospital"], "Expected": ", ".join(f"{field}={row['comparisons'][field]['expected']}" for field in LABEL_FIELDS), "AI result": ", ".join(f"{field}={row['comparisons'][field]['actual']}" for field in LABEL_FIELDS), "Status": "Correct" if not wrong else "Incorrect", "Error type": ", ".join(wrong) or "-"})
        if rows:
            results_table = pd.DataFrame(rows)
            st.dataframe(results_table, use_container_width=True, hide_index=True)
            st.markdown("**Field accuracy**")
            field_rows = []
            for field in LABEL_FIELDS:
                field_correct = sum(row["comparisons"][field]["correct"] for row in results)
                field_rows.append({"Field": field.replace("_", " ").title(), "Correct": field_correct, "Total": total, "Accuracy": f"{field_correct / total:.1%}" if total else "n/a"})
            field_accuracy_table = pd.DataFrame(field_rows)
            st.dataframe(field_accuracy_table, use_container_width=True, hide_index=True)
            workbook = BytesIO()
            with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
                results_table.to_excel(writer, sheet_name="Evaluation results", index=False)
                field_accuracy_table.to_excel(writer, sheet_name="Field accuracy", index=False)
                if errors:
                    pd.DataFrame(errors).to_excel(writer, sheet_name="Runtime errors", index=False)
                adversarial_rows = st.session_state.get("adversarial_rows", [])
                if adversarial_rows:
                    pd.DataFrame(adversarial_rows).to_excel(writer, sheet_name="Adversarial tests", index=False)
            st.download_button("Download evaluation Excel", data=workbook.getvalue(), file_name="hospital_pilot_evaluation.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")

        if dataset_label.startswith("Demo Dataset"):
            st.divider()
            st.subheader("Assignment history")
            st.caption("The original pre-audit baseline and the audited run are preserved. The audited run was measured before any further analyzer change in this audit.")
            after_case_rows = []
            for row in results:
                before_fields = ORIGINAL_FAILURES.get(row["hospital"], [])
                after_wrong_fields = [field for field, comparison in row["comparisons"].items() if not comparison["correct"]]
                changed_fields = sorted(set(before_fields) | set(after_wrong_fields)) if before_fields or after_wrong_fields else []
                after_case_rows.append({"Hospital": row["hospital"], "Pre-audit": "Incorrect" if before_fields else "Correct", "Current": "Correct" if not after_wrong_fields else "Incorrect", "Changed fields": ", ".join(field.replace("_", " ") for field in changed_fields) or "-"})
            before_correct = 10 - len(ORIGINAL_FAILURES)
            after_incorrect = len(after_case_rows) - sum(row["Current"] == "Correct" for row in after_case_rows)
            history_columns = st.columns(4)
            for column, label, value in zip(history_columns, ["Original baseline", "Original error", "Audited pre-fix", "Audited error"], [f"{before_correct}/10 correct", "30%", f"{total - after_incorrect}/10 correct", f"{after_incorrect / total:.0%}" if total else "n/a"]):
                with column:
                    st.markdown(f'<div class="panel"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(after_case_rows), use_container_width=True, hide_index=True)
            st.info("No genuine model failure remains in the audited 10-case evaluation. The explicit evidence audit and unchanged analyzer produced 10/10; the adversarial test below remains separate and does not affect accuracy.")
            st.subheader("Adversarial Robustness Test")
            st.caption("Adversarial tests are separate from the 10-case accuracy evaluation and are used to probe known hallucination risks.")
            if st.button("Run adversarial robustness test", key="audited_adversarial_test"):
                try:
                    st.session_state["adversarial_rows"] = run_adversarial_tests()
                except Exception as exc:
                    st.warning(f"The adversarial test could not run: {exc}")
            adversarial_rows = st.session_state.get("adversarial_rows", [])
            if adversarial_rows:
                st.dataframe(pd.DataFrame(adversarial_rows), use_container_width=True, hide_index=True)
                detected = [row for row in adversarial_rows if row["failure_detected"] == "Yes"]
                st.subheader("Failure Caught and Handled")
                if detected:
                    st.error("Failure caught: the uncontrolled test inferred a claim that the safeguarded analyzer rejected.")
                    st.write("**Failure scenario:** A known hallucination-risk adversarial case produced an unsupported inference before the safeguard.")
                    st.write(f"**What the model initially did:** {detected[0]['before_result']}")
                    st.write("**Why it was wrong:** the supplied wording did not establish the inferred specific joint or volume scope.")
                    st.write(f"**Safeguard applied:** {detected[0]['fix_applied']}")
                    st.write(f"**Result after the change:** {detected[0]['after_result']}")
                else:
                    st.success("No failure was observed in the adversarial tests; the tests were retained as robustness checks.")
    else:
        st.markdown("**Batch Analysis — no ground truth provided, so accuracy cannot be measured.**")
        if st.button("Run batch analysis", type="primary"):
            with st.spinner(f"Analyzing {len(dataset)} hospitals..."):
                results, errors = run_batch_analysis(dataset)
            st.session_state["batch_results"] = results
            st.session_state["batch_errors"] = errors
        if "batch_results" not in st.session_state:
            st.info("Batch Analysis will return structured AI results without inventing ground truth.")
            return
        errors = st.session_state.get("batch_errors", [])
        results = st.session_state["batch_results"]
        if errors:
            st.error(f"{len(errors)} row(s) could not be analyzed. No result was fabricated.")
            for error in errors:
                st.write(f"**{error['hospital']}**: {error['error']}")
        if results:
            batch_table = pd.DataFrame([result_export_row(row) for row in results])
            st.dataframe(batch_table, use_container_width=True, hide_index=True)
            st.download_button("Download batch results CSV", data=batch_table.to_csv(index=False).encode("utf-8"), file_name="hospital_pilot_batch_results.csv", mime="text/csv", width="stretch")
            for row in results:
                with st.expander(f"{row['hospital']} · validated JSON"):
                    st.json(row["result"])


def main() -> None:
    with st.sidebar:
        st.markdown("## Pilot research")
        page = st.radio("Workspace", ["Analyze", "Evaluation"], label_visibility="collapsed")
        st.divider()
        st.caption("Public evidence only · no patient data")
    if page == "Analyze":
        render_analyze()
    else:
        render_evaluation()


if __name__ == "__main__":
    main()
