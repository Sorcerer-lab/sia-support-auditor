"""
SIA — Support Integrity Auditor
Streamlit Web App
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import os
import torch
from pathlib import Path

st.set_page_config(
    page_title="SIA — Support Integrity Auditor",
    page_icon="🔍",
    layout="wide",
)

MODEL_DIR     = os.getenv('MODEL_DIR', 'JxPar/sia-deberta-half')
DOSSIERS_PATH = os.getenv('DOSSIERS_PATH', './dossiers.json')
MAX_LEN       = 256

PRIORITY_SCORE = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
SCORE_PRIORITY = {v: k for k, v in PRIORITY_SCORE.items()}

CRITICAL_KEYWORDS = [
    'outage', 'down', 'not working', 'cannot access', 'complete failure',
    'production down', 'system crash', 'data loss', 'breach', 'hacked',
    'security incident', 'urgent', 'emergency', 'immediately', 'asap',
    'blocking', 'showstopper', 'critical bug', 'broken', 'unresponsive',
    'corrupted', 'deleted', 'locked out', 'cannot login', 'escalate',
]
HIGH_KEYWORDS = [
    'error', 'failing', 'failed', 'issue', 'problem', 'bug', 'wrong',
    'crash', 'slow', 'timeout', 'not loading', 'degraded', 'unstable',
]
ESCALATION_PHRASES = [
    r'need.{0,10}now', r'will.{0,10}cancel', r'legal.{0,10}action',
    r'team.{0,10}blocked', r'production.{0,5}issue',
]
NEGATION_PATTERNS = [
    r"can't", r"cannot", r"won't", r"doesn't", r"isn't", r"unable to",
    r"failed to", r"not working",
]


def keyword_score(text: str) -> dict:
    text_lower = text.lower()
    crit_hits  = sum(1 for kw in CRITICAL_KEYWORDS if kw in text_lower)
    high_hits  = sum(1 for kw in HIGH_KEYWORDS     if kw in text_lower)
    esc_hits   = sum(1 for p  in ESCALATION_PHRASES if re.search(p, text_lower))
    neg_hits   = sum(1 for p  in NEGATION_PATTERNS  if re.search(p, text_lower))
    score = crit_hits * 4.0 + esc_hits * 3.5 + neg_hits * 1.0 + high_hits * 2.0
    if score >= 6 or esc_hits >= 1:    severity = 4
    elif score >= 3 or crit_hits >= 1: severity = 3
    elif score >= 1:                   severity = 2
    else:                              severity = 1
    top_kw = [kw for kw in CRITICAL_KEYWORDS if kw in text_lower][:3]
    top_kw += [kw for kw in HIGH_KEYWORDS    if kw in text_lower][:2]
    return {
        'nlp_severity':     severity,
        'crit_kw_count':    crit_hits,
        'esc_phrase_count': esc_hits,
        'negation_count':   neg_hits,
        'top_keywords':     top_kw[:3],
    }


@st.cache_resource(show_spinner="Loading model... (first run may take 30s)")
def load_model():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, DebertaV2Config
        
        # Load config and fix pos_att_type
        config = DebertaV2Config.from_pretrained(MODEL_DIR)
        if isinstance(config.pos_att_type, list):
            config.pos_att_type = "|".join(config.pos_att_type)
        
        config.__dict__['id2label'] = {0: "Consistent", 1: "Mismatch"}
        config.__dict__['label2id'] = {"Consistent": 0, "Mismatch": 1}
        config.num_labels = 2
        st.write("Config list fields:", {k: type(v).__name__ for k, v in config.__dict__.items() 
                                  if isinstance(v, list)})


        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            config=config,
            ignore_mismatched_sizes=True,
        )
        model = model.float()
        model.eval()
        return tokenizer, model, True
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None, False


@st.cache_data
def load_dossiers():
    if Path(DOSSIERS_PATH).exists():
        with open(DOSSIERS_PATH) as f:
            return json.load(f)
    return []


def predict_ticket(text, subject, channel, resolution_hours, priority_norm,
                   ticket_type, ticket_id, tokenizer, model, use_model):
    full_text  = f"{subject} {text}".strip()
    rt_str     = f"{float(resolution_hours):.1f} hours" if resolution_hours else "unknown"
    input_text = f"[CHANNEL: {channel}] [RESOLUTION_TIME: {rt_str}] [TYPE: {ticket_type}] {full_text}"

    nlp      = keyword_score(full_text)
    inferred = nlp['nlp_severity']
    assigned = PRIORITY_SCORE.get(priority_norm, 2)
    delta    = inferred - assigned
    mtype    = 'Consistent'
    if abs(delta) >= 2 or (inferred == 4 and assigned <= 2) or (inferred == 1 and assigned >= 4):
        mtype = 'Hidden Crisis' if delta > 0 else 'False Alarm'

    if use_model and tokenizer and model:
        model=model.float()
        device = next(model.parameters()).device
        enc = tokenizer(input_text, max_length=MAX_LEN, padding='max_length',
                        truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask']
            )
        probs      = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred_label = int(np.argmax(probs))
        confidence = float(probs[pred_label])
    else:
        pred_label = 1 if mtype != 'Consistent' else 0
        confidence = 0.75 if pred_label == 1 else 0.80

    evidence = []
    for kw in nlp['top_keywords']:
        if kw in full_text.lower():
            evidence.append({'signal': 'keyword', 'value': kw, 'weight': 'high'})
    if resolution_hours:
        evidence.append({
            'signal': 'resolution_time',
            'value': f"{resolution_hours} hours",
            'interpretation': ('Above average — suggests complex issue'
                               if float(resolution_hours) > 48 else
                               'Below average — relatively quick resolution'),
        })

    dossier = {
        'ticket_id':          ticket_id,
        'assigned_priority':  priority_norm,
        'inferred_severity':  SCORE_PRIORITY.get(inferred, 'Medium'),
        'mismatch_type':      mtype if pred_label == 1 else 'Consistent',
        'severity_delta':     delta,
        'feature_evidence':   evidence,
        'constraint_analysis': (
            f"The ticket contains {nlp['crit_kw_count']} critical urgency signals "
            f"and {nlp['esc_phrase_count']} escalation phrases, suggesting severity "
            f"{SCORE_PRIORITY.get(inferred, 'Medium')}. Assigned priority ({priority_norm}) "
            f"{'under-represents' if delta > 0 else 'over-represents'} urgency by {abs(delta)} level(s)."
        ) if pred_label == 1 else "Priority aligns with ticket content.",
        'confidence': round(confidence, 4),
    }

    return pred_label, mtype, confidence, dossier


def main():
    all_dossiers = load_dossiers()

    st.title("🔍 SIA — Support Integrity Auditor")
    st.caption("Detects priority mismatches in CRM support tickets.")

    tab1, tab2, tab3 = st.tabs(["🎫 Single Ticket", "📦 Batch CSV", "📊 Dashboard"])

    with tab1:
        st.subheader("Analyze a Single Ticket")
        col1, col2 = st.columns([2, 1])

        with col1:
            ticket_id   = st.text_input("Ticket ID", value="TKT_00001")
            subject     = st.text_input("Subject", placeholder="e.g. Cannot login to production")
            description = st.text_area("Description", height=150,
                                       placeholder="Full ticket description...")
        with col2:
            priority    = st.selectbox("Assigned Priority", ['Low', 'Medium', 'High', 'Critical'])
            channel     = st.selectbox("Channel", ['Email', 'Chat', 'Phone', 'Social Media'])
            ticket_type = st.selectbox("Ticket Type", ['Technical', 'Billing', 'Account', 'General'])
            res_time    = st.number_input("Resolution Time (hours)", min_value=0.0,
                                          max_value=9999.0, value=24.0, step=0.5)

        if st.button("🔍 Analyze Ticket", type="primary", use_container_width=True):
            if not description.strip():
                st.error("Please enter a ticket description.")
            else:
                with st.spinner("Analyzing... (loading model on first run, may take 30s)"):
                    tokenizer, model, use_model = load_model()
                    pred, mtype, conf, dossier = predict_ticket(
                        description, subject, channel, res_time,
                        priority, ticket_type, ticket_id,
                        tokenizer, model, use_model
                    )

                if use_model:
                    st.caption("✅ Running with fine-tuned DeBERTa model")
                else:
                    st.caption("ℹ️ Running in rule-based NLP mode")

                if pred == 0:
                    st.success(f"✅ **Consistent** — Priority aligns with ticket content. Confidence: {conf:.1%}")
                elif mtype == 'Hidden Crisis':
                    st.error(f"🚨 **Hidden Crisis** — Ticket is under-prioritized! Confidence: {conf:.1%}")
                else:
                    st.warning(f"⚠️ **False Alarm** — Ticket is over-prioritized. Confidence: {conf:.1%}")

                if pred == 1:
                    st.divider()
                    st.subheader("Evidence Dossier")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Assigned Priority", dossier['assigned_priority'])
                    col_b.metric("Inferred Severity", dossier['inferred_severity'])
                    col_c.metric("Severity Delta",    f"{dossier['severity_delta']:+d}")
                    st.json(dossier)
                    st.download_button(
                        "⬇️ Download Dossier JSON",
                        data=json.dumps(dossier, indent=2),
                        file_name=f"dossier_{ticket_id}.json",
                        mime="application/json",
                    )

    with tab2:
        st.subheader("Batch Analysis — Upload CSV")
        st.info("CSV must have columns: `ticket_id`, `description`, `priority`, "
                "`channel` (optional), `resolution_hours` (optional)", icon="ℹ️")

        uploaded = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            st.write(f"Loaded {len(df)} tickets.")
            st.dataframe(df.head(5), use_container_width=True)

            if st.button("▶️ Run Batch Analysis", type="primary"):
                tokenizer, model, use_model = load_model()
                results  = []
                progress = st.progress(0)

                for i, row in df.iterrows():
                    pred, mtype, conf, dossier = predict_ticket(
                        text=str(row.get('description', '')),
                        subject=str(row.get('subject', '')),
                        channel=str(row.get('channel', 'unknown')),
                        resolution_hours=row.get('resolution_hours', 24),
                        priority_norm=str(row.get('priority', 'Medium')),
                        ticket_type=str(row.get('ticket_type', 'General')),
                        ticket_id=str(row.get('ticket_id', i)),
                        tokenizer=tokenizer, model=model, use_model=use_model
                    )
                    results.append({
                        'ticket_id':         row.get('ticket_id', i),
                        'assigned_priority': row.get('priority', 'Medium'),
                        'inferred_severity': dossier['inferred_severity'],
                        'mismatch_label':    pred,
                        'mismatch_type':     mtype if pred == 1 else 'Consistent',
                        'severity_delta':    dossier['severity_delta'],
                        'confidence':        conf,
                    })
                    progress.progress((i + 1) / len(df))

                results_df = pd.DataFrame(results)
                st.success(f"Analysis complete! {results_df['mismatch_label'].sum()} mismatches detected.")
                st.dataframe(results_df, use_container_width=True)
                st.download_button(
                    "⬇️ Download Results CSV",
                    data=results_df.to_csv(index=False),
                    file_name="sia_batch_results.csv",
                    mime="text/csv"
                )

    with tab3:
        st.subheader("Priority Mismatch Dashboard")

        if not all_dossiers:
            st.info("No dossier data found. Run batch analysis or load dossiers.json.")
            return

        ddf = pd.DataFrame(all_dossiers)

        total = len(ddf)
        n_mis = (ddf.get('mismatch_type', pd.Series()) != 'Consistent').sum()
        n_hc  = (ddf.get('mismatch_type', pd.Series()) == 'Hidden Crisis').sum()
        n_fa  = (ddf.get('mismatch_type', pd.Series()) == 'False Alarm').sum()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Tickets", total)
        k2.metric("Mismatches", n_mis, f"{n_mis/max(total,1)*100:.0f}%")
        k3.metric("Hidden Crises 🚨", n_hc)
        k4.metric("False Alarms ⚠️", n_fa)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Mismatch Type Distribution**")
            if 'mismatch_type' in ddf.columns:
                st.bar_chart(ddf['mismatch_type'].value_counts())
        with col2:
            st.markdown("**Severity Delta Distribution**")
            if 'severity_delta' in ddf.columns:
                st.bar_chart(ddf['severity_delta'].value_counts().sort_index())

        st.divider()
        st.markdown("**Severity Delta Heatmap — Assigned Priority vs Inferred Severity**")
        if 'assigned_priority' in ddf.columns and 'inferred_severity' in ddf.columns:
            try:
                heat = pd.crosstab(ddf['assigned_priority'], ddf['inferred_severity'])
                st.dataframe(heat.style.background_gradient(cmap='RdYlGn_r'),
                             use_container_width=True)
            except Exception:
                st.write("Heatmap unavailable — insufficient data variety.")

        st.divider()
        st.markdown("**Top Contributing Signal Keywords**")
        all_kw = []
        for d in all_dossiers:
            for ev in d.get('feature_evidence', []):
                if ev.get('signal') == 'keyword':
                    val = ev.get('value', '')
                    if val and val.lower() != 'nan' and len(val) > 2:
                        all_kw.append(val)
        if all_kw:
            st.bar_chart(pd.Series(all_kw).value_counts().head(15))
        else:
            st.write("No keyword evidence data available.")


if __name__ == '__main__':
    main()