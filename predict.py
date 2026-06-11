"""
predict.py — SIA Inference Script
Usage: python predict.py --input tickets.csv --output predictions.csv

Accepts a CSV of tickets, outputs predictions + dossiers.
"""

import argparse
import json
import re
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch


# ── NLP utility (same as app.py — no model required for NLP-only mode) ───────
CRITICAL_KEYWORDS = [
    'outage', 'down', 'not working', 'cannot access', 'complete failure',
    'production down', 'system crash', 'data loss', 'breach', 'hacked',
    'urgent', 'emergency', 'immediately', 'asap', 'blocking', 'showstopper',
    'critical bug', 'broken', 'unresponsive', 'corrupted', 'deleted',
    'locked out', 'cannot login', 'escalate',
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
    r"can't", r"cannot", r"won't", r"doesn't", r"isn't",
    r"unable to", r"failed to", r"not working",
]
PRIORITY_SCORE = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
SCORE_PRIORITY = {v: k for k, v in PRIORITY_SCORE.items()}


def keyword_score(text: str) -> dict:
    text_lower = text.lower()
    crit_hits = sum(1 for kw in CRITICAL_KEYWORDS if kw in text_lower)
    high_hits = sum(1 for kw in HIGH_KEYWORDS     if kw in text_lower)
    esc_hits  = sum(1 for p  in ESCALATION_PHRASES if re.search(p, text_lower))
    neg_hits  = sum(1 for p  in NEGATION_PATTERNS  if re.search(p, text_lower))
    score = crit_hits * 4.0 + esc_hits * 3.5 + neg_hits * 1.0 + high_hits * 2.0
    if score >= 6 or esc_hits >= 1: severity = 4
    elif score >= 3 or crit_hits >= 1: severity = 3
    elif score >= 1: severity = 2
    else: severity = 1
    return {
        'nlp_severity': severity,
        'crit_kw_count': crit_hits,
        'esc_phrase_count': esc_hits,
        'top_keywords': [kw for kw in CRITICAL_KEYWORDS if kw in text_lower][:3],
    }


def load_model(model_dir: str):
    """Load fine-tuned DeBERTa + LoRA. Returns (tokenizer, model, device) or None."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel
        print(f"Loading model from {model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        base = AutoModelForSequenceClassification.from_pretrained(
            'microsoft/deberta-v3-small', num_labels=2, ignore_mismatched_sizes=True
        )
        model = PeftModel.from_pretrained(base, model_dir)
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"Model loaded. Device: {device}")
        return tokenizer, model, device
    except Exception as e:
        print(f"WARNING: Could not load model ({e}). Using NLP-only mode.", file=sys.stderr)
        return None, None, None


def predict_row(row: dict, tokenizer, model, device, max_len=256) -> dict:
    """Run inference on one ticket row dict."""
    text     = str(row.get('description', row.get('full_text', ''))).strip()
    subject  = str(row.get('subject', '')).strip()
    channel  = str(row.get('channel', 'unknown'))
    rt       = row.get('resolution_hours', 24)
    tt       = str(row.get('ticket_type', 'General'))
    pnorm    = str(row.get('priority', row.get('priority_norm', 'Medium')))
    tid      = str(row.get('ticket_id', 'N/A'))

    full_text  = f"{subject} {text}".strip()
    rt_str     = f"{float(rt):.1f} hours"
    input_text = f"[CHANNEL: {channel}] [RESOLUTION_TIME: {rt_str}] [TYPE: {tt}] {full_text}"

    nlp      = keyword_score(full_text)
    inferred = nlp['nlp_severity']
    assigned = PRIORITY_SCORE.get(pnorm, 2)
    delta    = inferred - assigned

    if model and tokenizer and device:
        enc = tokenizer(input_text, max_length=max_len, padding='max_length',
                        truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs      = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label = int(np.argmax(probs))
        confidence = float(probs[pred_label])
    else:
        pred_label = 1 if (abs(delta) >= 2 or
                           (inferred == 4 and assigned <= 2) or
                           (inferred == 1 and assigned >= 4)) else 0
        confidence = 0.80

    mtype = 'Consistent'
    if pred_label == 1:
        mtype = 'Hidden Crisis' if delta > 0 else 'False Alarm'

    evidence = []
    for kw in nlp['top_keywords']:
        if kw in full_text.lower():
            evidence.append({'signal': 'keyword', 'value': kw, 'weight': 'high'})
    evidence.append({
        'signal': 'resolution_time',
        'value': rt_str,
        'interpretation': ('Above average — suggests complex issue'
                           if float(rt) > 48 else 'Within normal range'),
    })

    dossier = None
    if pred_label == 1:
        dossier = {
            'ticket_id':         tid,
            'assigned_priority': pnorm,
            'inferred_severity': SCORE_PRIORITY.get(inferred, 'Medium'),
            'mismatch_type':     mtype,
            'severity_delta':    delta,
            'feature_evidence':  evidence,
            'constraint_analysis': (
                f"Ticket contains {nlp['crit_kw_count']} critical urgency keywords "
                f"and {nlp['esc_phrase_count']} escalation phrases. "
                f"Inferred severity ({SCORE_PRIORITY.get(inferred, 'Medium')}) "
                f"{'exceeds' if delta > 0 else 'falls below'} the assigned "
                f"priority ({pnorm}) by {abs(delta)} level(s)."
            ),
            'confidence': round(confidence, 4),
        }

    return {
        'ticket_id':         tid,
        'assigned_priority': pnorm,
        'inferred_severity': SCORE_PRIORITY.get(inferred, 'Medium'),
        'mismatch_label':    pred_label,
        'mismatch_type':     mtype,
        'severity_delta':    delta,
        'confidence':        round(confidence, 4),
        'dossier':           dossier,
    }


def main():
    parser = argparse.ArgumentParser(description='SIA — Support Integrity Auditor inference')
    parser.add_argument('--input',     required=True,  help='Path to input CSV')
    parser.add_argument('--output',    default='sia_predictions.csv', help='Output CSV path')
    parser.add_argument('--dossiers',  default='sia_dossiers.json',   help='Output dossiers JSON path')
    parser.add_argument('--model_dir', default='./sia_deberta_lora',  help='Path to model directory')
    args = parser.parse_args()

    # Load data
    print(f"Loading input: {args.input}")
    df = pd.read_csv(args.input)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    print(f"Loaded {len(df)} tickets.")

    # Load model
    tokenizer, model, device = load_model(args.model_dir)

    # Run predictions
    results  = []
    dossiers = []
    for i, row in df.iterrows():
        result = predict_row(row.to_dict(), tokenizer, model, device)
        if result['dossier']:
            dossiers.append(result.pop('dossier'))
        else:
            result.pop('dossier')
        results.append(result)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(df)}...")

    # Save outputs
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(f"\nPredictions saved: {args.output}")
    print(f"  Total tickets:  {len(results_df)}")
    print(f"  Mismatches:     {results_df['mismatch_label'].sum()}")
    print(f"  Mismatch rate:  {results_df['mismatch_label'].mean()*100:.1f}%")

    with open(args.dossiers, 'w') as f:
        json.dump(dossiers, f, indent=2)
    print(f"Dossiers saved:  {args.dossiers} ({len(dossiers)} dossiers)")

    # Print summary
    print("\n── Mismatch breakdown ──────────────────────────")
    print(results_df['mismatch_type'].value_counts().to_string())


if __name__ == '__main__':
    main()
