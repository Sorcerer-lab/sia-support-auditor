# SIA — Support Integrity Auditor

> Semantics-driven, evidence-grounded auditor that detects **Priority Mismatches** in CRM support tickets — where the human-assigned priority conflicts with the ticket's true objective severity.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sia-support-auditor.streamlit.app)

---

## Problem

In enterprise-scale CRM ecosystems, manual ticket triage is riddled with agent fatigue bias, keyword anchoring, and customer favoritism. This causes:
- **Hidden Crises**: Critical tickets labeled Low/Medium → SLA breaches
- **False Alarms**: Trivial tickets inflated to Critical → wasted resources

SIA detects both with no pre-annotated labels, bootstrapping its own supervision signal from raw ticket data.

---

## Architecture

```
Raw Tickets
     │
     ▼
┌─────────────────────────────────────────────┐
│  STAGE 1: Pseudo-Label Generation           │
│  Signal A: Rule-based NLP (keywords, esc.)  │──→ Fused Score (60/40)
│  Signal B: Resolution-time regression       │──→ Binary Mismatch Label
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│  STAGE 2: Classifier Training               │
│  DeBERTa-v3-small + LoRA adapters (r=8)     │
│  Input: text + channel + resolution_time    │
│  Loss: Weighted CrossEntropy (imbalance)    │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│  STAGE 3: Evidence Dossier Generation       │
│  Rule-grounded structured JSON per ticket   │
│  Zero hallucination — all evidence traced   │
└─────────────────────────────────────────────┘
```

---

## Pseudo-Label Fusion Strategy

| Signal | Method | Individual Mismatch Rate | Weight |
|--------|--------|--------------------------|--------|
| Signal A | Rule-based NLP (keyword density, escalation phrases, negation detection) | 20.1% | 60% |
| Signal B | Resolution-time z-score regression | 50.8% | 40% |
| **Fused** | Weighted average → round to 1–4 | **22.2%** | — |

**Rationale**: NLP signal gets higher weight (60%) because it directly reads urgency semantics from ticket text — the primary signal for severity. The resolution-time signal is a useful but noisy proxy; it has a high standalone mismatch rate (50.8%) because many tickets take long to resolve for non-severity reasons (e.g. waiting on customer response). Fusing at 60/40 keeps the NLP anchor while letting RT contribute for tickets where text is vague.

**Mismatch label**: Assigned when |inferred_severity − assigned_priority| ≥ 2, or Critical inferred as Low/Medium, or Low inferred as Critical.

### Ablation Table

| Configuration | Mismatch Rate | Pairwise Signal Agreement | Notes |
|---------------|--------------|--------------------------|-------|
| Signal A only (NLP) | 20.1% | — | High precision, misses context-free tickets |
| Signal B only (RT) | 50.8% | — | Noisy — many false positives |
| **A + B Fused (60/40)** | **22.2%** | **0.274** | Best balance — NLP anchors, RT supplements |

---

## Model

- **Base**: `microsoft/deberta-v3-small`
- **Adapter**: LoRA (r=8, α=16, target: `query_proj` + `key_proj`)
- **Trainable params**: ~1.3% of total
- **Input features**: Ticket text + `[CHANNEL: x] [RESOLUTION_TIME: x hrs] [TYPE: x]` prefix
- **Imbalance handling**: Weighted CrossEntropyLoss
- **Training**: 3 epochs, lr=2e-4, fp16=False (DeBERTa-v3 LoRA compatibility), batch=16

---

## Evaluation Results

### Classification Metrics (Test Set — 3000 tickets)

| Metric | Threshold | Result | Status |
|--------|-----------|--------|--------|
| Binary Accuracy | ≥ 83% | **92.73%** | ✅ PASS |
| Macro F1 | ≥ 0.82 | **0.8976** | ✅ PASS |
| Recall — Consistent (0) | ≥ 0.78 | **0.9418** | ✅ PASS |
| Recall — Mismatch (1) | ≥ 0.78 | **0.8767** | ✅ PASS |

### Per-Class Report

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Consistent | 0.96 | 0.94 | 0.95 | 2335 |
| Mismatch | 0.81 | 0.88 | 0.84 | 665 |
| **Macro avg** | **0.89** | **0.91** | **0.90** | 3000 |

### Adversarial Robustness (Bonus)

| System | Score | Bonus |
|--------|-------|-------|
| Keyword baseline | 4/10 | — |
| **SIA model** | **8/10** | **✅ +10% bonus earned** |

The SIA model correctly handled 8/10 adversarially crafted tickets, including:
- Tickets with urgency keywords in subject but benign body text (False Alarm detection)
- Tickets with calm/polite framing hiding critical operational failures (Hidden Crisis detection)
- Healthcare context tickets with no standard urgency vocabulary

---

## Dossier Schema

For every flagged ticket:

```json
{
  "ticket_id": "TKT_00042",
  "assigned_priority": "Low",
  "inferred_severity": "Critical",
  "mismatch_type": "Hidden Crisis",
  "severity_delta": 3,
  "feature_evidence": [
    { "signal": "keyword", "value": "production down", "weight": "high" },
    { "signal": "resolution_time", "value": "96.0 hours", "interpretation": "Above average — suggests complex issue" }
  ],
  "constraint_analysis": "The ticket contains 2 critical urgency keywords and 1 escalation phrase. Inferred severity (Critical) exceeds the assigned priority (Low) by 3 levels.",
  "confidence": 0.9241
}
```

**Hard rule**: Every `feature_evidence` value is traceable to an actual field in the input ticket. Zero hallucinated evidence.

---

## Dataset

**Customer Support Tickets — CRM Dataset**  
[kaggle.com/datasets/ajverse/customersupport-tickets-crm-dataset](https://kaggle.com/datasets/ajverse/customersupport-tickets-crm-dataset)

---

## Repository Structure

```
sia/
├── SIA_Phase1_2_PseudoLabeling.ipynb   # EDA + pseudo-label generation
├── SIA_Phase3_Training.ipynb            # DeBERTa fine-tuning
├── SIA_Phase4_Dossier.ipynb             # Dossier generation
├── app.py                               # Streamlit web app
├── train_pipeline.py                    # Standalone training script
├── predict.py                           # CLI inference script
├── adversarial_test.py                  # Adversarial robustness evaluation
├── requirements.txt
└── README.md
```

---

## Running Locally

```bash
pip install -r requirements.txt

# Full training pipeline
python train_pipeline.py --data tickets.csv --output ./sia_model

# CLI inference on a CSV
python predict.py --input tickets.csv --output predictions.csv --model_dir ./sia_deberta_lora

# Adversarial robustness test
python adversarial_test.py --model_dir ./sia_deberta_lora

# Streamlit app
streamlit run app.py
```

---

## Links

- 🌐 **Demo App**: [your-app-url.streamlit.app]
- 📁 **Models folder**: [your-drive-link]
- 🎥 **Demo video**: [your-video-link]