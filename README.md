# SIA — Support Integrity Auditor

> Semantics-driven, evidence-grounded auditor that detects **Priority Mismatches** in CRM support tickets — where the human-assigned priority conflicts with the ticket's true objective severity.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## Problem

In enterprise CRM ecosystems, manual ticket triage is riddled with agent fatigue bias, keyword anchoring, and customer favoritism. This causes:
- **Hidden Crises**: Critical tickets labeled Low/Medium → SLA breaches
- **False Alarms**: Trivial tickets inflated to Critical → wasted resources

SIA detects both, with no pre-annotated labels, bootstrapping its own supervision signal from raw ticket data.

---

## Architecture

```
Raw Tickets
     │
     ▼
┌─────────────────────────────────────────────┐
│  STAGE 1: Pseudo-Label Generation           │
│  Signal A: Rule-based NLP (keywords, esc.)  │──→ Fused Score
│  Signal B: Resolution-time regression       │──→ Binary Mismatch Label
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│  STAGE 2: Classifier Training               │
│  DeBERTa-v3-small + LoRA adapters           │
│  Input: text + channel + resolution_time    │
│  Loss: Weighted CrossEntropy (imbalance)    │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│  STAGE 3: Evidence Dossier Generation       │
│  Mistral-7B-Instruct via HF Inference API   │
│  Structured JSON per flagged ticket         │
└─────────────────────────────────────────────┘
```

---

## Pseudo-Label Fusion Strategy

| Signal | Method | Individual Mismatch Rate | Weight |
|--------|--------|--------------------------|--------|
| Signal A | Rule-based NLP (keyword density, escalation phrases, negation detection) | *see ablation* | 60% |
| Signal B | Resolution-time z-score regression | *see ablation* | 40% |
| **Fused** | Weighted average → round to 1–4 | *see ablation* | — |

**Rationale**: NLP signal gets higher weight because it directly reads urgency from text content. RT signal is noisy (short tickets have high variance in RT for unrelated reasons), but adds complementary operational-severity signal for tickets where text is vague.

**Mismatch label**: Assigned when |inferred − assigned| ≥ 2, or Critical inferred as Low, or Low inferred as Critical.

### Ablation Table

| Configuration | Signal Agreement | Mismatch Rate | Notes |
|---------------|-----------------|---------------|-------|
| Signal A only (NLP) | — | *run Phase 2 to fill* | Rule-based, high precision |
| Signal B only (RT) | — | *run Phase 2 to fill* | Noisy, misses short tickets |
| **A + B Fused (60/40)** | *run Phase 2 to fill* | *run Phase 2 to fill* | Best balance |

> Fill this table with actual numbers from `SIA_Phase1_2_PseudoLabeling.ipynb` output.

---

## Model

- **Base**: `microsoft/deberta-v3-small`
- **Adapter**: LoRA (r=8, α=16, target: query_proj + key_proj)
- **Trainable params**: ~1.3% of total (efficient for free T4 Colab)
- **Input features**: Ticket text + `[CHANNEL: x] [RESOLUTION_TIME: x hrs] [TYPE: x]` prefix
- **Imbalance handling**: Weighted CrossEntropyLoss (weights from Phase 2)
- **Training**: 3 epochs, lr=2e-4, fp16, batch size 16

---

## Evaluation Metrics

| Metric | Threshold | Result |
|--------|-----------|--------|
| Binary Accuracy | ≥ 83% | *fill after training* |
| Macro F1 | ≥ 0.82 | *fill after training* |
| Recall — Consistent | ≥ 0.78 | *fill after training* |
| Recall — Mismatch | ≥ 0.78 | *fill after training* |
| Signal Agreement (A vs B) | — | *fill from Phase 2* |

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

**Hard rule**: Every `feature_evidence` value is traceable to an actual field in the input ticket. No hallucinated evidence.

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
├── SIA_Phase4_Dossier.ipynb             # Dossier generation (HF API)
├── app.py                               # Streamlit web app
├── predict.py                           # CLI inference script
├── requirements.txt
└── README.md
```

---

## Running Locally

```bash
pip install -r requirements.txt

# CLI inference on a CSV
python predict.py --input tickets.csv --output predictions.csv --model_dir ./sia_deberta_lora

# Streamlit app
streamlit run app.py
```

---

## Links

- 🌐 **Demo App**: [your-app-url.streamlit.app]  
- 📁 **Models folder**: [your-drive-link]  
- 🎥 **Demo video**: [your-video-link]
