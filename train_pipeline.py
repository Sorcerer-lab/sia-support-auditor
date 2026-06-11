"""
train_pipeline.py — SIA Standalone Training Script
====================================================
Runs the full pipeline from raw CSV to trained model:
  1. Load + clean data
  2. Pseudo-label generation (Signal A: NLP + Signal B: Resolution-time)
  3. DeBERTa-v3-small + LoRA fine-tuning
  4. Evaluation + metric report
  5. Save model

Usage:
    python train_pipeline.py --data tickets.csv --output ./sia_model

Requirements:
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
"""

import argparse
import json
import os
import re
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_NAME   = 'microsoft/deberta-v3-small'
MAX_LEN      = 256
BATCH_SIZE   = 16
EPOCHS       = 3
LEARNING_RATE = 2e-4
NLP_WEIGHT   = 0.60
RT_WEIGHT    = 0.40

PRIORITY_SCORE = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
SCORE_PRIORITY = {v: k for k, v in PRIORITY_SCORE.items()}

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

# ── Signal A: NLP ─────────────────────────────────────────────────────────────
def keyword_score(text: str) -> dict:
    text_lower = text.lower()
    crit_hits = sum(1 for kw in CRITICAL_KEYWORDS if kw in text_lower)
    high_hits = sum(1 for kw in HIGH_KEYWORDS     if kw in text_lower)
    esc_hits  = sum(1 for p  in ESCALATION_PHRASES if re.search(p, text_lower))
    neg_hits  = sum(1 for p  in NEGATION_PATTERNS  if re.search(p, text_lower))
    score = crit_hits * 4.0 + esc_hits * 3.5 + neg_hits * 1.0 + high_hits * 2.0
    if score >= 6 or esc_hits >= 1:    severity = 4
    elif score >= 3 or crit_hits >= 1: severity = 3
    elif score >= 1:                   severity = 2
    else:                              severity = 1
    return {
        'nlp_severity':     severity,
        'crit_kw_count':    crit_hits,
        'esc_phrase_count': esc_hits,
        'negation_count':   neg_hits,
        'top_keywords':     [kw for kw in CRITICAL_KEYWORDS if kw in text_lower][:3],
    }

# ── Data loading + cleaning ───────────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    COLUMN_MAP = {
        'ticket_subject':        'subject',
        'ticket_description':    'description',
        'customer_email':        'customer_email',
        'priority_level':        'priority',
        'ticket_channel':        'channel',
        'resolution_time_hours': 'resolution_hours',
        'issue_category':        'ticket_type',
    }
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

    for col in ['subject', 'description']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)

    priority_map = {
        'low': 'Low', 'medium': 'Medium', 'high': 'High', 'critical': 'Critical',
    }
    df['priority_norm'] = df['priority'].astype(str).str.strip().str.lower().map(priority_map)
    df['priority_norm'] = df['priority_norm'].fillna('Medium')
    df['priority_score'] = df['priority_norm'].map(PRIORITY_SCORE).fillna(2)

    if 'ticket_id' not in df.columns:
        df['ticket_id'] = [f'TKT_{i:05d}' for i in range(len(df))]

    df['full_text'] = (df.get('subject', pd.Series([''] * len(df))).fillna('') + ' ' +
                       df.get('description', pd.Series([''] * len(df))).fillna('')).str.strip()

    df['resolution_hours'] = pd.to_numeric(df.get('resolution_hours', pd.Series([24.0] * len(df))), errors='coerce')
    df['resolution_hours'] = df['resolution_hours'].fillna(df['resolution_hours'].median())

    print(f"Loaded {len(df)} tickets. Priority distribution:")
    print(df['priority_norm'].value_counts().to_string())
    return df

# ── Pseudo-label generation ───────────────────────────────────────────────────
def generate_pseudo_labels(df: pd.DataFrame) -> pd.DataFrame:
    print("\nGenerating pseudo-labels...")

    # Signal A
    nlp_results = df['full_text'].apply(keyword_score)
    nlp_df = pd.DataFrame(nlp_results.tolist())
    df = pd.concat([df.reset_index(drop=True), nlp_df], axis=1)

    # Signal B
    z_scores = stats.zscore(df['resolution_hours'])
    df['rt_zscore'] = z_scores
    def zscore_to_severity(z):
        if z >= 1.5:    return 4
        elif z >= 0.5:  return 3
        elif z >= -0.5: return 2
        else:           return 1
    df['rt_severity'] = df['rt_zscore'].apply(zscore_to_severity)

    # Fusion
    df['fused_score']      = NLP_WEIGHT * df['nlp_severity'] + RT_WEIGHT * df['rt_severity']
    df['inferred_severity'] = df['fused_score'].round().clip(1, 4).astype(int)
    df['inferred_severity_label'] = df['inferred_severity'].map(SCORE_PRIORITY)
    df['severity_delta']   = df['inferred_severity'] - df['priority_score']

    def assign_mismatch(row):
        d = row['severity_delta']
        if d >= 2 or d <= -2: return 1
        if row['inferred_severity'] == 4 and row['priority_score'] <= 2: return 1
        if row['inferred_severity'] == 1 and row['priority_score'] >= 4: return 1
        return 0
    df['mismatch_label'] = df.apply(assign_mismatch, axis=1)
    df['mismatch_type'] = df.apply(
        lambda r: 'Consistent' if r['mismatch_label'] == 0
                  else ('Hidden Crisis' if r['severity_delta'] > 0 else 'False Alarm'), axis=1
    )

    # Ablation
    agreement = (df['nlp_severity'] == df['rt_severity']).mean()
    def signal_mismatch_rate(col):
        delta = df[col] - df['priority_score']
        m = ((delta >= 2) | (delta <= -2) |
             ((df[col] == 4) & (df['priority_score'] <= 2)) |
             ((df[col] == 1) & (df['priority_score'] >= 4)))
        return m.mean() * 100

    print("\n── ABLATION TABLE ──────────────────────────────────")
    print(f"  Signal A (NLP only):        {signal_mismatch_rate('nlp_severity'):.1f}% mismatch rate")
    print(f"  Signal B (ResTime only):    {signal_mismatch_rate('rt_severity'):.1f}% mismatch rate")
    print(f"  Fused (A+B {NLP_WEIGHT}/{RT_WEIGHT}):       {df['mismatch_label'].mean()*100:.1f}% mismatch rate")
    print(f"  Pairwise Signal Agreement:  {agreement:.3f}")
    print(f"  Mismatch type breakdown:")
    print(df['mismatch_type'].value_counts().to_string())
    print("────────────────────────────────────────────────────")

    return df

# ── Dataset ───────────────────────────────────────────────────────────────────
class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'token_type_ids': enc.get('token_type_ids',
                              torch.zeros(self.max_len, dtype=torch.long)).squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ── Weighted Trainer ──────────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    recalls = recall_score(labels, preds, average=None, zero_division=0)
    return {
        'accuracy':      round(accuracy_score(labels, preds), 4),
        'macro_f1':      round(f1_score(labels, preds, average='macro'), 4),
        'recall_class0': round(recalls[0], 4),
        'recall_class1': round(float(recalls[1]) if len(recalls) > 1 else 0.0, 4),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='SIA Training Pipeline')
    parser.add_argument('--data',    required=True,             help='Path to input CSV')
    parser.add_argument('--output',  default='./sia_model',     help='Output directory for model')
    parser.add_argument('--epochs',  type=int, default=EPOCHS,  help='Training epochs')
    parser.add_argument('--batch',   type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load + clean
    df = load_and_clean(args.data)

    # 2. Pseudo-labels
    df = generate_pseudo_labels(df)

    # 3. Build model input
    def build_input(row):
        rt_str = f"{row.get('resolution_hours', 0):.1f} hours"
        meta   = f"[CHANNEL: {row.get('channel','unknown')}] [RESOLUTION_TIME: {rt_str}] [TYPE: {row.get('ticket_type','unknown')}]"
        return f"{meta} {row.get('full_text', '')}".strip()[:512]

    df['model_input'] = df.apply(build_input, axis=1)

    # 4. Split
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42,
                                          stratify=df['mismatch_label'])
    val_df, test_df   = train_test_split(temp_df, test_size=0.50, random_state=42,
                                          stratify=temp_df['mismatch_label'])
    print(f"\nSplit — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # 5. Class weights
    labels_arr = train_df['mismatch_label'].values
    cw = compute_class_weight('balanced', classes=np.unique(labels_arr), y=labels_arr)
    cw_tensor = torch.tensor(cw, dtype=torch.float).to(device)
    print(f"Class weights: {cw}")

    # 6. Tokenizer + datasets
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = TicketDataset(train_df['model_input'].tolist(), train_df['mismatch_label'].tolist(), tokenizer, MAX_LEN)
    val_ds   = TicketDataset(val_df['model_input'].tolist(),   val_df['mismatch_label'].tolist(),   tokenizer, MAX_LEN)
    test_ds  = TicketDataset(test_df['model_input'].tolist(),  test_df['mismatch_label'].tolist(),  tokenizer, MAX_LEN)

    # 7. Model + LoRA
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
        target_modules=['query_proj', 'key_proj'], lora_dropout=0.1, bias='none',
    )
    model = get_peft_model(base_model, lora_config)
    model = model.float().to(device)
    model.print_trainable_parameters()

    # 8. Train
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=32,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=LEARNING_RATE,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        greater_is_better=True,
        fp16=False,
        logging_steps=50,
        save_total_limit=2,
        report_to='none',
        dataloader_pin_memory=False,
    )
    trainer = WeightedTrainer(
        class_weights=cw_tensor,
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    print("\nStarting training...")
    trainer.train()

    # 9. Evaluate
    print("\nEvaluating on test set...")
    preds_out   = trainer.predict(test_ds)
    test_preds  = np.argmax(preds_out.predictions, axis=1)
    test_labels = preds_out.label_ids

    acc      = accuracy_score(test_labels, test_preds)
    macro_f1 = f1_score(test_labels, test_preds, average='macro')
    recalls  = recall_score(test_labels, test_preds, average=None, zero_division=0)

    print("\n" + "="*55)
    print("FINAL TEST RESULTS")
    print("="*55)
    print(f"Binary Accuracy:         {acc*100:.2f}%  (≥83% required)")
    print(f"Macro F1:                {macro_f1:.4f}   (≥0.82 required)")
    print(f"Recall — Consistent(0):  {recalls[0]:.4f}   (≥0.78 required)")
    print(f"Recall — Mismatch(1):    {recalls[1]:.4f}   (≥0.78 required)")
    print("="*55)
    print(classification_report(test_labels, test_preds,
                                 target_names=['Consistent', 'Mismatch']))

    # Check thresholds
    passed = all([acc >= 0.83, macro_f1 >= 0.82, recalls[0] >= 0.78,
                  len(recalls) > 1 and recalls[1] >= 0.78])
    print(f"\nAll thresholds met: {'YES ✓' if passed else 'NO — check metrics above'}")

    # 10. Save
    trainer.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    metrics = {
        'accuracy': round(acc, 4), 'macro_f1': round(macro_f1, 4),
        'recall_class0': round(recalls[0], 4),
        'recall_class1': round(float(recalls[1]), 4) if len(recalls) > 1 else 0.0,
    }
    with open(os.path.join(args.output, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to: {args.output}")
    print(f"Metrics saved to: {args.output}/metrics.json")


if __name__ == '__main__':
    main()
