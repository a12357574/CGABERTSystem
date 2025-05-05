# Install required packages
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("Installing required packages...")
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "transformers", "datasets", "evaluate", "peft", "psutil", "pandas", "numpy", "huggingface_hub", "-q"])
print("Packages installed successfully.")

# Test to confirm script execution
print(f'Python version: {sys.version}')
print('Script execution started')

# Import libraries
import torch
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import time
import json
import psutil
import torch.nn.functional as F
from transformers import (
    BertForMaskedLM,
    RobertaForMaskedLM,
    AutoTokenizer,
    DistilBertForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import evaluate
from peft import LoraConfig, get_peft_model
from torch.quantization import quantize_dynamic

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Starting script execution")

# Setup
input_dir = Path('/kaggle/input') if Path('/kaggle/input').exists() else Path('.')
output_dir = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('.')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
quant_device = torch.device('cpu')  # Quantized models run on CPU
logger.info(f'Using device: {device} for main models, {quant_device} for quantized models')
torch.cuda.empty_cache()

# Baseline memory
baseline_memory = psutil.Process().memory_info().rss / 1024**2  # MB
logger.info(f"Baseline memory: {baseline_memory:.2f} MB")

# Load dataset
dataset_path = os.environ.get('DATASET_PATH', '')
logger.info(f"Dataset path from environment: {dataset_path}")
if dataset_path:
    try:
        if dataset_path.endswith('.csv'):
            dataset = load_dataset('csv', data_files=dataset_path)
            logger.info(f"Loaded CSV dataset from {dataset_path}")
        else:
            dataset = load_dataset(dataset_path, split='train')
            logger.info(f"Loaded dataset from {dataset_path}")
        dataset = dataset.filter(lambda x: x['text'].strip() != '' and len(x['text'].split()) > 5)
    except Exception as e:
        logger.error(f'Failed to load dataset {dataset_path}: {e}')
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        dataset = dataset.filter(lambda x: x['text'].strip() != '' and len(x['text'].split()) > 5)
        logger.info("Fell back to wikitext dataset")
else:
    try:
        dataset_files = list(input_dir.glob('*.csv'))
        if dataset_files:
            dataset = load_dataset('csv', data_files=str(dataset_files[0]))
            logger.info(f"Loaded dataset from {dataset_files[0]}")
        else:
            raise FileNotFoundError('No CSV file found; falling back to wikitext.')
        dataset = dataset.filter(lambda x: x['text'].strip() != '' and len(x['text'].split()) > 5)
    except Exception as e:
        logger.error(f"Dataset loading error: {e}")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        dataset = dataset.filter(lambda x: x['text'].strip() != '' and len(x['text'].split()) > 5)
        logger.info("Fell back to wikitext dataset")

# Log dataset details
num_samples = len(dataset)
logger.info(f"Dataset size: {num_samples} samples")
sample_texts = dataset[:5]['text'] if 'text' in dataset.column_names else ["No text column found"]
logger.info(f"Sample texts: {sample_texts}")
language = "Tagalog" if dataset_path and dataset_path.endswith('.csv') else "English (wikitext)"
logger.info(f"Dataset language: {language}")

# Classify dataset
classification = 'small' if num_samples < 512 else 'big'
data_type = 'low-resource NLP' if num_samples < 1000 else 'standard NLP'
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split['train']
val_dataset = train_test_split['test']

# Tokenization
base_tokenizer = AutoTokenizer.from_pretrained('GKLMIP/bert-tagalog-base-uncased')
improved_model_path = 'distilbert-base-uncased' if classification == 'small' else 'jcblaise/roberta-tagalog-base'
improved_tokenizer = AutoTokenizer.from_pretrained(improved_model_path, do_lower_case=False)

def tokenize(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=64)

# Filter non-Tagalog tokens if using wikitext
def filter_non_tagalog(examples, tokenizer):
    if language != "English (wikitext)":
        return True
    tokens = tokenizer(examples['text'], truncation=True, max_length=64).input_ids
    vocab = set(tokenizer.get_vocab().keys())
    decoded = tokenizer.convert_ids_to_tokens(tokens)
    return all(
        token in vocab and 
        not token.startswith('##') and 
        all(ord(c) < 128 for c in tokenizer.decode([t]) if t not in [tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id])
        for t, token in zip(tokens, decoded)
        if token not in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]
    )

if language == "English (wikitext)":
    train_dataset = train_dataset.filter(lambda x: filter_non_tagalog(x, base_tokenizer))
    val_dataset = val_dataset.filter(lambda x: filter_non_tagalog(x, base_tokenizer))
    logger.info(f"Filtered dataset size: train={len(train_dataset)}, val={len(val_dataset)}")

tokenized_train_base = train_dataset.map(lambda x: tokenize(x, base_tokenizer), batched=True, remove_columns=['text'])
tokenized_val_base = val_dataset.map(lambda x: tokenize(x, base_tokenizer), batched=True, remove_columns=['text'])
tokenized_train_improved = train_dataset.map(lambda x: tokenize(x, improved_tokenizer), batched=True, remove_columns=['text'])
tokenized_val_improved = val_dataset.map(lambda x: tokenize(x, improved_tokenizer), batched=True, remove_columns=['text'])

# Validate tokens
def validate_tokens(dataset, tokenizer):
    vocab = set(tokenizer.get_vocab().keys())
    invalid_count = 0
    for example in dataset:
        tokens = tokenizer.convert_ids_to_tokens(example['input_ids'])
        if any(token not in vocab for token in tokens if token not in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]):
            invalid_count += 1
    logger.info(f"Invalid tokens found: {invalid_count}/{len(dataset)} samples")
    return invalid_count == 0

logger.info("Validating base tokenizer tokens...")
base_tokens_valid = validate_tokens(tokenized_val_base, base_tokenizer)
logger.info("Validating improved tokenizer tokens...")
improved_tokens_valid = validate_tokens(tokenized_val_improved, improved_tokenizer)

# Load models
base_model = BertForMaskedLM.from_pretrained('GKLMIP/bert-tagalog-base-uncased').to(device)
improved_model = RobertaForMaskedLM.from_pretrained(improved_model_path).to(device)

# Evaluation function with focal loss and batched inference
def evaluate_mlm(model, tokenizer, dataset, device, gamma=2.0, batch_size=16):
    model.eval()
    accuracy_metric = evaluate.load('accuracy')
    f1_metric = evaluate.load('f1')
    predictions, labels = [], []
    perplexity_scores = []
    latencies = []
    memory_usages = []
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        start_time = time.time()
        input_ids = torch.tensor(batch['input_ids']).to(device)
        attention_mask = torch.tensor(batch['attention_mask']).to(device)
        logger.info(f"Batch {i//batch_size}: input_ids shape = {input_ids.shape}, attention_mask shape = {attention_mask.shape}")
        
        # Ensure at least one mask token per sequence
        mask_indices = []
        original_tokens = []
        for j in range(input_ids.size(0)):
            valid_indices = (input_ids[j] != tokenizer.pad_token_id) & (input_ids[j] != tokenizer.cls_token_id) & (input_ids[j] != tokenizer.sep_token_id)
            valid_idx = valid_indices.nonzero(as_tuple=True)[0]
            if len(valid_idx) > 0:
                mask_idx = valid_idx[torch.randint(0, len(valid_idx), (1,)).item()]
                original_token = input_ids[j, mask_idx].clone()
                input_ids[j, mask_idx] = tokenizer.mask_token_id
                mask_indices.append(mask_idx)
                original_tokens.append(original_token.item())
            else:
                mask_indices.append(torch.tensor([0]))  # Default to first position if no valid indices
                original_tokens.append(input_ids[j, 0].item())
        
        if not original_tokens:
            logger.info(f"Batch {i//batch_size}: No valid tokens, skipping")
            continue
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            for j in range(len(original_tokens)):
                predicted_token_id = torch.argmax(logits[j, mask_indices[j]], dim=-1)
                probs = F.softmax(logits[j, mask_indices[j]], dim=-1)
                log_probs = F.log_softmax(logits[j, mask_indices[j]], dim=-1)
                log_probs = torch.clamp(log_probs, min=-100, max=0)
                try:
                    neg_log_prob = -log_probs[0, original_tokens[j]]
                    if neg_log_prob > 100:
                        logger.info(f"Sample {i+j}: Extreme neg_log_prob {neg_log_prob.item():.4f}, skipping")
                        continue
                    perplexity = torch.exp(torch.clamp(neg_log_prob, max=100))
                    if not torch.isfinite(perplexity):
                        logger.info(f"Sample {i+j}: Non-finite perplexity, skipping")
                        continue
                    perplexity_scores.append(perplexity.item())
                    logger.info(f"Sample {i+j}: Perplexity = {perplexity.item():.4f}, Neg log prob = {neg_log_prob.item():.4f}")
                except Exception as e:
                    logger.error(f"Sample {i+j}: Perplexity error: {e}")
                    continue
                ce_loss = F.cross_entropy(logits[j, mask_indices[j]].unsqueeze(0), torch.tensor([original_tokens[j]]).to(device), reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = (1 - pt) ** gamma * ce_loss
                logger.info(f"Sample {i+j}: Focal loss = {focal_loss.item():.4f}")
                predictions.append(predicted_token_id.item())
                labels.append(original_tokens[j])
        
        latency = (time.time() - start_time) / len(original_tokens)
        latencies.extend([latency] * len(original_tokens))
        memory = (psutil.Process().memory_info().rss / 1024**2) - baseline_memory
        memory_usages.extend([memory] * len(original_tokens))
        del input_ids, attention_mask, logits, outputs
        torch.cuda.empty_cache()
    
    if not predictions:
        logger.warning('No valid predictions; returning 0 metrics')
        return {'accuracy': 0.0, 'f1': 0.0, 'perplexity': float('inf'), 'latency': 0.0, 'memory': 0.0}
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)['accuracy']
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')['f1']
    avg_perplexity = np.mean(perplexity_scores) if perplexity_scores else float('inf')
    avg_latency = np.mean(latencies)
    avg_memory = np.mean(memory_usages)
    return {'accuracy': accuracy, 'f1': f1, 'perplexity': avg_perplexity, 'latency': avg_latency, 'memory': avg_memory}

# Run evaluation with quantization
logger.info("Applying dynamic quantization (qint8) for CPU backend")
base_model_quantized = quantize_dynamic(base_model.to('cpu'), {torch.nn.Linear}, dtype=torch.qint8).to(quant_device)
improved_model_quantized = quantize_dynamic(improved_model.to('cpu'), {torch.nn.Linear}, dtype=torch.qint8).to(quant_device)

# Log model size
def get_model_size(model):
    torch.save(model.state_dict(), "temp.pt")
    size = os.path.getsize("temp.pt") / 1024**2
    os.remove("temp.pt")
    return size

logger.info(f"BaseBERT quantized size: {get_model_size(base_model_quantized):.2f} MB")
logger.info(f"Improved quantized size: {get_model_size(improved_model_quantized):.2f} MB")

start_time = time.time()
base_metrics = evaluate_mlm(base_model_quantized, base_tokenizer, tokenized_val_base, quant_device, batch_size=16)
improved_metrics = evaluate_mlm(improved_model_quantized, improved_tokenizer, tokenized_val_improved, quant_device, batch_size=16)
eval_time = time.time() - start_time

logger.info(f"BaseBERT metrics: {base_metrics}")
logger.info(f"Improved model metrics: {improved_metrics}")
logger.info(f"Evaluation time: {eval_time:.2f} seconds")

# Save results
results = pd.DataFrame({
    'Model': ['BaseBERT', 'Improved'],
    'Accuracy': [base_metrics['accuracy'], improved_metrics['accuracy']],
    'F1': [base_metrics['f1'], improved_metrics['f1']],
    'Perplexity': [base_metrics['perplexity'], improved_metrics['perplexity']],
    'LatencySeconds': [base_metrics['latency'], improved_metrics['latency']],
    'MemoryMB': [base_metrics['memory'], improved_metrics['memory']],
    'EvalTimeSeconds': [eval_time / 2, eval_time / 2]
})
results.to_csv(os.path.join(output_dir, 'output_results.csv'), index=False)
logger.info(f"Results saved: {results}")

# Interpret results
threshold = 0.60
is_good = improved_metrics['accuracy'] >= threshold
interpretation = {
    'status': 'Good' if is_good else 'Needs Improvement',
    'reason': f'Improved model accuracy ({improved_metrics["accuracy"]:.4f}) {"exceeds" if is_good else "is below"} threshold ({threshold})',
    'dataset': {'size': num_samples, 'language': language, 'samples': sample_texts[:3]},
    'metrics': improved_metrics
}
with open(os.path.join(output_dir, 'output_interpretation.json'), 'w') as f:
    json.dump(interpretation, f, indent=4)
logger.info("Interpretation saved")

logger.info("Script executed successfully")
print(f"Execution completed. Check output_results.csv and output_interpretation.json in {output_dir}")