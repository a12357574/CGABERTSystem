import os
import torch
import pandas as pd
import numpy as np
from transformers import (
    BertForMaskedLM,
    RobertaForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
import evaluate
from pathlib import Path
import time
import json
import psutil
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F

# Install dependencies
import subprocess
subprocess.check_call(['pip', 'install', 'transformers', 'datasets', 'torch', 'evaluate', 'huggingface_hub', 'peft'])

# Setup
input_dir = Path('/kaggle/input') if Path('/kaggle/input').exists() else Path('.')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
torch.cuda.empty_cache()

# Load dataset
dataset_path = os.environ.get('DATASET_PATH', '')
if dataset_path:
    try:
        if dataset_path.endswith('.csv'):
            dataset = load_dataset('csv', data_files=dataset_path)
        else:
            dataset = load_dataset(dataset_path, split='train')
        dataset = dataset.filter(lambda x: x['text'].strip() != '' and len(x['text'].split()) > 5)
    except Exception as e:
        print(f'Failed to load dataset {dataset_path}: {e}')
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        dataset = dataset.filter(lambda x: x['text'].strip() != '' and len(x['text'].split()) > 5)
else:
    try:
        dataset_files = list(input_dir.glob('*.csv'))
        if not dataset_files:
            raise FileNotFoundError('No CSV file found.')
        dataset = load_dataset('csv', data_files=str(dataset_files[0]))
    except Exception:
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        dataset = dataset.filter(lambda x: x['text'].strip() != '' and len(x['text'].split()) > 5)

# Log dataset details
num_samples = len(dataset)
print(f"Dataset size: {num_samples} samples")
sample_texts = dataset[:5]['text']
print(f"Sample texts: {sample_texts}")
language = "Tagalog" if dataset_path.endswith('.csv') else "English (wikitext)"
print(f"Dataset language: {language}")

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
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

tokenized_train_base = train_dataset.map(lambda x: tokenize(x, base_tokenizer), batched=True, remove_columns=['text'])
tokenized_val_base = val_dataset.map(lambda x: tokenize(x, base_tokenizer), batched=True, remove_columns=['text'])
tokenized_train_improved = train_dataset.map(lambda x: tokenize(x, improved_tokenizer), batched=True, remove_columns=['text'])
tokenized_val_improved = val_dataset.map(lambda x: tokenize(x, improved_tokenizer), batched=True, remove_columns=['text'])

# Load models
base_model = BertForMaskedLM.from_pretrained('GKLMIP/bert-tagalog-base-uncased').to(device)
improved_model = RobertaForMaskedLM.from_pretrained(improved_model_path).to(device)

# Apply LoRA for fine-tuning (Q1: Database optimization & fine-tuning)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none"
)
improved_model = get_peft_model(improved_model, lora_config)
improved_model.print_trainable_parameters()

# Fine-tuning
fine_tuned_model_path = './fine_tuned_model'
if os.path.exists(fine_tuned_model_path):
    print(f'Loading fine-tuned model from {fine_tuned_model_path}')
    if classification == 'small':
        from transformers import DistilBertForMaskedLM
        improved_model = DistilBertForMaskedLM.from_pretrained(fine_tuned_model_path).to(device)
    else:
        improved_model = RobertaForMaskedLM.from_pretrained(fine_tuned_model_path).to(device)
else:
    print('Fine-tuning model with LoRA...')
    training_args = TrainingArguments(
        output_dir='./output',
        num_train_epochs=1,
        per_device_train_batch_size=4,
        eval_strategy='no',
        logging_dir='./logs',
        report_to='none',
        fp16=True  # Mixed-precision
    )
    if classification == 'big':
        trainer = Trainer(
            model=improved_model,
            args=training_args,
            train_dataset=tokenized_train_improved,
            data_collator=DataCollatorForLanguageModeling(tokenizer=improved_tokenizer, mlm=True)
        )
        trainer.train()
        print(f'Saving fine-tuned model to {fine_tuned_model_path}')
        improved_model.save_pretrained(fine_tuned_model_path)
    else:
        print('Dataset is small, skipping fine-tuning')

# Evaluation function with focal loss (Q2: Token prediction accuracy)
def evaluate_mlm(model, tokenizer, dataset, device, gamma=2.0):
    model.eval()
    accuracy_metric = evaluate.load('accuracy')
    f1_metric = evaluate.load('f1')
    predictions, labels = [], []
    perplexity_scores = []
    latencies = []
    memory_usages = []
    
    for i in range(len(dataset)):
        start_time = time.time()
        input_ids = torch.tensor(dataset[i]['input_ids']).unsqueeze(0).to(device)
        attention_mask = torch.tensor(dataset[i]['attention_mask']).unsqueeze(0).to(device)
        mask_token_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        
        if len(mask_token_index) == 0:
            valid_indices = (input_ids[0] != tokenizer.pad_token_id) & \
                            (input_ids[0] != tokenizer.cls_token_id) & \
                            (input_ids[0] != tokenizer.sep_token_id)
            valid_indices = valid_indices.nonzero(as_tuple=True)[0]
            if len(valid_indices) == 0:
                continue
            mask_idx = valid_indices[torch.randint(0, len(valid_indices), (1,)).item()]
            original_token = input_ids[0, mask_idx].clone()
            input_ids[0, mask_idx] = tokenizer.mask_token_id
            mask_token_index = torch.tensor([mask_idx]).to(device)
            original_token = [original_token.item()]
        else:
            original_token = input_ids[0, mask_token_index].cpu().numpy()
            if original_token.ndim == 0:
                original_token = [original_token.item()]
            else:
                original_token = original_token.tolist()
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_token_id = torch.argmax(logits[0, mask_token_index], dim=-1)
            
            # Perplexity
            probs = F.softmax(logits[0, mask_token_index], dim=-1)
            log_probs = F.log_softmax(logits[0, mask_token_index], dim=-1)
            perplexity = torch.exp(-log_probs[torch.arange(len(original_token)), original_token].mean())
            perplexity_scores.append(perplexity.item())
            
            # Focal loss for logging
            ce_loss = F.cross_entropy(logits[0, mask_token_index], torch.tensor(original_token).to(device), reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** gamma * ce_loss
            print(f"Sample {i}: Focal loss = {focal_loss.mean().item():.4f}")
        
        predictions.extend(predicted_token_id.cpu().numpy().tolist())
        labels.extend(original_token)
        
        # Latency
        latency = time.time() - start_time
        latencies.append(latency)
        
        # Memory usage
        memory = psutil.Process().memory_info().rss / 1024**2  # MB
        memory_usages.append(memory)
    
    if not predictions:
        print('No valid predictions; returning 0 metrics')
        return {'accuracy': 0.0, 'f1': 0.0, 'perplexity': float('inf'), 'latency': 0.0, 'memory': 0.0}
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)['accuracy']
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')['f1']
    avg_perplexity = np.mean(perplexity_scores)
    avg_latency = np.mean(latencies)
    avg_memory = np.mean(memory_usages)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'perplexity': avg_perplexity,
        'latency': avg_latency,
        'memory': avg_memory
    }

# Run evaluation with quantization (Q3: Efficiency)
from torch.quantization import quantize_dynamic
base_model_quantized = quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
improved_model_quantized = quantize_dynamic(improved_model, {torch.nn.Linear}, dtype=torch.qint8)

start_time = time.time()
base_metrics = evaluate_mlm(base_model_quantized, base_tokenizer, tokenized_val_base, device)
improved_metrics = evaluate_mlm(improved_model_quantized, improved_tokenizer, tokenized_val_improved, device)
eval_time = time.time() - start_time

print(f"BaseBERT metrics: {base_metrics}")
print(f"Improved model metrics: {improved_metrics}")
print(f"Evaluation time: {eval_time:.2f} seconds")

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
results.to_csv('results.csv', index=False)
print(results)

# Interpret results
threshold = 0.60
is_good = improved_metrics['accuracy'] >= threshold
interpretation = {
    'status': 'Good' if is_good else 'Needs Improvement',
    'reason': f'Improved model (CGABERT) accuracy ({improved_metrics["accuracy"]:.4f}) {"exceeds" if is_good else "is below"} threshold ({threshold}) for effective text autocomplete.',
    'dataset': {'size': num_samples, 'language': language, 'samples': sample_texts[:3]},
    'metrics': improved_metrics
}
with open('interpretation.json', 'w') as f:
    json.dump(interpretation, f, indent=4)