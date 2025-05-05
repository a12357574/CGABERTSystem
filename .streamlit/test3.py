import streamlit as st
import pandas as pd
import json
import logging
import os
import time
import tempfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define training script content with enhanced logging
TRAINING_SCRIPT_CONTENT = """
# Install dependencies
import subprocess
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting dependency installation...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "datasets", "torch", "evaluate", "huggingface_hub", "scikit-learn"])
    logger.info("Dependencies installed successfully")
except Exception as e:
    logger.error(f"Failed to install dependencies: {e}")
    raise

# Import libraries
import os
import torch
import pandas as pd
import numpy as np
from transformers import (
    BertForMaskedLM,
    RobertaForMaskedLM,
    DistilBertForMaskedLM,
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
import logging
from sklearn.metrics import f1_score
import shutil

# Setup
os.makedirs("/kaggle/working", exist_ok=True)
input_dir = Path('/kaggle/input') if Path('/kaggle/input').exists() else Path('.')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
torch.cuda.empty_cache()

# Clear previous fine-tuned model to force fresh training
fine_tuned_model_path = Path("/kaggle/working/fine_tuned_model")
if fine_tuned_model_path.exists():
    shutil.rmtree(fine_tuned_model_path)
    logger.info(f"Cleared existing fine-tuned model at {fine_tuned_model_path}")

# Log environment details
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Input directory contents: {list(input_dir.glob('*'))}")

# Load dataset
dataset_path = os.environ.get('DATASET_PATH', '')
logger.info(f"Dataset path from environment: {dataset_path}")
if dataset_path:
    try:
        logger.info(f"Attempting to load dataset from {dataset_path}")
        if dataset_path.endswith('.csv'):
            dataset = load_dataset('csv', data_files=dataset_path)
        else:
            dataset = load_dataset(dataset_path, split='train')
        logger.info("Dataset loaded successfully")
        dataset = dataset.filter(lambda x: x['text'] is not None and x['text'].strip() != '' and len(x['text'].split()) > 5)
        logger.info("Dataset filtered successfully")
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_path}: {e}")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        dataset = dataset.filter(lambda x: x['text'] is not None and x['text'].strip() != '' and len(x['text'].split()) > 5)
        logger.info("Fallback to WikiText dataset")
else:
    try:
        dataset_files = list(input_dir.glob('*.csv'))
        logger.info(f"Found dataset files: {dataset_files}")
        if not dataset_files:
            raise FileNotFoundError("No CSV file found.")
        dataset = load_dataset('csv', data_files=str(dataset_files[0]))
        logger.info("Dataset loaded successfully from input directory")
        dataset = dataset.filter(lambda x: x['text'] is not None and x['text'].strip() != '' and len(x['text'].split()) > 5)
        logger.info("Dataset filtered successfully")
    except Exception as e:
        logger.error(f"Failed to load dataset from input directory: {e}")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        dataset = dataset.filter(lambda x: x['text'] is not None and x['text'].strip() != '' and len(x['text'].split()) > 5)
        logger.info("Fallback to WikiText dataset")

# Debug dataset structure
logger.info(f"Dataset structure: {dataset.column_names}")
logger.info(f"Sample data: {dataset[:2]}")

# Classify dataset
num_samples = len(dataset)
classification = "small" if num_samples < 512 else "big"
data_type = "standard NLP" if num_samples > 1000 else "low-resource NLP"
logger.info(f"Dataset size: {num_samples}, classification: {classification}, type: {data_type}")
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split['train']
val_dataset = train_test_split['test']

# Tokenization
try:
    base_tokenizer = AutoTokenizer.from_pretrained("GKLMIP/bert-tagalog-base-uncased")
    logger.info("Base tokenizer loaded successfully")
    improved_model_path = "distilbert-base-uncased" if classification == "small" else "jcblaise/roberta-tagalog-base"
    improved_tokenizer = AutoTokenizer.from_pretrained(improved_model_path, do_lower_case=False)
    logger.info("Improved tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load tokenizers: {e}")
    raise

def tokenize(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

logger.info("Tokenizing datasets...")
try:
    tokenized_train_base = train_dataset.map(lambda x: tokenize(x, base_tokenizer), batched=True, remove_columns=["text"])
    tokenized_val_base = val_dataset.map(lambda x: tokenize(x, base_tokenizer), batched=True, remove_columns=["text"])
    tokenized_train_improved = train_dataset.map(lambda x: tokenize(x, improved_tokenizer), batched=True, remove_columns=["text"])
    tokenized_val_improved = val_dataset.map(lambda x: tokenize(x, improved_tokenizer), batched=True, remove_columns=["text"])
    logger.info("Datasets tokenized successfully")
except Exception as e:
    logger.error(f"Failed to tokenize datasets: {e}")
    raise

# Debug tokenized dataset
logger.info(f"Tokenized val base sample: {tokenized_val_base[0]}")
logger.info(f"Tokenized val base column names: {tokenized_val_base.column_names}")

# Load models
logger.info("Loading models...")
try:
    base_model = BertForMaskedLM.from_pretrained("GKLMIP/bert-tagalog-base-uncased").to(device)
    logger.info("Base model loaded successfully")
    improved_model = (DistilBertForMaskedLM.from_pretrained(improved_model_path) if classification == "small" 
                      else RobertaForMaskedLM.from_pretrained(improved_model_path)).to(device)
    logger.info("Improved model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise

# Fine-tuning
logger.info("Fine-tuning model...")
training_args = TrainingArguments(
    output_dir="/kaggle/working/output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    eval_strategy="no",
    logging_dir="/kaggle/working/logs",
    logging_steps=500,
    report_to="none",
    save_strategy="no"
)
if classification == "big":
    try:
        trainer = Trainer(
            model=improved_model,
            args=training_args,
            train_dataset=tokenized_train_improved,
            eval_dataset=tokenized_val_improved,
            data_collator=DataCollatorForLanguageModeling(tokenizer=improved_tokenizer, mlm=True, mlm_probability=0.2)
        )
        logger.info("Trainer initialized successfully")
        trainer.train()
        logger.info("Training completed successfully")
        logger.info(f"Saving fine-tuned model to {fine_tuned_model_path}")
        improved_model.save_pretrained(fine_tuned_model_path)
        # Verify model was saved
        if (fine_tuned_model_path / "model.safetensors").exists():
            logger.info(f"Model successfully saved to {fine_tuned_model_path}")
        else:
            logger.error(f"Failed to save model to {fine_tuned_model_path}")
    except Exception as e:
        logger.error(f"Failed during fine-tuning: {e}")
        raise
else:
    logger.info("Dataset is small, skipping fine-tuning")
"""

# Define comparison script content (unchanged for now)
COMPARISON_SCRIPT_CONTENT = """
# Install dependencies
!pip install transformers datasets torch evaluate huggingface_hub scikit-learn

# Import libraries
import os
import torch
import pandas as pd
import numpy as np
from transformers import (
    BertForMaskedLM,
    RobertaForMaskedLM,
    DistilBertForMaskedLM,
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
import logging
from sklearn.metrics import f1_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup
os.makedirs("/kaggle/working", exist_ok=True)
input_dir = Path('/kaggle/input') if Path('/kaggle/input').exists() else Path('.')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
torch.cuda.empty_cache()

# Load dataset
dataset_path = os.environ.get('DATASET_PATH', '')
if dataset_path:
    try:
        if dataset_path.endswith('.csv'):
            dataset = load_dataset('csv', data_files=dataset_path)
        else:
            dataset = load_dataset(dataset_path, split='train')
        dataset = dataset.filter(lambda x: x['text'] is not None and x['text'].strip() != '' and len(x['text'].split()) > 5)
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_path}: {e}")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        dataset = dataset.filter(lambda x: x['text'] is not None and x['text'].strip() != '' and len(x['text'].split()) > 5)
else:
    try:
        dataset_files = list(input_dir.glob('*.csv'))
        if not dataset_files:
            raise FileNotFoundError("No CSV file found.")
        dataset = load_dataset('csv', data_files=str(dataset_files[0]))
        dataset = dataset.filter(lambda x: x['text'] is not None and x['text'].strip() != '' and len(x['text'].split()) > 5)
    except Exception:
        logger.warning("No CSV found, falling back to WikiText")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        dataset = dataset.filter(lambda x: x['text'] is not None and x['text'].strip() != '' and len(x['text'].split()) > 5)

# Debug dataset structure
logger.info(f"Dataset structure: {dataset.column_names}")
logger.info(f"Sample data: {dataset[:2]}")

# Classify dataset
num_samples = len(dataset)
classification = "small" if num_samples < 512 else "big"
data_type = "standard NLP" if num_samples > 1000 else "low-resource NLP"
logger.info(f"Dataset size: {num_samples}, classification: {classification}, type: {data_type}")
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split['train']
val_dataset = train_test_split['test']

# Tokenization
base_tokenizer = AutoTokenizer.from_pretrained("GKLMIP/bert-tagalog-base-uncased")
improved_model_path = "distilbert-base-uncased" if classification == "small" else "jcblaise/roberta-tagalog-base"
improved_tokenizer = AutoTokenizer.from_pretrained(improved_model_path, do_lower_case=False)

def tokenize(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

logger.info("Tokenizing datasets...")
tokenized_train_base = train_dataset.map(lambda x: tokenize(x, base_tokenizer), batched=True, remove_columns=["text"])
tokenized_val_base = val_dataset.map(lambda x: tokenize(x, base_tokenizer), batched=True, remove_columns=["text"])
tokenized_train_improved = train_dataset.map(lambda x: tokenize(x, improved_tokenizer), batched=True, remove_columns=["text"])
tokenized_val_improved = val_dataset.map(lambda x: tokenize(x, improved_tokenizer), batched=True, remove_columns=["text"])

# Debug tokenized dataset
logger.info(f"Tokenized val base sample: {tokenized_val_base[0]}")
logger.info(f"Tokenized val base column names: {tokenized_val_base.column_names}")

# Load models
logger.info("Loading models...")
base_model = BertForMaskedLM.from_pretrained("GKLMIP/bert-tagalog-base-uncased").to(device)
fine_tuned_model_path = Path("/kaggle/working/fine_tuned_model")
if fine_tuned_model_path.exists() and (fine_tuned_model_path / "model.safetensors").exists():
    logger.info(f"Loading fine-tuned model from {fine_tuned_model_path}")
    try:
        improved_model = (DistilBertForMaskedLM.from_pretrained(fine_tuned_model_path) if classification == "small" 
                          else RobertaForMaskedLM.from_pretrained(fine_tuned_model_path)).to(device)
        logger.info("Fine-tuned model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load fine-tuned model: {e}")
        logger.info(f"Falling back to pre-trained model: {improved_model_path}")
        improved_model = (DistilBertForMaskedLM.from_pretrained(improved_model_path) if classification == "small" 
                          else RobertaForMaskedLM.from_pretrained(improved_model_path)).to(device)
else:
    logger.error("Fine-tuned model not found, comparison cannot proceed")
    raise FileNotFoundError("Fine-tuned model not found in /kaggle/working/fine_tuned_model")

# Evaluation function with enhanced metrics
def evaluate_mlm(model, tokenizer, dataset, device):
    model.eval()
    accuracy_metric = evaluate.load("accuracy")
    predictions, labels = [], []
    log_probs = []
    top_k_correct = {3: 0, 5: 0}
    total_samples = 0
    masked_token_freq = {}
    logger.info(f"Starting evaluation with dataset size: {len(dataset)}")
    for i in range(len(dataset)):
        try:
            input_ids = torch.tensor(dataset[i]['input_ids']).unsqueeze(0).to(device)
            attention_mask = torch.tensor(dataset[i]['attention_mask']).unsqueeze(0).to(device)
            mask_token_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            if len(mask_token_index) == 0:
                valid_indices = (input_ids[0] != tokenizer.pad_token_id) & \
                                (input_ids[0] != tokenizer.cls_token_id) & \
                                (input_ids[0] != tokenizer.sep_token_id)
                valid_indices = valid_indices.nonzero(as_tuple=True)[0]
                if len(valid_indices) == 0:
                    logger.warning(f"Sample {i}: No valid tokens to mask")
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
            # Track masked token frequency
            for token in original_token:
                masked_token_freq[token] = masked_token_freq.get(token, 0) + 1
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                # Compute log probabilities for perplexity
                probs = torch.softmax(logits[0, mask_token_index], dim=-1)
                for j, token in enumerate(original_token):
                    log_prob = torch.log(probs[j, token]).item()
                    if not np.isinf(log_prob):
                        log_probs.append(log_prob)
                # Top-1 prediction (accuracy)
                predicted_token_id = torch.argmax(logits[0, mask_token_index], dim=-1)
                predictions.extend(predicted_token_id.cpu().numpy().tolist())
                labels.extend(original_token)
                # Top-k accuracy
                total_samples += len(original_token)
                top_k_probs, top_k_indices = torch.topk(logits[0, mask_token_index], k=5, dim=-1)
                for j, true_token in enumerate(original_token):
                    for k in [3, 5]:
                        if true_token in top_k_indices[j, :k].cpu().numpy():
                            top_k_correct[k] += 1
        except Exception as e:
            logger.error(f"Error evaluating sample {i}: {e}")
            continue
    # Compute metrics
    metrics = {}
    if predictions:
        metrics["accuracy"] = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        metrics["f1"] = f1_score(labels, predictions, average="weighted")
        metrics["top_3_accuracy"] = top_k_correct[3] / total_samples if total_samples > 0 else 0.0
        metrics["top_5_accuracy"] = top_k_correct[5] / total_samples if total_samples > 0 else 0.0
    else:
        logger.warning("No valid predictions; setting metrics to 0")
        metrics["accuracy"] = 0.0
        metrics["f1"] = 0.0
        metrics["top_3_accuracy"] = 0.0
        metrics["top_5_accuracy"] = 0.0
    # Compute perplexity
    if log_probs:
        mean_log_prob = np.mean(log_probs)
        metrics["perplexity"] = np.exp(-mean_log_prob)
    else:
        metrics["perplexity"] = float("inf")
    metrics["num_predictions"] = len(predictions)
    metrics["num_samples_processed"] = total_samples
    # Log masked token frequency (top 5 tokens)
    top_masked_tokens = sorted(masked_token_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    metrics["top_masked_tokens"] = {tokenizer.convert_ids_to_tokens(k): v for k, v in top_masked_tokens}
    logger.info(f"Evaluation completed: {metrics}")
    return metrics

# Run evaluation with fallback
start_time = time.time()
logger.info("Evaluating BaseBERT model...")
try:
    base_metrics = evaluate_mlm(base_model, base_tokenizer, tokenized_val_base, device)
except Exception as e:
    logger.error(f"BaseBERT evaluation failed: {e}")
    base_metrics = {"accuracy": 0.0, "f1": 0.0, "perplexity": float("inf"), "top_3_accuracy": 0.0, "top_5_accuracy": 0.0, 
                    "num_predictions": 0, "num_samples_processed": 0, "top_masked_tokens": {}}
logger.info("Evaluating Improved model...")
try:
    improved_metrics = evaluate_mlm(improved_model, improved_tokenizer, tokenized_val_improved, device)
except Exception as e:
    logger.error(f"Improved model evaluation failed: {e}")
    improved_metrics = {"accuracy": 0.0, "f1": 0.0, "perplexity": float("inf"), "top_3_accuracy": 0.0, "top_5_accuracy": 0.0, 
                       "num_predictions": 0, "num_samples_processed": 0, "top_masked_tokens": {}}
eval_time = time.time() - start_time
logger.info(f"BaseBERT metrics: {base_metrics}")
logger.info(f"Improved model metrics: {improved_metrics}")
logger.info(f"Evaluation time: {eval_time:.2f} seconds")

# Save results
results = pd.DataFrame({
    "Model": ["BaseBERT", "Improved"],
    "Accuracy": [base_metrics["accuracy"], improved_metrics["accuracy"]],
    "F1_Score": [base_metrics["f1"], improved_metrics["f1"]],
    "Perplexity": [base_metrics["perplexity"], improved_metrics["perplexity"]],
    "Top_3_Accuracy": [base_metrics["top_3_accuracy"], improved_metrics["top_3_accuracy"]],
    "Top_5_Accuracy": [base_metrics["top_5_accuracy"], improved_metrics["top_5_accuracy"]],
    "Num_Predictions": [base_metrics["num_predictions"], improved_metrics["num_predictions"]],
    "EvalTimeSeconds": [eval_time / 2, eval_time / 2]
})
results.to_csv("/kaggle/working/results.csv", index=False)
logger.info("Saved results to /kaggle/working/results.csv")
with open("/kaggle/working/results.csv", "r") as f:
    logger.info(f"results.csv content: {f.read()}")

# Interpret results
threshold = 0.60
is_good = improved_metrics["accuracy"] >= threshold
interpretation = {
    "status": "Good" if is_good else "Needs Improvement",
    "reason": f"Improved model accuracy ({improved_metrics['accuracy']:.4f}) {'exceeds' if is_good else 'is below'} threshold ({threshold})",
    "context": {
        "base_metrics": base_metrics,
        "improved_metrics": improved_metrics,
        "notes": "Perplexity indicates model confidence (lower is better). F1 score balances precision and recall. Top-k accuracies show prediction robustness. Num_Predictions reflects valid samples processed."
    }
}
with open("/kaggle/working/interpretation.json", "w") as f:
    json.dump(interpretation, f, indent=4)
logger.info("Saved interpretation to /kaggle/working/interpretation.json")
with open("/kaggle/working/interpretation.json", "r") as f:
    logger.info(f"interpretation.json content: {f.read()}")
"""

st.title("CGABERT Model Comparison")

# Initialize Kaggle API
try:
    api = KaggleApi()
    api.authenticate()
    logger.info("Kaggle API authenticated successfully")
    # Test authentication with a simple API call
    datasets = api.dataset_list(max_size=1)
    logger.info(f"Successfully listed datasets: {datasets[:1]}")
    kernels = api.kernels_list(page_size=1)
    logger.info(f"Successfully listed kernels: {kernels[:1]}")
except Exception as e:
    st.error(f"Failed to authenticate Kaggle API: {e}")
    logger.error(f"Kaggle API authentication failed: {e}")
    st.error("Please ensure your Kaggle API token is correctly set up in C:\\Users\\Home\\.kaggle\\kaggle.json")
    st.stop()

# Use temporary directory for uploads and outputs
base_dir = Path(tempfile.mkdtemp())
upload_dir = base_dir / 'uploads'
kernel_dir = base_dir / 'kernels'
output_dir = base_dir / 'output'
for d in [upload_dir, kernel_dir, output_dir]:
    d.mkdir(exist_ok=True)

# File upload
uploaded_file = st.file_uploader("Upload a CSV dataset for comparison", type=["csv"])
dataset_path = None

if uploaded_file is not None:
    # Save the uploaded file
    dataset_path = upload_dir / uploaded_file.name
    with open(dataset_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Validate CSV
    try:
        df = pd.read_csv(dataset_path)
        if 'text' not in df.columns:
            st.error("CSV must contain a 'text' column")
            st.stop()
        st.success(f"Uploaded {uploaded_file.name} successfully")
    except Exception as e:
        st.error(f"Invalid CSV file: {e}")
        st.stop()

# Run comparison button
if st.button("Run Comparison"):
    logger.debug("Run Comparison button clicked")

    # If no dataset is uploaded, show warning
    if dataset_path is None:
        st.warning("Please upload a CSV file to run the comparison")
        st.stop()

    with st.spinner("Processing dataset on Kaggle..."):
        # Upload dataset to Kaggle
        dataset_name = f"cgabert-dataset-{int(time.time())}"
        dataset_metadata = {
            "title": dataset_name,
            "id": f"martinangelomagno/{dataset_name}",
            "licenses": [{"name": "CC0-1.0"}]
        }
        metadata_path = upload_dir / 'dataset-metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(dataset_metadata, f)
        logger.info(f"Uploading dataset to Kaggle: {dataset_name}")
        try:
            api.dataset_create_new(str(upload_dir), public=False)
            st.info(f"Dataset uploaded to Kaggle: {dataset_name}")
        except Exception as e:
            st.error(f"Failed to upload dataset: {e}")
            logger.error(f"Dataset upload failed: {e}")
            st.stop()

        # Step 1: Run training kernel
        train_kernel_dir = kernel_dir / 'train_kernel'
        train_kernel_dir.mkdir(exist_ok=True)
        train_kernel_script_path = train_kernel_dir / 'training.py'
        with open(train_kernel_script_path, 'w') as f:
            f.write(TRAINING_SCRIPT_CONTENT)

        train_kernel_id_suffix = str(uuid.uuid4())[:8]
        train_kernel_id = f"martinangelomagno/cgabert-train-{train_kernel_id_suffix}"

        os.environ['DATASET_PATH'] = f"/kaggle/input/{dataset_name}/{uploaded_file.name}"
        train_kernel_metadata = {
            "id": train_kernel_id,
            "title": f"cgabert-train-{train_kernel_id_suffix}",
            "code_file": "training.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [f"martinangelomagno/{dataset_name}"],
            "competition_sources": [],
            "kernel_sources": []
        }
        train_metadata_path = train_kernel_dir / 'kernel-metadata.json'
        logger.info(f"Writing training kernel metadata to {train_metadata_path}")
        with open(train_metadata_path, 'w') as f:
            json.dump(train_kernel_metadata, f)
        # Verify metadata file exists
        if not train_metadata_path.exists():
            logger.error(f"Training metadata file not created at {train_metadata_path}")
            st.error("Failed to create training kernel metadata file")
            st.stop()
        logger.debug(f"Pushing training kernel from {train_kernel_dir}")
        try:
            api.kernels_push(str(train_kernel_dir))
            st.info(f"Pushed training kernel: {train_kernel_id}")
            logger.info(f"Pushed training kernel: {train_kernel_id}")
        except Exception as e:
            st.error(f"Failed to push training kernel: {e}")
            logger.error(f"Training kernel push failed: {e}")
            st.stop()

        # Check training kernel status
        logger.debug("Checking training kernel status")
        max_attempts = 20
        for attempt in range(max_attempts):
            try:
                status_response = api.kernels_status(train_kernel_id)
                status = str(status_response.status).split('.')[-1]
                failure_message = getattr(status_response, 'failure_message', '') or ''
                logger.info(f"Training kernel status: {status}, Failure message: {failure_message}")
                st.write(f"Training kernel status: {status}")
                if status == "COMPLETE":
                    break
                elif status == "ERROR":
                    st.error(f"Training kernel failed: {failure_message or 'Unknown error'}")
                    logger.error(f"Training kernel failed: {failure_message or 'Unknown error'}")
                    st.stop()
                time.sleep(30)
            except Exception as e:
                st.warning(f"Training status check failed: {e}")
                logger.warning(f"Training status check failed: {e}")
                time.sleep(30)

        if status != "COMPLETE":
            st.error("Training kernel did not complete in time")
            logger.error("Training kernel did not complete in time")
            st.stop()

        # Step 2: Run comparison kernel
        compare_kernel_dir = kernel_dir / 'compare_kernel'
        compare_kernel_dir.mkdir(exist_ok=True)
        compare_kernel_script_path = compare_kernel_dir / 'comparison.py'
        with open(compare_kernel_script_path, 'w') as f:
            f.write(COMPARISON_SCRIPT_CONTENT)

        compare_kernel_id_suffix = str(uuid.uuid4())[:8]
        compare_kernel_id = f"martinangelomagno/cgabert-compare-{compare_kernel_id_suffix}"

        os.environ['DATASET_PATH'] = f"/kaggle/input/{dataset_name}/{uploaded_file.name}"
        compare_kernel_metadata = {
            "id": compare_kernel_id,
            "title": f"cgabert-compare-{compare_kernel_id_suffix}",
            "code_file": "comparison.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [f"martinangelomagno/{dataset_name}"],
            "competition_sources": [],
            "kernel_sources": []
        }
        compare_metadata_path = compare_kernel_dir / 'kernel-metadata.json'
        logger.info(f"Writing comparison kernel metadata to {compare_metadata_path}")
        with open(compare_metadata_path, 'w') as f:
            json.dump(compare_kernel_metadata, f)
        # Verify metadata file exists
        if not compare_metadata_path.exists():
            logger.error(f"Comparison metadata file not created at {compare_metadata_path}")
            st.error("Failed to create comparison kernel metadata file")
            st.stop()
        logger.debug(f"Pushing comparison kernel from {compare_kernel_dir}")
        try:
            api.kernels_push(str(compare_kernel_dir))
            st.info(f"Pushed comparison kernel: {compare_kernel_id}")
            logger.info(f"Pushed comparison kernel: {compare_kernel_id}")
        except Exception as e:
            st.error(f"Failed to push comparison kernel: {e}")
            logger.error(f"Comparison kernel push failed: {e}")
            st.stop()

        # Check comparison kernel status
        logger.debug("Checking comparison kernel status")
        for attempt in range(max_attempts):
            try:
                status_response = api.kernels_status(compare_kernel_id)
                status = str(status_response.status).split('.')[-1]
                failure_message = getattr(status_response, 'failure_message', '') or ''
                logger.info(f"Comparison kernel status: {status}, Failure message: {failure_message}")
                st.write(f"Comparison kernel status: {status}")
                if status == "COMPLETE":
                    break
                elif status == "ERROR":
                    st.error(f"Comparison kernel failed: {failure_message or 'Unknown error'}")
                    logger.error(f"Comparison kernel failed: {failure_message or 'Unknown error'}")
                    st.stop()
                time.sleep(30)
            except Exception as e:
                st.warning(f"Comparison status check failed: {e}")
                logger.warning(f"Comparison status check failed: {e}")
                time.sleep(30)

        if status != "COMPLETE":
            st.error("Comparison kernel did not complete in time")
            logger.error("Comparison kernel did not complete in time")
            st.stop()

        # Download comparison kernel output
        try:
            api.kernels_output(compare_kernel_id, str(output_dir))
            st.info("Downloaded comparison kernel output")
        except Exception as e:
            st.error(f"Failed to download output: {e}")
            logger.error(f"Output download failed: {e}")
            st.stop()

        # Display results
        try:
            results_df = pd.read_csv(output_dir / 'results.csv')
            with open(output_dir / 'interpretation.json', 'r') as f:
                interpretation = json.load(f)
            st.write("### Comparison Results")
            st.write(results_df)
            st.write("### Interpretation")
            st.json(interpretation)
        except FileNotFoundError as e:
            st.error(f"Error: Could not find output files: {e}")
            logger.error(f"Failed to load output files: {e}")
            st.stop()

# Display placeholder message if no action taken
else:
    st.info("Upload a CSV and click 'Run Comparison' to start")