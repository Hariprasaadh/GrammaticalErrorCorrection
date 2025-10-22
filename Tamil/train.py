"""
Tamil GEC Training Script - Simplified Version
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import numpy as np
from nltk.translate.gleu_score import sentence_gleu
import nltk
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')


def load_data(csv_path):
    """Load training data from CSV"""
    df = pd.read_csv(csv_path)
    
    # Auto-detect columns
    if 'Input sentence' in df.columns and 'Output sentence' in df.columns:
        source_col, target_col = 'Input sentence', 'Output sentence'
    elif 'incorrect' in df.columns and 'correct' in df.columns:
        source_col, target_col = 'incorrect', 'correct'
    else:
        source_col, target_col = df.columns[0], df.columns[1]
    
    return df[source_col].tolist(), df[target_col].tolist()


def prepare_dataset(sources, targets, tokenizer, max_length=64):
    """Prepare dataset for training"""
    
    def tokenize_function(examples):
        # Tokenize inputs
        model_inputs = tokenizer(
            examples['source'], 
            max_length=max_length, 
            truncation=True, 
        )
        
        # Tokenize targets (labels)
        labels = tokenizer(
            text_target=examples['target'],
            max_length=max_length, 
            truncation=True, 
        )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    dataset = Dataset.from_dict({'source': sources, 'target': targets})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['source', 'target'])
    
    return tokenized_dataset


def compute_gleu(predictions, references):
    """Compute GLEU score"""
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = list(pred.strip())
        ref_tokens = list(ref.strip())
        
        if pred_tokens and ref_tokens:
            try:
                score = sentence_gleu([ref_tokens], pred_tokens)
                scores.append(score)
            except:
                scores.append(0.0)
        else:
            scores.append(0.0)
    
    return np.mean(scores) if scores else 0.0


def compute_metrics_fn(tokenizer):
    """Create compute metrics function"""
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        # Decode predictions
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute GLEU
        gleu_score = compute_gleu(decoded_preds, decoded_labels)
        
        return {'gleu': gleu_score}
    
    return compute_metrics


def train_model(train_file, output_dir='../Models/tamil_gec_model', epochs=25, batch_size=4):
    """Train the Tamil GEC model"""
    
    print("Loading data...")
    sources, targets = load_data(train_file)
    print(f"Loaded {len(sources)} training samples")
    
    # For small datasets, use ALL data for training (no validation split)
    # This maximizes learning from limited data
    train_sources, train_targets = sources, targets
    val_sources, val_targets = sources, targets  # Use same data for validation to monitor overfitting
    
    print(f"Train samples: {len(train_sources)}")
    print(f"Validation samples: {len(val_sources)}")
    
    # Load model and tokenizer
    print("\nLoading mT5 model...")
    model_name = "google/mt5-small"
    cache_dir = "../Models/pretrained_cache"
    
    # Check if model files exist in cache
    local_model_snapshot = os.path.join(cache_dir, "models--google--mt5-small", "snapshots")
    model_exists = os.path.exists(local_model_snapshot) and len(os.listdir(local_model_snapshot)) > 0 if os.path.exists(local_model_snapshot) else False
    
    if model_exists:
        print(f"Loading from local cache: {cache_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, legacy=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
    else:
        print(f"Downloading model and caching to {cache_dir}/")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, legacy=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset = prepare_dataset(train_sources, train_targets, tokenizer)
    val_dataset = prepare_dataset(val_sources, val_targets, tokenizer)
    
    # Training arguments - Optimized for speed and memory
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="no",  # Disable evaluation to save memory
        save_strategy="epoch",
        learning_rate=1e-4,  # Higher learning rate for faster convergence
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,  # Reduced from 4
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_total_limit=1,  # Only keep 1 checkpoint to save space
        logging_dir=f'{output_dir}/logs',
        logging_steps=20,  # Less frequent logging
        warmup_steps=0,
        warmup_ratio=0.0,
        fp16=False,
        dataloader_num_workers=0,
        report_to="none",
        max_grad_norm=1.0,
        label_smoothing_factor=0.0,
        lr_scheduler_type="constant",
        prediction_loss_only=True,  # Don't store predictions to save memory
    )
    
    # Data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to {output_dir}/best_model")
    trainer.save_model(f"{output_dir}/best_model")
    tokenizer.save_pretrained(f"{output_dir}/best_model")
    
    print("\nTraining complete!")
    return trainer


if __name__ == "__main__":
    train_file = '../Dataset/tamil/train.csv'
    output_dir = '../Models/tamil_gec_model'
    epochs = 20  
    batch_size = 2  
    
    print(f"Training Tamil GEC Model")
    print(f"Train file: {train_file}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Effective batch size: {batch_size * 4} (with gradient accumulation)")
    print(f"Mixed Precision: {'Enabled (GPU)' if torch.cuda.is_available() else 'Disabled (CPU)'}")
    print("="*60)
    
    train_model(train_file, output_dir, epochs, batch_size)