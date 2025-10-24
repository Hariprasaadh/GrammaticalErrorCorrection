"""
Bangla GEC Training Script - Using mT5-small
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
    
    # Remove rows with NaN values and convert to string
    df = df.dropna(subset=[source_col, target_col])
    sources = df[source_col].astype(str).tolist()
    targets = df[target_col].astype(str).tolist()
    
    return sources, targets


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


def train_model(train_file, output_dir='../Models/bangla_gec_mt5', epochs=10, batch_size=4):
    """Train the Bangla GEC model"""
    
    print("Loading data...")
    sources, targets = load_data(train_file)
    print(f"Loaded {len(sources)} training samples")
    
    # For medium-sized datasets, use 90% for training
    split_idx = int(0.9 * len(sources))
    train_sources, train_targets = sources[:split_idx], targets[:split_idx]
    val_sources, val_targets = sources[split_idx:], targets[split_idx:]
    
    print(f"Train samples: {len(train_sources)}")
    print(f"Validation samples: {len(val_sources)}")
    
    # Load model and tokenizer
    print("\nLoading mT5-small model...")
    model_name = "google/mt5-small"
    cache_dir = "../Models/pretrained_cache"
    
    # Check if model files exist in cache
    local_model_snapshot = os.path.join(cache_dir, "models--google--mt5-small", "snapshots")
    model_exists = os.path.exists(local_model_snapshot) and len(os.listdir(local_model_snapshot)) > 0 if os.path.exists(local_model_snapshot) else False
    
    if model_exists:
        print(f"Loading from local cache: {cache_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_safetensors=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir, use_safetensors=False)
    else:
        print(f"Downloading model and caching to {cache_dir}/")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_safetensors=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir, use_safetensors=False)
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset = prepare_dataset(train_sources, train_targets, tokenizer)
    val_dataset = prepare_dataset(val_sources, val_targets, tokenizer)
    
    # Training arguments - Optimized for FASTEST training (no eval)
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="no",  # Disabled for faster training
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,  # Reduced for faster training
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_total_limit=1,  # Keep only last checkpoint
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,  # Reduced logging frequency
        warmup_steps=50,  # Reduced warmup
        fp16=False,
        dataloader_num_workers=0,
        report_to="none",
        max_grad_norm=1.0,
        label_smoothing_factor=0.1,
        lr_scheduler_type="linear",
        prediction_loss_only=True,
    )
    
    # Data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100
    )
    
    # Create trainer (no evaluation during training)
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
    train_file = '../Dataset/bangala/train.csv'
    output_dir = '../Models/bangla_gec_mt5'
    epochs = 10  # Fast training with mT5
    batch_size = 4
    
    print(f"Training Bangla GEC Model with mT5-small")
    print(f"Train file: {train_file}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Effective batch size: {batch_size * 1} (with gradient accumulation)")
    print(f"Evaluation during training: Disabled (run evaluate.py after training)")
    print(f"Estimated time: 8-10 minutes")
    print(f"Mixed Precision: {'Enabled (GPU)' if torch.cuda.is_available() else 'Disabled (CPU)'}")
    print("="*60)
    
    train_model(train_file, output_dir, epochs, batch_size)
