"""
Malayalam GEC Training Script using mT5-small
Optimized for fast training with minimal evaluation overhead
"""

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)


def load_data(file_path, source_col='Input sentence', target_col='Output sentence'):
    """Load and prepare data from CSV"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Try alternate column names if standard ones don't exist
    if source_col not in df.columns:
        if 'Input Sentence' in df.columns:
            source_col = 'Input Sentence'
    if target_col not in df.columns:
        if 'Output Sentence' in df.columns:
            target_col = 'Output Sentence'
    
    # Remove rows with NaN values and convert to string
    df = df.dropna(subset=[source_col, target_col])
    df[source_col] = df[source_col].astype(str)
    df[target_col] = df[target_col].astype(str)
    
    print(f"Loaded {len(df)} samples")
    
    data = {
        'source': df[source_col].tolist(),
        'target': df[target_col].tolist()
    }
    
    return Dataset.from_dict(data)


def preprocess_function(examples, tokenizer, max_length=64):
    """Tokenize input and target texts"""
    model_inputs = tokenizer(
        examples['source'],
        max_length=max_length,
        padding='max_length',
        truncation=True
    )
    
    labels = tokenizer(
        examples['target'],
        max_length=max_length,
        padding='max_length',
        truncation=True
    )
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def train_model():
    # Configuration
    model_name = 'google/mt5-small'
    train_file = '../Dataset/malayalam/train.csv'
    output_dir = '../Models/malayalam_gec_mt5'
    
    print("="*60)
    print("Malayalam GEC Training (mT5-small)")
    print("="*60)
    
    # Load tokenizer and model
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_safetensors=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=False)
    
    # Load and tokenize data
    train_dataset = load_data(train_file)
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=['source', 'target']
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments - optimized for speed
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="no",  # No evaluation during training
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_train_epochs=10,
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        fp16=False,
        predict_with_generate=False,
        load_best_model_at_end=False,
        report_to="none"
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    print("\nStarting training...")
    print(f"Total samples: {len(train_dataset)}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*60)
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(f"{output_dir}/best_model")
    tokenizer.save_pretrained(f"{output_dir}/best_model")
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Model saved to: {output_dir}/best_model")
    print("="*60)


if __name__ == "__main__":
    train_model()
