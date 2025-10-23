"""
Telugu GEC Evaluation Script
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')


def load_model(model_path):
    """Load trained model and tokenizer"""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    return tokenizer, model, device


def generate_corrections(sentences, tokenizer, model, device, num_beams=5):
    """Generate corrections for sentences"""
    corrections = []
    
    print(f"Generating corrections for {len(sentences)} sentences...")
    for i, sentence in enumerate(sentences):
        if i % 10 == 0:
            print(f"Processing {i}/{len(sentences)}...")
        
        inputs = tokenizer(
            sentence,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=128,
                num_beams=num_beams,
                early_stopping=True
            )
        
        corrected = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        corrections.append(corrected)
    
    return corrections


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


def compute_bleu(predictions, references):
    """Compute BLEU score"""
    smoothing = SmoothingFunction().method1
    scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = list(pred.strip())
        ref_tokens = list(ref.strip())
        
        if pred_tokens and ref_tokens:
            try:
                score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
                scores.append(score)
            except:
                scores.append(0.0)
        else:
            scores.append(0.0)
    
    return np.mean(scores) if scores else 0.0


def compute_exact_match(predictions, references):
    """Compute exact match accuracy"""
    exact_matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    total = len(predictions)
    accuracy = exact_matches / total if total > 0 else 0.0
    
    return exact_matches, total, accuracy


def compute_cer(predictions, references):
    """Compute Character Error Rate"""
    
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    total_distance = 0
    total_length = 0
    
    for pred, ref in zip(predictions, references):
        distance = levenshtein_distance(pred.strip(), ref.strip())
        total_distance += distance
        total_length += len(ref.strip())
    
    return total_distance / total_length if total_length > 0 else 0.0


def evaluate(model_path, test_file, output_file='evaluation_results.json', num_beams=5):
    """Evaluate model on test file"""
    
    # Load test data
    print(f"Loading test data from {test_file}...")
    df = pd.read_csv(test_file)
    
    # Auto-detect columns
    if 'Input sentence' in df.columns and 'Output sentence' in df.columns:
        source_col, target_col = 'Input sentence', 'Output sentence'
    elif 'incorrect' in df.columns and 'correct' in df.columns:
        source_col, target_col = 'incorrect', 'correct'
    else:
        source_col, target_col = df.columns[0], df.columns[1]
    
    sources = df[source_col].tolist()
    references = df[target_col].tolist()
    print(f"Loaded {len(sources)} test samples")
    
    # Load model
    tokenizer, model, device = load_model(model_path)
    
    # Generate predictions
    predictions = generate_corrections(sources, tokenizer, model, device, num_beams)
    
    # Compute metrics
    print("\nComputing metrics...")
    gleu_score = compute_gleu(predictions, references)
    bleu_score = compute_bleu(predictions, references)
    cer = compute_cer(predictions, references)
    exact_matches, total, exact_match_acc = compute_exact_match(predictions, references)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"GLEU Score:              {gleu_score:.4f}")
    print(f"BLEU Score:              {bleu_score:.4f}")
    print(f"Character Error Rate:    {cer:.4f}")
    print(f"Exact Match Accuracy:    {exact_match_acc:.4f}")
    print(f"Exact Matches:           {exact_matches}/{total}")
    print("="*60)
    
    # Save results
    results = {
        'metrics': {
            'gleu_score': float(gleu_score),
            'bleu_score': float(bleu_score),
            'character_error_rate': float(cer),
            'exact_match_accuracy': float(exact_match_acc),
            'exact_matches': int(exact_matches),
            'total': int(total)
        },
        'examples': [
            {
                'source': src,
                'prediction': pred,
                'reference': ref
            }
            for src, pred, ref in zip(sources[:20], predictions[:20], references[:20])
        ]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    # Set paths directly in code
    model_path = '../Models/telugu_gec_indicbart/best_model'
    test_file = '../Dataset/telugu/dev.csv'
    output_file = 'evaluation_results.json'
    num_beams = 5
    
    print(f"Evaluating Telugu GEC Model (IndicBART)")
    print(f"Model: {model_path}")
    print(f"Test file: {test_file}")
    print(f"Output: {output_file}")
    print("="*60)
    
    evaluate(model_path, test_file, output_file, num_beams)
