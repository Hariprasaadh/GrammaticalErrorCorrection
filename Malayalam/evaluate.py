"""
Malayalam GEC Evaluation Script
Evaluates the trained model on dev set using GLEU, BLEU, CER metrics
"""

import pandas as pd
import torch
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.bleu_score import sentence_bleu


def calculate_cer(reference, hypothesis):
    """Calculate Character Error Rate"""
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    
    # Simple Levenshtein distance
    m, n = len(ref_chars), len(hyp_chars)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n] / max(m, 1)


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


def correct_sentence(sentence, tokenizer, model, device):
    """Correct a single sentence"""
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
            num_beams=5,
            early_stopping=True
        )
    
    corrected = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # Normalize whitespace
    corrected = ' '.join(corrected.split())
    return corrected


def evaluate_model(model_path, test_file, output_file='evaluation_results.json'):
    """Evaluate model on test set"""
    print("="*60)
    print("Malayalam GEC Evaluation")
    print("="*60)
    
    # Load model
    tokenizer, model, device = load_model(model_path)
    
    # Load test data
    print(f"\nLoading test data from {test_file}...")
    df = pd.read_csv(test_file)
    
    # Check for column names (handle both formats)
    source_col = 'Input sentence' if 'Input sentence' in df.columns else 'Input Sentence'
    target_col = 'Output sentence' if 'Output sentence' in df.columns else 'Output Sentence'
    
    # Remove NaN and convert to string
    df = df.dropna(subset=[source_col, target_col])
    df[source_col] = df[source_col].astype(str)
    df[target_col] = df[target_col].astype(str)
    
    sources = df[source_col].tolist()
    references = df[target_col].tolist()
    
    print(f"Loaded {len(sources)} test samples")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = []
    for i, source in enumerate(sources):
        if i % 10 == 0:
            print(f"Processing {i}/{len(sources)}...")
        
        pred = correct_sentence(source, tokenizer, model, device)
        predictions.append(pred)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    gleu_scores = []
    bleu_scores = []
    cer_scores = []
    exact_matches = 0
    
    for pred, ref in zip(predictions, references):
        # GLEU (character-level for Malayalam)
        gleu = sentence_gleu([list(ref)], list(pred))
        gleu_scores.append(gleu)
        
        # BLEU (character-level)
        bleu = sentence_bleu([list(ref)], list(pred))
        bleu_scores.append(bleu)
        
        # CER
        cer = calculate_cer(ref, pred)
        cer_scores.append(cer)
        
        # Exact match
        if pred.strip() == ref.strip():
            exact_matches += 1
    
    # Average metrics
    avg_gleu = sum(gleu_scores) / len(gleu_scores)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_cer = sum(cer_scores) / len(cer_scores)
    exact_match_acc = exact_matches / len(references)
    
    # Prepare results
    results = {
        'metrics': {
            'gleu_score': avg_gleu,
            'bleu_score': avg_bleu,
            'character_error_rate': avg_cer,
            'exact_match_accuracy': exact_match_acc,
            'exact_matches': exact_matches,
            'total': len(references)
        },
        'examples': [
            {
                'source': src,
                'prediction': pred,
                'reference': ref
            }
            for src, pred, ref in list(zip(sources, predictions, references))[:20]
        ]
    }
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"GLEU Score:              {avg_gleu:.4f}")
    print(f"BLEU Score:              {avg_bleu:.4f}")
    print(f"Character Error Rate:    {avg_cer:.4f}")
    print(f"Exact Match Accuracy:    {exact_match_acc:.4f} ({exact_matches}/{len(references)})")
    print("="*60)
    print(f"\nResults saved to: {output_file}")
    
    # Print some examples
    print("\nExample Predictions:")
    print("="*60)
    for i, (src, pred, ref) in enumerate(zip(sources[:5], predictions[:5], references[:5])):
        print(f"\nExample {i+1}:")
        print(f"Source:     {src}")
        print(f"Prediction: {pred}")
        print(f"Reference:  {ref}")
        print(f"GLEU:       {gleu_scores[i]:.4f}")


if __name__ == "__main__":
    model_path = '../Models/malayalam_gec_mt5/best_model'
    test_file = '../Dataset/malayalam/dev.csv'
    
    evaluate_model(model_path, test_file)
