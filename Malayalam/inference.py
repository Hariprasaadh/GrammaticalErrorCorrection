"""
Malayalam GEC Inference Script
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


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


def correct_sentence(sentence, tokenizer, model, device, num_beams=5):
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
            num_beams=num_beams,
            early_stopping=True
        )
    
    corrected = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return corrected


def interactive_mode(model_path, num_beams=5):
    """Run in interactive mode"""
    tokenizer, model, device = load_model(model_path)
    
    print("\n" + "="*60)
    print("Malayalam GEC Interactive Mode")
    print("Type 'quit' or 'exit' to stop")
    print("="*60)
    
    while True:
        sentence = input("\nEnter Malayalam sentence: ")
        
        if sentence.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not sentence.strip():
            continue
        
        corrected = correct_sentence(sentence, tokenizer, model, device, num_beams)
        print(f"Corrected: {corrected}")


def correct_file(model_path, input_file, output_file=None, num_beams=5):
    """Correct sentences from a file"""
    tokenizer, model, device = load_model(model_path)
    
    print(f"\nReading from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    print(f"Correcting {len(sentences)} sentences...")
    corrected_sentences = []
    
    for i, sentence in enumerate(sentences):
        if i % 10 == 0:
            print(f"Processing {i}/{len(sentences)}...")
        
        corrected = correct_sentence(sentence, tokenizer, model, device, num_beams)
        corrected_sentences.append(corrected)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for corrected in corrected_sentences:
                f.write(corrected + '\n')
        print(f"\nCorrected sentences saved to {output_file}")
    else:
        print("\nResults:")
        print("="*60)
        for orig, corr in zip(sentences, corrected_sentences):
            print(f"Original:  {orig}")
            print(f"Corrected: {corr}")
            print()


def correct_single(model_path, sentence, num_beams=5):
    """Correct a single sentence"""
    tokenizer, model, device = load_model(model_path)
    corrected = correct_sentence(sentence, tokenizer, model, device, num_beams)
    
    print(f"\nInput:     {sentence}")
    print(f"Corrected: {corrected}")


if __name__ == "__main__":
    # Set paths directly in code
    model_path = '../Models/malayalam_gec_mt5/best_model'
    
    print(f"Malayalam GEC Inference (mT5-small)")
    print(f"Model: {model_path}")
    print("="*60)
    
    # Run in interactive mode
    interactive_mode(model_path)
