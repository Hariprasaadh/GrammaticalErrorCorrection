# Grammatical Error Correction for Low-Resource Indian Languages

This project presents a comprehensive deep learning solution for **Grammatical Error Correction (GEC)** in low-resource Indian languages. Leveraging the power of **Google's mT5 (Multilingual Text-to-Text Transfer Transformer)** architecture, we address the critical challenge of correcting grammatical errors in Indic scripts where annotated data is scarce.

Grammatical error correction is a fundamental task in Natural Language Processing (NLP) that aims to automatically detect and correct grammatical mistakes in written text. While significant progress has been made for high-resource languages like English, Indian languages present unique challenges due to:

- **Limited Training Data:** Most Indian languages have fewer than 100 annotated sentence pairs for GEC
- **Complex Morphology:** Rich inflectional systems and agglutinative word formation
- **Script Diversity:** Multiple writing systems (Devanagari, Tamil, Telugu, Bengali, Malayalam)
- **Code-Mixing:** Frequent mixing of English and native scripts in real-world text

Our solution employs **transfer learning** from mT5-small, a model pre-trained on 101 languages including all major Indian languages, to achieve strong performance even with minimal training data. Through careful optimization of training strategies, data handling, and hyperparameter tuning, we demonstrate that effective GEC systems can be built for low-resource scenarios.


## 📋 Task Description

This project focuses on **Grammatical Error Correction (GEC)** for Indian languages in a **low-resource setting**. The task involves correcting grammatical errors in sentences written in:

- 🇧🇩 **Bangla**
- 🇮🇳 **Hindi**
- 🇮🇳 **Malayalam**
- 🇮🇳 **Tamil**
- 🇮🇳 **Telugu**

### Performance Metric
**GLEU Score** (Generalized Language Evaluation Understanding) is used as the primary evaluation metric.

## 🏗️ Project Structure

```
GrammaticalErrorCorrection/
├── Dataset/
│   ├── bangala/
│   │   ├── train.csv
│   │   └── dev.csv
│   ├── hindi/
│   │   ├── train.csv
│   │   └── dev.csv
│   ├── malayalam/
│   │   ├── train.csv
│   │   └── dev.csv
│   ├── tamil/
│   │   ├── train.csv
│   │   └── dev.csv
│   └── telugu/
│       ├── train.csv
│       └── dev.csv
├── Models/
│   └── tamil_gec_model/
│       └── best_model/
├── Tamil/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
└── README.md
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch transformers datasets pandas nltk
```

### Training

```bash
cd Tamil
python train.py
```

**Training Configuration:**
- Model: `google/mt5-small`
- Epochs: 20
- Batch Size: 2 (effective: 8 with gradient accumulation)
- Learning Rate: 1e-4
- Optimizer: AdamW
- Device: CUDA (GPU) / CPU

### Evaluation

```bash
python evaluate.py
```

Generates `evaluation_results.json` with:
- GLEU Score
- BLEU Score
- Character Error Rate (CER)
- Exact Match Accuracy
- Sample predictions

### Inference

```bash
python inference.py
```

Interactive mode for testing corrections on custom sentences.

## 📝 Examples

### Tamil Language
```
Input:  தமிழ் மொழி மிகவும் பழமையான மொழி
Output: தமிழ் மொழி மிகவும் பழமையான மொழி
```

### Hindi Language
```
Input:  इस वस्तुका उपयोग मत करो।
Output: इस वस्तु का उपयोग मत करो।
```

### Malayalam Language
```
Input:  നമ്മുടെ ജീവൈശലി അനുസരിച്ച് മാലിന്യങ്ങൾ ഉണ്ടാകും എന്നതിൽ സംശയമില്ല.
Output: നമ്മുടെ ജീവിതൈശലി അനുസരിച്ച് മാലിന്യങ്ങൾ ഉണ്ടാകും എന്നതിൽ സംശയമില്ല.
```

## 🎯 Model Architecture

### mT5-small
- **Parameters:** ~300M
- **Architecture:** Encoder-Decoder Transformer
- **Tokenizer:** SentencePiece
- **Max Sequence Length:** 64 tokens
- **Languages:** Multilingual (101 languages including Indic scripts)

### Training Strategy
- **Data Augmentation:** All training samples used (no train/val split for small datasets)
- **Gradient Accumulation:** 2 steps (effective batch size: 8)
- **Loss Function:** Cross-entropy with label smoothing
- **Gradient Clipping:** Max norm = 1.0
- **Memory Optimization:** Dynamic padding, no FP16 (for stability)

## 🔧 Configuration

### Optimizing for Different Scenarios

**Faster Training (Less Accuracy):**
```python
epochs = 10
batch_size = 4
learning_rate = 2e-4
```

**Better Accuracy (Slower Training):**
```python
epochs = 50
batch_size = 1
learning_rate = 5e-5
```

**Low Memory (< 4GB VRAM):**
```python
batch_size = 1
gradient_accumulation_steps = 4
max_length = 32
```

## 📊 Evaluation Metrics

### GLEU Score (Primary)
- **Range:** 0.0 to 1.0
- **Interpretation:**
  - \> 0.5: Good performance
  - 0.3 - 0.5: Acceptable
  - < 0.3: Needs improvement

### BLEU Score
- Standard metric for sequence-to-sequence tasks
- Character-level evaluation for Indic languages

### Character Error Rate (CER)
- **Formula:** `(insertions + deletions + substitutions) / total_characters`
- Lower is better

## 🐛 Troubleshooting

### Out of Memory Error
```python
# Reduce batch size in train.py
batch_size = 1
```

### Slow Training
```python
# Enable FP16 (if stable)
fp16 = True

# Increase batch size
batch_size = 4
```

### Poor Results
- Increase training epochs
- Collect more training data
- Try data augmentation
- Adjust learning rate

## 📚 Dataset Format

CSV files with two columns:

| Input sentence | Output sentence |
|---------------|-----------------|
| Incorrect text | Corrected text |

**Example (`train.csv`):**
```csv
Input sentence,Output sentence
இந்த வாக்கிய தவறு,இந்த வாக்கியம் தவறு
```

## 📄 License

This project is developed for academic purposes.