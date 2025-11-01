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
│   ├── bangla_gec_mt5/
│   │   └── best_model/
│   ├── hindi_gec_mt5/
│   │   └── best_model/
│   ├── malayalam_gec_mt5/
│   │   └── best_model/
│   ├── tamil_gec_model/
│   │   └── best_model/
│   └── telugu_gec_mt5/
│       └── best_model/
├── Bangla/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── Hindi/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── Malayalam/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── Tamil/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── Telugu/
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

#### Tamil (91 samples)
```bash
cd Tamil
python train.py
```

**Configuration:**
- Model: `google/mt5-small`
- Batch Size: 2 (effective: 4 with gradient accumulation)
- Final Loss: 1.22

#### Telugu (599 samples)
```bash
cd Telugu
python train.py
```

**Configuration:**
- Model: `google/mt5-small`
- Batch Size: 4 (effective: 4 with gradient accumulation)
- Final Loss: ~2.1-2.4

#### Bangla (598 samples)
```bash
cd Bangla
python train.py
```

**Configuration:**
- Model: `google/mt5-small`
- Batch Size: 4 (effective: 4 with gradient accumulation)
- Final Loss: ~2.0-2.5

#### Hindi (600 samples)
```bash
cd Hindi
python train.py
```

**Configuration:**
- Model: `google/mt5-small`
- Batch Size: 4 (effective: 4 with gradient accumulation)

#### Malayalam (313 samples)
```bash
cd Malayalam
python train.py
```

**Configuration:**
- Model: `google/mt5-small`
- Batch Size: 4 (effective: 4 with gradient accumulation)

### Evaluation

```bash
# Tamil
cd Tamil
python evaluate.py

# Telugu
cd Telugu
python evaluate.py

# Bangla
cd Bangla
python evaluate.py

# Hindi
cd Hindi
python evaluate.py

# Malayalam
cd Malayalam
python evaluate.py
```

Generates `evaluation_results.json` with:
- GLEU Score
- BLEU Score
- Character Error Rate (CER)
- Sample predictions

## 📊 Results

### Tamil (mT5-small)
| Metric | Score |
|--------|-------|
| **GLEU Score** | **0.5344** |
| **BLEU Score** | 0.5059 |
| **Character Error Rate** | 0.9917 |
| **Training Samples** | 91 |
| **Test Samples** | 16 |

### Telugu (mT5-small)
| Metric | Score |
|--------|-------|
| **GLEU Score** | **0.7217** |
| **BLEU Score** | 0.6902 |
| **Character Error Rate** | 0.2987 |
| **Training Samples** | 539 |
| **Test Samples** | 100 |

### Bangla (mT5-small)
| Metric | Score |
|--------|-------|
| **GLEU Score** | **0.9278** |
| **BLEU Score** | 0.9252 |
| **Character Error Rate** | 0.0442 |
| **Training Samples** | 538 |
| **Test Samples** | 101 |

### Hindi (mT5-small)
| Metric | Score |
|--------|-------|
| **GLEU Score** | **0.8236** |
| **BLEU Score** | 0.8098 |
| **Character Error Rate** | 0.2126 |
| **Training Samples** | 600 |
| **Test Samples** | 107 |

### Malayalam (mT5-small)
| Metric | Score |
|--------|-------|
| **GLEU Score** | **0.6725** |
| **BLEU Score** | 0.6470 |
| **Character Error Rate** | 0.4401 |
| **Training Samples** | 313 |
| **Test Samples** | 50 |

### Key Findings
- **Bangla achieves highest performance:** Bangla (538 samples, GLEU 0.93) shows exceptional results, followed by Hindi (600 samples, GLEU 0.82)
- **Data quality matters:** Well-curated datasets lead to better performance
- **Low-resource effectiveness:** mT5-small performs well even with minimal training data (91 samples for Tamil achieves 0.53 GLEU)
- **Scalable approach:** Works efficiently on consumer-grade hardware (RTX 3050 4GB)

### Inference

```bash
# Tamil
cd Tamil
python inference.py

# Telugu
cd Telugu
python inference.py

# Bangla
cd Bangla
python inference.py

# Hindi
cd Hindi
python inference.py

# Malayalam
cd Malayalam
python inference.py
```

Interactive mode for testing corrections on custom sentences.

## 📝 Examples

### Tamil Language
```
Input:  -ஒளிப்பெருக்கி பயன்படுத்துதல்
Output: இதனால் ஒளிப்பெருக்கி பயன்படுத்துதல்
Reference: -ஒலிப்பெருக்கி பயன்படுத்துதல்
GLEU: 0.5344
```

### Telugu Language
```
Input:  లక్నో చేరుకునేసరికి సమయం ఎనిమిది అయ్యింది.
Output: మాక్నో చేరుకునేసరికి సమయం ఎనిమిది అయ్యింది.
Reference: లఖనౌ చేరుకునే సరికి సమయం ఎనిమిది అయ్యింది.
GLEU: 0.7217
```

### Bangla Language
```
Input:  ওই রূপ এবং ওই রুচির মূল্য কী করখ দেওয়া যায় তাই ভাবছি।
Output: ওই রূপ এবং ওই রুচির মূল্য কী করখ দেওয়া যায় তাই ভাবছি।
Reference: ওই রূপ এবং ওই রুচির মূল্য কী করে দেওয়া যায় তাই ভাবছি।
GLEU: 0.9278
```

### Hindi Language
```
Input:  दरअसल मानवीय गतिविधिया जैसे कि शहरीकरण, औद्योगिकरण इत्यादि के कारण विश्व का तापमान तेजी से बढ़ रहा है।
Output: परअसल मानवीय गतिविधिया जैसे कि शहरीकरण, औद्योगिकरण इत्यादि के कारण विश्व का तापमान तेजी से बढ़ रहा है।
Reference: दरअसल मानवीय गतिविधियां जैसे कि शहरीकरण, औद्योगीकरण इत्यादि के कारण विश्व का तापमान तेजी से बढ़ रहा है।
GLEU: 0.8236
```

### Malayalam Language
```
Input:  നമ്മള്ളുടെ ജീവശൈലിക്കനുസരിച്ച് മാലിന്യങ്ങൾ ഉണ്ടാകും എന്നതിൽ സംശയമില്ല.
Output: ആലിന്യങ്ങൾ ഉണ്ടാകും എന്നതിൽ സംശയമില്ല.
Reference: നമ്മുടെ ജീവിതശൈലിക്കനുസരിച്ച് മാലിന്യങ്ങൾ ഉണ്ടാകും എന്നതിൽ സംശയമില്ല.
GLEU: 0.6725
```

## 🎯 Model Architecture

### mT5-small (Multilingual T5)
- **Parameters:** ~300M
- **Architecture:** Encoder-Decoder Transformer
- **Tokenizer:** SentencePiece (handles all Indic scripts)
- **Max Sequence Length:** 64 tokens (optimized for memory)
- **Languages:** 101 languages including all major Indian languages
- **Pre-training Corpus:** Multilingual C4 (mC4)
- **Pre-training Task:** Span corruption (mask and predict text spans)

### Why mT5 for Low-Resource Indian Languages?

**1. Multilingual Pre-training**
- Trained on massive corpus covering Tamil, Telugu, Hindi, Malayalam, Bengali
- Cross-lingual transfer learning from high-resource to low-resource languages
- Shared vocabulary across Indic scripts

**2. Strong Performance**
- Telugu: GLEU 0.72 with just 599 training samples
- Tamil: GLEU 0.53 with only 91 training samples
- Fast convergence with limited data

**3. Resource Efficiency**
- Works on 4GB VRAM (RTX 3050)
- No need for expensive hardware
- Optimized for low-bandwidth environments

### Training Strategy
- **Data Usage:** All training samples used (no train/val split for small datasets)
- **Gradient Accumulation:** 2-4 steps (effective batch size: 4-8)
- **Loss Function:** Cross-entropy with label smoothing (0.1)
- **Gradient Clipping:** Max norm = 1.0
- **Memory Optimization:** Dynamic padding, FP16 disabled for stability
- **No evaluation during training:** Saves time and memory

## 🔧 Configuration

### Optimizing for Different Scenarios

**Standard Configuration:**
```python
batch_size = 4
learning_rate = 5e-5
gradient_accumulation_steps = 1
```

**Small Datasets:**
```python
batch_size = 2
learning_rate = 1e-4
gradient_accumulation_steps = 2
```

**Low Memory (< 4GB VRAM):**
```python
batch_size = 1
gradient_accumulation_steps = 4
max_length = 32
fp16 = False
# Required for GPUs with limited VRAM
```

**High Memory (> 8GB VRAM):**
```python
batch_size = 8
gradient_accumulation_steps = 1
max_length = 128
fp16 = True
# For faster training with powerful GPUs
```

## 📊 Evaluation Metrics

### GLEU Score (Primary)
- **Range:** 0.0 to 1.0
- **Interpretation:**
  - \> 0.7: Excellent performance ✅ (Telugu)
  - 0.5 - 0.7: Good performance ✅ (Tamil)
  - 0.3 - 0.5: Acceptable
  - < 0.3: Needs improvement
- **Character-level evaluation** for Indic scripts
- **Sensitive to spacing and punctuation**

### BLEU Score
- Standard metric for sequence-to-sequence tasks
- Character-level evaluation for Indic languages
- Complements GLEU score

### Character Error Rate (CER)
- **Formula:** `(insertions + deletions + substitutions) / total_characters`
- **Lower is better**
- Telugu: 0.30 (Good)
- Tamil: 0.99 (Needs improvement)

## 🐛 Troubleshooting

### Out of Memory Error
```python
# Reduce batch size in train.py
batch_size = 1
gradient_accumulation_steps = 4
```

### Slow Training
```python
# Increase batch size (if memory allows)
batch_size = 4

# Reduce epochs
epochs = 10

# Disable evaluation during training (already done)
eval_strategy = "no"
```

### Poor Results
- **Check for spacing issues:** GLEU is sensitive to whitespace
- **Collect more training data:** More data = better results (Telugu vs Tamil)
- **Adjust learning rate:** Try 5e-5 to 1e-4
- **Verify data quality:** Check for inconsistent annotations

### Model Download Issues
```python
# Use PyTorch format instead of SafeTensors (smaller, faster)
use_safetensors=False

# Already implemented in code for bandwidth optimization
```

### `<extra_id_0>` tokens in output
- Normal for T5 models during training
- Use `skip_special_tokens=True` in decoding (already done)

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