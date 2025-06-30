# Advanced Forgetting Mechanisms for Knowledge Graph-Based Recommendation Systems

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Kaggle Dataset](https://img.shields.io/badge/dataset-Amazon_Reviews-orange.svg)](https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Bachelor Thesis** | Constructor University Bremen | June 2025  
> **Author:** Anactacia Lomtadze  
> **Supervisor:** Prof. Dr. A. Tormasov  

---

## 🧠 Overview

This research explores whether **cognitive forgetting mechanisms** can enhance **knowledge graph-based recommender systems (KGRS)**.  
In sparse data environments like Amazon product reviews, they fail dramatically.

### 🔍 Key Finding

> Forgetting mechanisms led to up to **79.2% performance loss** for just **5–15% memory savings**.

---

## 📊 Results Summary

| Method                      | Hit Rate@10 | Status         |
|----------------------------|-------------|----------------|
| **Quality-Based (Baseline)** | **12.0%**    | ✅ Best         |
| Popularity-Based           | 9.0%        | ✅ Good         |
| Neural Adaptive Forgetting | 2.0%        | ❌ -83%         |
| Attention-Based Forgetting | 2.0%        | ❌ -83%         |
| Cascade Forgetting         | 1.0%        | ❌ -92%         |
| Contextual Forgetting      | 1.0%        | ❌ -92%         |

---

## 🚀 Quick Start

### 1. Installation

```bash
pip install pandas numpy matplotlib seaborn networkx scikit-learn tqdm kagglehub[pandas-datasets]
```

### 2. Run Full Evaluation

```bash
python thesis_final_evaluation.py
```

### 3. Run Individual Mechanism Test

```bash
python test_fixed_mechanisms.py
```

### 4. Load Dataset

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "saurav9786/amazon-product-reviews",
    ""
)
```

---

## 🧩 Core Files

- `thesis_final_evaluation.py` — Main pipeline for evaluation
- `fixed_advanced_forgetting.py` — All forgetting mechanisms implemented here
- `test_fixed_mechanisms.py` — Unit tests and case studies
- `amazon_knowledge_graph.py` — Knowledge graph construction and enrichment

---

## 📘 Scientific Insights

### Why Forgetting Failed

1. **Data sparsity** — Only ~0.3% of potential user-item edges exist
2. **No redundancy** — Every edge is valuable; pruning causes breakage
3. **Cold-start amplification** — Makes new user experiences even worse

### Cognitive Mechanism Evaluation

| Mechanism              | Inspired by            | Insight                        |
|------------------------|------------------------|--------------------------------|
| Neural Adaptive        | Synaptic plasticity    | Can't adapt to sparse signals |
| Attention-Based        | Selective attention    | Discards critical paths        |
| Cascade Forgetting     | Spreading activation   | Over-prunes interconnected nodes |
| Contextual Forgetting  | Episodic/context memory| Too few signals to use        |

---

## 📦 Dataset Details

- **Source**: [Amazon Product Reviews](https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews)
- **Category**: Electronics
- **Size**: 147,412 ratings, 4,883 users, 9,956 items
- **Sparsity**: ~97%

---

## 🧪 Scientific Contribution

> 🔬 **First study to systematically demonstrate that cognitive forgetting mechanisms harm recommendation performance in sparse graph-based settings.**

### Impact

- Sets boundaries on applicability of cognitive-inspired models
- Highlights the risk of over-pruning in low-density graphs
- Provides an extensible evaluation framework for future research

---

## 🔮 Future Work

- **Sparsity-aware forgetting** — Apply forgetting only where data is dense
- **Hybrid methods** — Combine retention with selective decay
- **Cross-domain evaluation** — Test on MovieLens, Last.fm, etc.
- **User-aware decay** — Personalize forgetting per user profile

---

## 📖 Citation

```bibtex
@thesis{lomtadze2025forgetting,
  title={Advanced Forgetting Mechanisms for Knowledge Graph-Based Recommendation Systems: A Study on Amazon Product Data},
  author={Lomtadze, Anactacia},
  year={2025},
  school={Constructor University Bremen},
  type={Bachelor Thesis}
}
```

---

## ❗ Final Takeaway

> ❌ **Don't apply cognitive forgetting mechanisms to sparse recommendation data**  
They significantly reduce performance without meaningful memory savings.

---

**📎 Dataset**: [Amazon Product Reviews](https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews)  
**📄 Full Thesis**: `thesis_document.pdf`
