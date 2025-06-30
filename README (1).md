
# Advanced Forgetting Mechanisms for Knowledge Graph-Based Recommendation Systems

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Kaggle Dataset](https://img.shields.io/badge/dataset-Amazon_Reviews-orange.svg)](https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Bachelor Thesis**  
> **Institution**: Constructor University Bremen  
> **Submission Date**: June 2025  
> **Author**: Anactacia Lomtadze  
> **Supervisor**: Prof. Dr. A. Tormasov  

---

## Overview

This research project investigates whether biologically inspired forgetting mechanisms can improve the adaptability, personalization, and efficiency of knowledge graph-based recommender systems. The study systematically implements and evaluates four cognitive-inspired forgetting paradigms on the Amazon Electronics dataset.

---

## Key Finding

Cognitive forgetting mechanisms result in up to **79.2% performance loss**, while offering only marginal **5–15% memory savings** in sparse data environments.

---

## Summary of Results

| Method                      | Hit Rate@10 | Performance Impact |
|----------------------------|-------------|---------------------|
| Quality-Based (Baseline)   | 12.0%       | Highest             |
| Popularity-Based           | 9.0%        | Moderate            |
| Neural Adaptive Forgetting | 2.0%        | -83%                |
| Attention-Based Forgetting | 2.0%        | -83%                |
| Cascade Forgetting         | 1.0%        | -92%                |
| Contextual Forgetting      | 1.0%        | -92%                |

---

## Installation Instructions

Install the required dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn networkx scikit-learn tqdm kagglehub[pandas-datasets]
```

---

## Execution

To evaluate the system, execute the following scripts:

```bash
# Full pipeline execution
python thesis_final_evaluation.py

# Individual forgetting mechanism testing
python test_fixed_mechanisms.py
```

---

## Dataset Access

The dataset used is available on Kaggle:

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

## Core Components

- `thesis_final_evaluation.py`: Evaluation pipeline
- `fixed_advanced_forgetting.py`: Implementation of forgetting mechanisms
- `test_fixed_mechanisms.py`: Mechanism-specific testing
- `amazon_knowledge_graph.py`: Knowledge graph construction logic

---

## Scientific Insights

### Why Forgetting Mechanisms Underperform

1. **Data sparsity** severely limits pruning flexibility.
2. **High interaction value** means even weak links are informative.
3. **Cold-start issues** are exacerbated by removal of early data.

### Mechanism Summary

| Mechanism              | Cognitive Basis            | Finding                        |
|------------------------|----------------------------|--------------------------------|
| Neural Adaptive        | Synaptic Plasticity        | Cannot adapt to low signal     |
| Attention-Based        | Selective Attention        | Discards critical interactions |
| Cascade Forgetting     | Spreading Activation       | Over-prunes connected nodes    |
| Contextual Forgetting  | Context-Dependent Memory   | Insufficient context signals   |

---

## Dataset Information

- **Domain**: Electronics (Amazon)
- **Ratings**: 147,412
- **Users**: 4,883
- **Items**: 9,956
- **Sparsity**: ~97%

---

## Contributions

- Introduces a formal evaluation framework for memory-aware KGRS.
- Demonstrates that cognitive forgetting harms performance in sparse domains.
- Provides baseline comparisons and reproducible benchmarks.

---

## Future Research Directions

- **Sparsity-aware forgetting**: Adaptive forgetting based on data richness.
- **Hybrid retention-forgetting models**.
- **Cross-domain generalization**: Evaluate on denser datasets such as MovieLens.
- **Personalized decay logic** for user segments.

---

## Citation

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

## Final Remark

This thesis serves as the first comprehensive analysis showing that biologically motivated forgetting techniques, while conceptually grounded, are ineffective in sparsity-dominated recommendation settings. Their application requires careful tailoring to data characteristics.

---

**Dataset**: [Amazon Product Reviews](https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews)  
**Thesis Document**: `thesis_document.pdf`
