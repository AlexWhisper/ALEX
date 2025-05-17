
# ALEX: A Light Editing-knowledge Extractor  

## Overview  
ALEX is an open-source framework designed to address the challenges of **factual obsolescence** and **scalability** in large language models (LLMs), particularly for multi-hop question answering (MQA). By integrating **hierarchical memory compression** and **dynamic retrieval arbitration**, ALEX achieves efficient knowledge editing with minimal computational overhead. This repository contains the official implementation of the ALEX framework, along with experimental datasets and evaluation scripts.  

## Key Features  
- **Hierarchical Retrieval**: Decouples retrieval into **cluster-level coarse filtering** and **edit-level fine-grained scoring**, reducing search complexity from \(O(N)\) to \(O(K + N/C)\).  
- **Dual-Objective Training**: Optimizes intra-cluster cohesion and inter-cluster contrastiveness for high retrieval accuracy (99%+ with 5 epochs).  
- **Zero-Shot Hypothetical Caching**: Generates query variants via LLMs to enhance semantic coverage without retraining.  
- **Efficiency**: Reduces edit search space by up to 80% while maintaining accuracy on MQuAKE datasets.  


## Architecture Overview  
ALEX consists of three core modules:  
1. **Cluster Module**: Uses K-means++ clustering to compress edits into semantic clusters, minimizing inter-cluster distance and maximizing intra-cluster similarity.  
2. **Arbitration Module**: Implements a two-stage filter (z-score normalization and multi-faceted scoring) to select relevant edits efficiently.  
3. **E2Q (Edit-to-Question) Module**: Generates hypothetical queries from edits using LLMs, cached for reuse to reduce latency.  


## Installation  
### Dependencies  
- Python 3.9+  
- PyTorch 2.0+  
- Hugging Face Transformers  
- SentenceTransformers  
- Scikit-learn  
- NumPy  


## Usage Guide  
### 1. Data Preparation  
- Download MQuAKE datasets and extract them into the `datasets/` directory.  


### 2. Evaluation  
Run multi-hop accuracy evaluation on MQuAKE datasets:  
```bash  
python main.py
```  




## License  
ALEX is released under the **MIT License**.  
- LLaMA models are licensed under the [LLaMA 2 Community License](https://ai.meta.com/llama/).  
- MQuAKE datasets follow their original licensing terms.  

For inquiries, contact: alex-project@example.com  

---  
**Note**: This is a pre-release version for research purposes. Contributions and issue reports are welcome!
