# Flow-Based Graph Clustering for Network Intrusion Detection

> Graph-based anomaly detection (تشخیص ناهنجاری) for network traffic using similarity graphs, clustering, and flow-based cluster refinement.

## Overview

This project investigates whether network-flow data can benefit from a graph representation before clustering. It was developed as an Advanced Network Optimization course project and connects machine learning with graph algorithms.

## Pipeline

```text
NSL-KDD
   ↓
Preprocessing
   ↓
Feature Encoding
   ↓
Baseline: K-Means + PCA
   ↓
k-NN Similarity Graph
   ↓
Spectral Clustering
   ↓
Flow-Based Refinement
   ├── MQI
   └── LFI
   ↓
Evaluation
```

## Methods

- Pandas / NumPy preprocessing
- One-hot encoding for categorical features
- PCA for the baseline representation
- K-Means baseline
- k-Nearest Neighbors similarity graph
- Spectral Clustering
- Flow-based cluster improvement using MQI and LFI

## Dataset

The project uses **NSL-KDD**, an intrusion-detection benchmark derived from KDD Cup '99. The dataset is not redistributed by this repository.

## Reported Experiment

The existing project reports the following F1 scores from its previous experiment:

| Method | F1 |
|---|---:|
| K-Means | 0.8806 |
| Spectral Clustering | 0.9252 |
| MQI | 0.9255 |
| LFI | 0.9416 |

These values are retained as **previously reported project results**; they are not presented as a newly reproduced experiment in the current repository state.

## Status

**Completed course/research prototype with room for reproducibility improvements.**

## Future Work

- Rebuild the experiment as a command-line pipeline.
- Add automated tests for graph construction and clustering.
- Add seed/configuration control.
- Separate train/evaluation data handling more clearly.
- Add precision, recall, confusion matrix, and class-wise metrics.
- Document computational cost and scalability.

## Technology

Python • NumPy • Pandas • Scikit-learn • NetworkX • Matplotlib • localgraphclustering

## Author

Mohammad Mahdi Shafighi — M.Sc. Artificial Intelligence
