# Flow-Based Graph Clustering for Network Intrusion Detection

> A reproducible research pipeline for **network intrusion detection (IDS)** using feature-space similarity graphs, clustering, and flow-based local refinement.

[GitHub Repository](https://github.com/9M3a1h3d9i9/FlowBased-Graph-Clustering-for-IDS)

## 1. Research Objective

The project asks a concrete question:

> **Can a graph representation of network-flow records improve unsupervised intrusion detection compared with conventional clustering?**

The current graph is a **feature-similarity graph**: each node is one network-flow record and edges connect records that are close in feature space. It is therefore not a communication-topology graph whose nodes are IP addresses.

This distinction is important for scientific reproducibility and for the later real-world formulation.

## 2. Current Pipeline

```text
NSL-KDD
   │
   ├── categorical encoding
   ├── binary Normal / Attack label
   ├── stratified sampling
   └── StandardScaler
          │
          ▼
        PCA
          │
          ▼
   Symmetric k-NN Graph
          │
          ├──────────────┐
          ▼              ▼
      K-Means       Spectral Clustering
          │              │
          └──────┬───────┘
                 ▼
        Unsupervised seed set
                 │
                 ▼
        True flow-based MQI
                 │
                 ▼
             IDS metrics
```

## 3. Phase 1 — Reproducibility and Correctness

The repository is being migrated from a notebook-centered prototype toward a modular, testable implementation.

Implemented modules:

- `src/config.py` — centralized experiment configuration and validation.
- `src/preprocessing.py` — explicit NSL-KDD loading, encoding, scaling and PCA utilities.
- `src/graph.py` — k-NN graph construction with explicit symmetrization and graph diagnostics.
- `src/clustering.py` — reusable K-Means and Spectral Clustering baselines.
- `src/evaluation.py` — Precision, Recall, F1, Specificity, FPR, Balanced Accuracy, ARI and NMI.
- `src/flow_refinement.py` — integration with the official `localgraphclustering` API for true MQI and SimpleLocal refinement.
- `tests/test_graph.py` — graph construction tests.
- `tests/test_flow_refinement.py` — validation and integration tests for flow refinement.
- `experiments/run_baseline.py` — reproducible end-to-end baseline runner.

### Graph-construction correction

The graph module explicitly constructs an undirected affinity matrix through either:

- `union`: `A = max(A, Aᵀ)`
- `mean`: `A = (A + Aᵀ) / 2`

It also reports connected components and degree statistics. The semantic definition is fixed: **node = network-flow record; edge = feature-space proximity**.

## 4. True Flow-Based Refinement

The previous notebook used NetworkX `minimum_cut` fallbacks when `localgraphclustering` was unavailable. Those fallbacks are **not equivalent** to the formal MQI/FI/LFI algorithms and must not be reported as such.

The new implementation removes that scientific ambiguity:

- `mqi_refine(...)` calls the published LocalGraphClustering **MQI** implementation.
- `simple_local_refine(...)` calls its strongly-local **SimpleLocal** method.
- `refine_from_labels(...)` derives seeds only from an unsupervised baseline clustering; ground-truth labels are not used for seed construction.
- The graph must be symmetric and seed indices are validated before execution.

The currently inspected `localgraphclustering` API exposes MQI and SimpleLocal through its public package interface. It does **not** expose functions named `FlowImprove` or `LocalFlowImprove`; therefore the project will not fabricate those APIs. If those algorithms are required later, they will be added only after locating and validating a genuine implementation.

## 5. Dataset

The project uses **NSL-KDD**, with the current prototype using binary labels:

- `Normal = 0`
- `Attack = 1`

The original experiment uses one-hot encoding for `protocol_type`, `service`, and `flag`, followed by standardization and PCA. The current working configuration is 15,000 stratified records, PCA to 50 dimensions and a symmetric k-NN graph with `k=15`.

This binary formulation is a controlled proof-of-concept, not the final research setting. Attack-family analysis (DoS, Probe, R2L and U2R) remains a planned extension.

## 6. Reproducibility Problem Being Investigated

Historical README values were:

| Method | Historical F1 |
|---|---:|
| K-Means | 0.8806 |
| Spectral | 0.9252 |
| MQI | 0.9255 |
| LFI | 0.9416 |

A later notebook run produced approximately:

| Method | Current notebook F1 |
|---|---:|
| K-Means | 0.8843 |
| Spectral | 0.5516 |
| MQI approximation | 0.6416 |
| FI approximation | 0.5462 |
| LFI approximation | 0.5462 |

These discrepancies are **not being hidden or averaged together**. Historical numbers remain unvalidated until the exact preprocessing, graph construction, algorithms and evaluation protocol are reproduced.

## 7. Reproducible Experiment Runner

After installing the pinned dependencies, run:

```bash
python -m experiments.run_baseline --data Data/KDDTrain+.txt
```

Optional controls:

```bash
python -m experiments.run_baseline \
  --data Data/KDDTrain+.txt \
  --output results/baseline_mqi.json \
  --sample-size 15000 \
  --pca 50 \
  --k 15
```

The runner saves configuration, preprocessing dimensions, PCA explained variance, graph statistics and evaluation metrics as JSON.

**Important:** MQI is a local refinement method, not automatically a generic two-way clustering algorithm. The current runner therefore reports MQI refinement from an unsupervised seed cluster separately rather than pretending it is an independent global partitioner.

## 8. Evaluation Protocol

The evaluation layer reports:

- Precision
- Recall
- F1
- Specificity
- False Positive Rate (FPR)
- Balanced Accuracy
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Silhouette score
- MQI conductance when applicable

Binary cluster-label orientation is aligned **only after prediction for offline evaluation**. Ground-truth labels must never be used to choose clustering seeds in the experiment pipeline.

Silhouette is currently computed in PCA Euclidean space, whereas flow refinement optimizes graph-local structure. These are intentionally different views of cluster quality and should not be conflated.

## 9. Research Roadmap

### Phase 1 — Reproducibility and correctness 🟡

- [x] Modularize preprocessing
- [x] Centralize parameters
- [x] Symmetrize k-NN graph explicitly
- [x] Add graph diagnostics
- [x] Add IDS metrics
- [x] Add unit tests
- [x] Add true MQI integration
- [ ] Reproduce the exact historical experiment

### Phase 2 — Systematic graph experiments

Evaluate:

- `k = 5, 10, 15, 20, 30`
- unweighted vs weighted k-NN
- union vs mean symmetrization
- mutual k-NN
- graph connectivity and degree distribution

Report both predictive and graph-quality metrics.

### Phase 3 — Real-world graph formulation

Move beyond a purely feature-similarity graph by incorporating network-flow structure such as:

- source/destination relationships
- ports and protocols
- temporal locality
- flow frequency
- bytes and packet statistics
- local neighborhood behavior

This creates a stronger bridge between graph clustering and operational IDS.

### Phase 4 — Innovation

Only after the baseline is validated should a novel method be introduced.

The strongest candidate direction is:

> **Adaptive Flow-Aware Graph Refinement for Network Intrusion Detection**

Candidate components:

1. adaptive edge weighting from feature similarity + traffic statistics + local graph structure;
2. uncertainty-aware refinement;
3. automatic selection of informative seed nodes;
4. adaptive local neighborhood size `k`;
5. optional reinforcement learning for sequential seed/action selection.

The innovation should be compared against the validated classical baseline rather than replacing it prematurely.

## 10. Real-World Relevance

The problem is operationally meaningful: an IDS must detect malicious traffic while controlling false positives, missed attacks and computational cost.

NSL-KDD is useful for controlled benchmarking but is insufficient for a production claim. Later validation should consider newer traffic datasets, temporal behavior, class imbalance, concept drift and scalability.

## 11. Project Structure

```text
FlowBased-Graph-Clustering-for-IDS/
├── Data/
├── notebooks/
│   └── Flow_b_Graph_Clust_for_IDS.ipynb
├── experiments/
│   └── run_baseline.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── graph.py
│   ├── clustering.py
│   ├── evaluation.py
│   └── flow_refinement.py
├── tests/
│   ├── test_graph.py
│   └── test_flow_refinement.py
├── results/
├── requirements.txt
└── README.md
```

## 12. Scientific Position

**Current status: a substantially stronger research prototype, but not yet a validated research benchmark.**

The order of work is deliberate:

**correctness → reproducibility → systematic graph experiments → real-world graph formulation → innovation**

Introducing a novel algorithm before resolving the MQI implementation and historical-result discrepancy would weaken the research claim.

## Author

Mohammad Mahdi Shafighi — M.Sc. Artificial Intelligence
