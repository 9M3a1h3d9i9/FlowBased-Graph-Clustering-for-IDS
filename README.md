# Flow-Based Graph Clustering for Network Intrusion Detection

> A reproducible research pipeline for **network intrusion detection (IDS)** using feature-space similarity graphs, clustering, and flow-based local graph refinement.

urlGitHub Repositoryhttps://github.com/9M3a1h3d9i9/FlowBased-Graph-Clustering-for-IDS

## 1. Research Objective

The project asks a concrete question:

> **Can a graph representation of network-flow records improve unsupervised intrusion detection compared with conventional clustering?**

The current graph is a **feature-similarity graph**: each node is one network-flow record and edges connect records that are close in feature space. It is therefore not a communication-topology graph whose nodes are IP addresses.

This distinction is important for both scientific reproducibility and future real-world deployment.

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
          │              ▼
          │       Flow-based refinement
          │        (next implementation stage)
          │
          └──────────────┬──────────────┘
                         ▼
                  IDS Evaluation
```

## 3. What Was Improved in Phase 1

The repository is being migrated from a notebook-centered prototype toward a research-grade, modular implementation.

### Added source modules

- `src/config.py` — centralized experiment configuration and validation.
- `src/preprocessing.py` — explicit NSL-KDD loading, encoding, scaling and PCA utilities.
- `src/graph.py` — k-NN graph construction with explicit symmetrization and graph diagnostics.
- `src/clustering.py` — reusable K-Means and Spectral Clustering baselines.
- `src/evaluation.py` — Precision, Recall, F1, Specificity, FPR, Balanced Accuracy, ARI and NMI.
- `tests/test_graph.py` — initial tests for graph symmetry and graph statistics.

### Graph-construction correction

The original workflow allowed scikit-learn to warn that the affinity matrix was asymmetric. The new graph module makes the intended undirected graph explicit through either:

- `union`: `A = max(A, Aᵀ)`
- `mean`: `A = (A + Aᵀ) / 2`

It also reports the number of connected components and degree statistics.

## 4. Dataset

The project uses **NSL-KDD**, with the current prototype using binary labels:

- `Normal = 0`
- `Attack = 1`

The original experiment used one-hot encoding for `protocol_type`, `service`, and `flag`, followed by standardization and PCA.

The repository should not treat the current binary formulation as the final IDS research setting. Attack-family analysis (DoS, Probe, R2L and U2R) is a planned extension.

## 5. Reproducibility Status

The previous README reported:

| Method | Previously reported F1 |
|---|---:|
| K-Means | 0.8806 |
| Spectral | 0.9252 |
| MQI | 0.9255 |
| LFI | 0.9416 |

These numbers remain historical results. They are **not yet accepted as reproduced research results**.

The notebook's more recent run produced substantially different values, including approximately:

| Method | Current notebook F1 |
|---|---:|
| K-Means | 0.8843 |
| Spectral | 0.5516 |
| MQI approximation | 0.6416 |
| FI approximation | 0.5462 |
| LFI approximation | 0.5462 |

The discrepancy is now treated as a research reproducibility issue rather than hidden. Possible causes include graph construction, graph symmetry, preprocessing, sampling, hyperparameters, and whether the earlier experiment used the actual LocalGraphClustering algorithms.

## 6. Critical Scientific Note: MQI / FI / LFI

The current notebook contains fallback approximations based on NetworkX minimum-cut operations. These are **not equivalent to the formal MQI, FlowImprove and LocalFlowImprove algorithms**.

Therefore, until the `localgraphclustering` implementation is successfully executed and validated:

> **The project must not claim that the current approximation results are true MQI/FI/LFI results.**

This is the highest-priority technical validation task in the next phase.

## 7. Research Roadmap

### Phase 1 — Reproducibility and correctness ✅ in progress

- [x] Separate preprocessing from notebook code
- [x] Centralize experiment parameters
- [x] Make k-NN graph symmetry explicit
- [x] Add graph diagnostics
- [x] Add IDS-oriented evaluation metrics
- [x] Add initial unit tests
- [ ] Reproduce the exact historical experiment

### Phase 2 — True flow-based algorithms

- [ ] Validate `localgraphclustering==0.6.1` in a clean environment
- [ ] Replace minimum-cut approximations with the actual MQI / FlowImprove / LocalFlowImprove APIs
- [ ] Record algorithm parameters and seeds
- [ ] Re-run all baselines
- [ ] Explain the 0.925–0.941 historical results versus the current 0.55–0.64 results

### Phase 3 — Systematic graph experiments

Evaluate:

- k = 5, 10, 15, 20, 30
- unweighted k-NN
- weighted k-NN
- mutual k-NN
- different graph symmetrization rules
- connected-component structure

Report F1, Precision, Recall, Specificity, FPR, Balanced Accuracy, ARI, NMI and Silhouette.

### Phase 4 — Real-world-oriented improvement

Move beyond a purely feature-similarity graph by incorporating network-flow structure such as:

- source/destination relationships
- ports and protocols
- temporal locality
- flow frequency
- bytes and packet statistics
- local neighborhood behavior

This creates a stronger bridge between the academic graph-clustering formulation and operational IDS.

### Phase 5 — Innovation

Only after the baseline is correct should we introduce a novel method.

The strongest candidate direction is:

> **Adaptive Flow-Aware Graph Refinement for Network Intrusion Detection**

Potential innovation components:

1. adaptive edge weighting using feature similarity + flow statistics + local graph structure;
2. confidence/uncertainty-aware cluster refinement;
3. automatic selection of the most informative local seed set;
4. adaptive choice of k based on local density;
5. optional reinforcement learning for selecting the next refinement action.

The innovation should be evaluated against the validated classical baselines rather than replacing them prematurely.

## 8. Real-World Relevance

This project is close to a genuine operational problem: IDS systems must identify malicious traffic while controlling false positives and computational cost.

The current NSL-KDD formulation is useful for controlled experiments, but it is not sufficient by itself to claim production readiness. A realistic extension should evaluate newer traffic data, temporal behavior, class imbalance, concept drift and computational scalability.

## 9. Project Structure

```text
FlowBased-Graph-Clustering-for-IDS/
├── Data/
├── notebooks/
│   └── Flow_b_Graph_Clust_for_IDS.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── graph.py
│   ├── clustering.py
│   └── evaluation.py
├── tests/
│   └── test_graph.py
├── results/
├── requirements.txt
└── README.md
```

## 10. Technology

Python · NumPy · Pandas · SciPy · Scikit-learn · NetworkX · LocalGraphClustering · Matplotlib · Seaborn · Pytest

## 11. Scientific Position

**Current status: strong course project / promising research prototype, but not yet a validated research benchmark.**

The priority is deliberately **correctness → reproducibility → systematic experiments → real-world graph formulation → innovation**.

Adding a novel algorithm before resolving the current MQI/FI/LFI and historical-result discrepancy would make the research claim weaker, not stronger.

## Author

Mohammad Mahdi Shafighi — M.Sc. Artificial Intelligence
