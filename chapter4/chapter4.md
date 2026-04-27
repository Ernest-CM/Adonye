# CHAPTER FOUR: SYSTEM DESIGN AND IMPLEMENTATION

---

## 4.1 Introduction

This chapter presents the complete system design and implementation of an early Parkinson's Disease (PD) detection pipeline based on multimodal deep learning. Building on the methodology established in Chapter Three, this chapter translates theoretical design decisions into a concrete, reproducible software system capable of accepting speech, handwriting, and gait inputs and producing a binary Parkinson's Disease versus Healthy Control classification.

The implementation realises all five project objectives. Six model configurations are implemented and evaluated: three unimodal deep learning baselines (one per modality) and three multimodal fusion architectures — Early Fusion using feature-level concatenation, Late Fusion using decision-level probability averaging, and Hybrid Fusion using a cross-attention mechanism. All five deep learning models are implemented in TensorFlow/Keras. The Hybrid Fusion cross-attention module is implemented in PyTorch, with a bridging layer that converts intermediate embeddings computed by the TensorFlow models into PyTorch tensors.

The chapter is structured as follows. Section 4.2 describes the overall system architecture. Section 4.3 details the technology stack and justifies each selection. Section 4.4 specifies the datasets used and the preprocessing pipeline for each modality. Section 4.5 describes the architecture and training configuration of each unimodal model. Section 4.6 describes the three fusion architectures. Section 4.7 presents evaluation results. Section 4.8 summarises the chapter.

---

## 4.2 System Architecture

The system follows a four-tier layered pipeline architecture, chosen for its modularity, testability, and alignment with the linear data flow from raw modality input to final classification output. The four tiers are described below.

### Tier 1 — Data Ingestion and Preprocessing Layer

Three independent preprocessing sub-pipelines operate in parallel, one for each modality. Each sub-pipeline is responsible for loading raw data from its source, performing modality-specific feature extraction and normalisation, applying subject-wise stratified data splitting, and persisting processed arrays and split metadata to disk for reproducibility.

The speech sub-pipeline reads the pre-extracted 22-feature acoustic dataset in CSV format, normalises features using a StandardScaler fitted on training subjects only, and serialises the fitted scaler for use at inference time.

The handwriting sub-pipeline loads JPEG spiral drawing images, resizes each to 224×224 pixels, normalises pixel values to the range [0, 1], and applies data augmentation (horizontal flip, rotation of ±15 degrees, brightness variation of ±0.1) exclusively to the training partition to improve generalisation.

The gait sub-pipeline reads raw 16-channel vertical ground reaction force signals from the PhysioNet Gait-in-PD database, trims the first and last 20 seconds of each recording to exclude transitional artefacts, detects left and right heel-strike events using peak detection on the summed foot force channels, computes eight stride-level biomechanical features per detected stride, and pads or truncates the resulting stride sequences to a fixed length of 300 strides.

### Tier 2 — Unimodal Feature Learning Layer

Three independently trained deep learning models extract discriminative representations from each modality. Each model produces two outputs: a binary classification probability (used for unimodal evaluation and Late Fusion) and an intermediate embedding vector (used for Early Fusion and Hybrid Fusion). The models are trained, validated, and saved to disk before any fusion training begins, ensuring that unimodal performance is first established as a baseline.

### Tier 3 — Multimodal Fusion Layer

Three fusion strategies are implemented in parallel, each operating on the same trained unimodal models.

Early Fusion concatenates the frozen embedding vectors from all three unimodal models into a 320-dimensional joint representation, which is passed through a trainable classification head. The unimodal weights are frozen, so only the head parameters are updated during fusion training.

Late Fusion collects the prediction probabilities from the three trained unimodal models and combines them using a weighted average. Weights are not trained by gradient descent; instead, they are determined by grid search over the validation set to maximise F1-score.

Hybrid Fusion extracts intermediate embeddings from the three trained Keras models, transfers them to PyTorch tensors, and applies six pairwise cross-attention blocks — one for each directed pair of modalities — before aggregating the attended representations into a 768-dimensional vector for final classification.

### Tier 4 — Inference and Evaluation Layer

A unified evaluation module loads all six trained models and evaluates each on its respective held-out test partition. An inference latency benchmark measures wall-clock time per single-sample prediction over 100 trials after a warm-up pass, reporting mean, median, and 95th-percentile (p95) latency and comparing each result against the near-real-time acceptance threshold of p95 < 1000 ms.

A prototype inference pipeline (`PDInferencePipeline`) wraps all models behind a single interface that accepts any combination of modality inputs, selects the appropriate model, preprocesses inputs at inference time using the saved scalers, and returns a structured result containing the prediction, probability, PD risk label, model used, latency in milliseconds, and a timestamp.

---

## 4.3 Technology Stack

| Technology | Version | Role in System |
|---|---|---|
| Python | 3.10+ | Primary implementation language |
| TensorFlow / Keras | 2.12+ | Speech, handwriting, and gait unimodal models; Early Fusion and Late Fusion weight optimisation |
| PyTorch | 2.0+ | Hybrid Fusion cross-attention module |
| NumPy | 1.24+ | Array operations, embedding storage, stride feature computation |
| Pandas | 2.0+ | CSV dataset loading, results tabulation |
| Scikit-learn | 1.3+ | Stratified subject-wise splits, StandardScaler, class weight computation, evaluation metrics |
| SciPy | 1.11+ | `scipy.signal.find_peaks` for heel-strike detection in gait preprocessing |
| Pillow | 10.0+ | Image loading, resizing, and augmentation for handwriting modality |
| Matplotlib / Seaborn | 3.7+ / 0.12+ | Confusion matrix and ROC curve visualisation |
| Joblib | 1.3+ | Serialisation of fitted StandardScaler objects for inference-time reuse |
| Psutil | 5.9+ | Hardware profiling for the latency benchmark report |

**TensorFlow/Keras** was selected as the primary deep learning framework because it provides a high-level API that accelerates development of standard architectures (dense networks, convolutional networks, LSTM), offers built-in support for model checkpointing, early stopping, and learning rate scheduling, and enables exportable preprocessing pipelines.

**PyTorch** was selected specifically for the Hybrid Fusion cross-attention module because `torch.nn.MultiheadAttention` provides a well-tested, configurable implementation of the scaled dot-product attention mechanism with support for batch-first input and multiple attention heads, which would require manual implementation in Keras to achieve the same level of control.

The **TF→PyTorch bridging** pattern resolves the framework boundary cleanly: intermediate embeddings are computed once in TensorFlow, saved as NumPy arrays, and loaded as PyTorch tensors. This avoids repeated forward passes through the base models during PyTorch training iterations.

---

## 4.4 Dataset Specification and Preprocessing

### 4.4.1 Speech Dataset

**Source:** UCI Oxford Parkinson's Disease Detection Dataset (Little et al., 2007), available from the UCI Machine Learning Repository.

**Composition:** 195 voice recordings from 31 subjects (23 subjects diagnosed with Parkinson's Disease, 8 Healthy Controls). Each recording captures a sustained /a/ phonation. Multiple recordings per subject are present; the dataset is subject-identified by a name column.

**Features:** 22 pre-extracted acoustic features per recording. These include MDVP fundamental frequency statistics (Fo, Fhi, Flo), vocal jitter measures (local, absolute, RAP, PPQ5, DDP), vocal shimmer measures (local, dB, APQ3, APQ5, APQ11, DDA), noise-to-harmonics ratio (NHR), harmonics-to-noise ratio (HNR), and non-linear dynamic measures (RPDE, DFA, spread1, spread2, D2, PPE).

**Preprocessing:** Features are normalised using a StandardScaler fitted exclusively on training subjects. The fitted scaler is serialised to disk for use during inference. No feature engineering is applied; the pre-extracted acoustic features are used directly.

**Class imbalance:** The dataset contains 147 PD recordings and 48 HC recordings, an approximate ratio of 3:1. Class weighting (`class_weight='balanced'` in Keras) is applied during training to prevent the model from trivially predicting the majority class.

**Split:** Subject-wise stratified 70/15/15 split. All recordings from a given subject are assigned to a single partition. No subject's recordings appear in more than one partition. Subject-level leakage is verified programmatically before training.

### 4.4.2 Handwriting Dataset

**Source:** HandPD and/or NewHandPD dataset (Papa et al.), available from the Universidade Estadual Paulista.

**Composition:** JPEG images of spiral and/or meander drawing tasks performed by PD and Healthy Control subjects. Binary label: PD (class 1) or Healthy Control (class 0), determined by folder convention (PD/ and HC/ subfolders) or by filename prefix (P for PD, H for Healthy Control).

**Preprocessing:**
- Images are loaded using Pillow and resized to 224×224 pixels using Lanczos resampling.
- Pixel values are normalised from [0, 255] to [0.0, 1.0].
- Subject IDs are extracted from filenames using numeric pattern matching.
- Training partition only: augmentation is applied — horizontal flip (50% probability), random rotation in [−15°, +15°], and random brightness adjustment in [−0.1, +0.1]. Augmentation is deterministic (fixed random seed) for reproducibility.

**Split:** Subject-wise stratified 70/15/15. Augmented samples are generated from training images only; validation and test images are not augmented.

### 4.4.3 Gait Dataset

**Source:** PhysioNet Gait in Neurodegenerative Disease Database (Hausdorff et al., 2000), available from PhysioNet.

**Composition:** Vertical ground reaction force signals from subjects with neurodegenerative diseases and healthy controls. This implementation uses only PD subjects (files prefixed `pd`) and Control subjects (files prefixed `co`). 16 sensors per subject (8 left foot, 8 right foot) sampled at 100 Hz. Walking trials last approximately 2 minutes on level ground.

**Preprocessing:**
1. Raw signal files are parsed, skipping comment lines. The first column (elapsed time) is dropped; the 16 force channels are retained.
2. The first and last 20 seconds (2,000 samples at 100 Hz) are trimmed from each recording to exclude starting and stopping artefacts.
3. Left-foot and right-foot heel-strike events are identified by applying `scipy.signal.find_peaks` to the summed left and right foot force channels respectively, with a minimum inter-peak distance of 300 ms.
4. Eight stride-level features are computed per detected stride: left stride interval, right stride interval, left swing time, right swing time, left stance time, right stance time, left double-support time, and right double-support time.
5. Each subject's stride feature matrix is padded with zeros or truncated to a fixed 300-stride length.
6. Feature sequences are normalised using a StandardScaler fitted on training subjects. The fitted scaler is serialised for inference reuse.

**Split:** Subject-wise stratified 70/15/15. Each subject contributes a single stride-feature sequence of shape (300, 8).

### 4.4.4 Data Split Summary and Leakage Verification

All three modalities use a fixed random seed of 42 for reproducibility. Split ratios are defined globally in `src/config.py` and applied consistently across all modalities. Before training, a programmatic leakage check verifies that no subject identifier appears in more than one partition; a `ValueError` is raised and training is halted if any leakage is detected.

Note on cross-dataset evaluation: the three datasets contain independent subject cohorts. The UCI speech dataset, HandPD handwriting dataset, and PhysioNet gait dataset do not share subjects. Accordingly, each unimodal model is trained and evaluated on its own modality-specific subject pool. For fusion model evaluation, test samples are drawn independently from each modality's test partition and treated as a combined inference batch, which is consistent with the standard approach reported in multimodal PD detection literature when co-registered datasets are not available.

---

## 4.5 Unimodal Model Implementation

### 4.5.1 Speech Model

**Architecture:**

| Layer | Output Shape | Trainable Parameters |
|---|---|---|
| Input | (22,) | 0 |
| Dense(128, ReLU) + L2 reg | (128,) | 2,944 |
| BatchNormalisation | (128,) | 512 |
| Dropout(0.4) | (128,) | 0 |
| Dense(64, ReLU) + L2 reg | (64,) | 8,256 |
| BatchNormalisation | (64,) | 256 |
| Dropout(0.4) | (64,) | 0 |
| Dense(32, ReLU) **[embedding]** | (32,) | 2,080 |
| Dense(1, Sigmoid) | (1,) | 33 |
| **Total** | | **~14,081** |

**Training configuration:**

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1×10⁻³ |
| Loss | Binary cross-entropy |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping patience | 15 (monitors val_roc_auc) |
| Class weight | Balanced (computed from training set) |
| Regularisation | L2 (λ = 1×10⁻⁴), Dropout (0.4) |
| Checkpoint criterion | Best validation ROC-AUC |

The BatchNormalisation layers stabilise training on the small dataset. Class weight balancing is essential given the 3:1 PD-to-HC imbalance in the UCI dataset.

### 4.5.2 Handwriting Model

**Architecture:**

| Layer | Output Shape | Trainable Parameters (Phase 2) |
|---|---|---|
| Input | (224, 224, 3) | 0 |
| DenseNet121 base (top 50 layers trainable) | (7, 7, 1024) | ~250,000 |
| GlobalAveragePooling2D | (1024,) | 0 |
| Dense(256, ReLU) **[embedding]** | (256,) | 262,400 |
| Dropout(0.5) | (256,) | 0 |
| Dense(1, Sigmoid) | (1,) | 257 |
| **Total trainable (Phase 2)** | | **~513,000** |

**Two-phase training configuration:**

| Parameter | Phase 1 | Phase 2 |
|---|---|---|
| DenseNet121 base | Fully frozen | Top 50 layers trainable |
| Learning rate | 1×10⁻³ | 1×10⁻⁵ |
| Epochs | 10 | Up to 90 (early stopping) |
| Optimiser | Adam | Adam |
| Checkpoint criterion | Best val ROC-AUC | Best val ROC-AUC |

Phase 1 trains only the classification head, allowing it to adapt to the target task before any fine-tuning of pretrained convolutional weights. Phase 2 fine-tunes the top 50 DenseNet121 layers with a reduced learning rate to preserve the low-level ImageNet feature representations in the frozen base layers.

### 4.5.3 Gait Model

**Architecture:**

| Layer | Output Shape | Trainable Parameters |
|---|---|---|
| Input | (300, 8) | 0 |
| Conv1D(64, kernel=5, ReLU, same) | (300, 64) | 2,624 |
| MaxPooling1D(2) | (150, 64) | 0 |
| Conv1D(128, kernel=3, ReLU, same) | (150, 128) | 24,704 |
| MaxPooling1D(2) | (75, 128) | 0 |
| LSTM(64) | (64,) | 49,408 |
| Dropout(0.3) | (64,) | 0 |
| Dense(32, ReLU) **[embedding]** | (32,) | 2,080 |
| Dense(1, Sigmoid) | (1,) | 33 |
| **Total** | | **~78,849** |

**Training configuration:**

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1×10⁻³ |
| Loss | Binary cross-entropy |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping patience | 15 (monitors val_roc_auc) |
| Checkpoint criterion | Best validation ROC-AUC |

The Conv1D layers act as local pattern detectors on the stride sequence, capturing short-range temporal patterns in biomechanical features. The subsequent LSTM layer models longer-range sequential dependencies in the stride dynamics.

---

## 4.6 Fusion Architecture Implementation

### 4.6.1 Early Fusion

Early Fusion operates by concatenating the embedding vectors produced by the three frozen unimodal models into a single joint representation. The individual embedding dimensions are 32 (speech), 256 (handwriting), and 32 (gait), yielding a concatenated vector of dimension 320.

A trainable classification head is placed on top of this fused representation. The unimodal model weights are frozen throughout fusion training, ensuring that the pre-trained modality-specific feature extractors are not distorted. Only the head parameters are updated via backpropagation.

**Classification head architecture:**

| Layer | Output Shape |
|---|---|
| Concatenate [speech, hw, gait embeddings] | (320,) |
| Dense(256, ReLU) | (256,) |
| BatchNormalisation | (256,) |
| Dropout(0.4) | (256,) |
| Dense(128, ReLU) | (128,) |
| Dropout(0.3) | (128,) |
| Dense(1, Sigmoid) | (1,) |

This architecture allows the fusion head to learn cross-modal correlations from the concatenated feature space while preserving the integrity of the independently validated unimodal representations.

### 4.6.2 Late Fusion

Late Fusion combines the independent prediction probabilities produced by the three trained unimodal models using a weighted average. No gradient-based training is performed; the fusion weights are determined by exhaustive grid search over the validation set.

The grid search evaluates all combinations of weight values from the candidate set {0.0, 0.5, 1.0, 1.5, 2.0} for each of the three modalities (speech weight w_s, handwriting weight w_h, gait weight w_g), selects the combination maximising validation F1-score, and saves the resulting weights to disk for use during evaluation and inference.

The fused probability is computed as:

```
p_fused = (w_s × p_speech + w_h × p_handwriting + w_g × p_gait) / (w_s + w_h + w_g)
```

Late Fusion is computationally the lightest of the three fusion strategies: it requires no additional training beyond the three independently trained unimodal models. It is also robust to missing modalities — any single-modality input can be provided alone when the other sensors are unavailable at inference time.

### 4.6.3 Hybrid Fusion (Cross-Attention, PyTorch)

Hybrid Fusion models inter-modal dependencies using pairwise cross-attention. Given an intermediate embedding from modality A (the query) and an intermediate embedding from modality B (the context), a cross-attention block learns which aspects of modality A's representation are most relevant to modality B, and vice versa.

Formally, for query embedding **e_A ∈ ℝ^{d_A}** and context embedding **e_B ∈ ℝ^{d_B}**:

1. Linear projections map both to a common d_model-dimensional space:
   **Q = W_Q · e_A ∈ ℝ^{d_model}**,  **K = W_K · e_B ∈ ℝ^{d_model}**,  **V = W_V · e_B ∈ ℝ^{d_model}**

2. Scaled dot-product attention with h=4 heads:
   **Attended(e_A) = softmax(QKᵀ / √d_model) · V**

3. Layer normalisation is applied to the attended output.

Six directed cross-attention blocks are instantiated — one for each ordered pair of the three modalities:

| Block | Query modality | Context modality |
|---|---|---|
| s_from_hw | Speech (32-dim) | Handwriting (256-dim) |
| s_from_gait | Speech (32-dim) | Gait (32-dim) |
| hw_from_s | Handwriting (256-dim) | Speech (32-dim) |
| hw_from_gait | Handwriting (256-dim) | Gait (32-dim) |
| gait_from_s | Gait (32-dim) | Speech (32-dim) |
| gait_from_hw | Gait (32-dim) | Handwriting (256-dim) |

Each block produces a 128-dimensional attended output (d_model = 128). The six outputs are concatenated to form a 768-dimensional fused representation, which is passed to the classification head.

**Classification head (PyTorch):**

| Layer | Output Dimension |
|---|---|
| Linear(768 → 256) + ReLU | 256 |
| Dropout(0.4) + LayerNorm(256) | 256 |
| Linear(256 → 64) + ReLU | 64 |
| Dropout(0.3) | 64 |
| Linear(64 → 1) + Sigmoid | 1 |

**TF→PyTorch bridge:** The three trained Keras embedding models are used to pre-compute embedding arrays for all training and validation samples. These arrays are saved as NumPy `.npy` files and loaded as PyTorch tensors, allowing the cross-attention model to train on clean embedding inputs without re-executing the Keras forward passes on every epoch. At inference time, a single forward pass through each Keras embedding model followed by a single PyTorch forward pass is executed within the same function call.

**PyTorch training configuration:**

| Parameter | Value |
|---|---|
| Optimiser | Adam (weight_decay = 1×10⁻⁴) |
| Learning rate | 1×10⁻³ |
| Loss | BCELoss |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping patience | 15 (monitors val F1-score) |
| Checkpoint criterion | Best validation F1-score |

---

## 4.7 Evaluation Results

### 4.7.1 Model Performance Comparison

All six models are evaluated on their respective held-out test partitions using the fixed evaluation protocol defined in Chapter Three. The results are presented in Table 4.1.

**Table 4.1: Performance Comparison — All Six Models**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Speech Baseline | — | — | — | — | — |
| Handwriting Baseline | — | — | — | — | — |
| Gait Baseline | — | — | — | — | — |
| Early Fusion | — | — | — | — | — |
| Late Fusion | — | — | — | — | — |
| Hybrid Fusion | — | — | — | — | — |

*Table 4.1 is populated from `results/metrics_table.csv` after running `python -m src.evaluation.evaluate_all`.*

The evaluation metrics are computed using scikit-learn. Confusion matrices and ROC curves for each model are saved to `results/confusion_matrices/` and `results/roc_curves/` respectively. Statistical confidence intervals (mean ± standard deviation) are computed over validation folds where cross-validation is applied.

### 4.7.2 Inference Latency Benchmark

Inference latency is measured on the deployment target platform (CPU) using a standardised benchmarking procedure. Each model undergoes 10 warm-up inference calls followed by 100 timed calls. Wall-clock time is measured using `time.perf_counter()` from the point of model-ready input to prediction output, including inference-time preprocessing. Offline training time is excluded.

**Table 4.2: Inference Latency Benchmark**

| Model | Mean (ms) | Median (ms) | p95 (ms) | Status |
|---|---|---|---|---|
| Speech Baseline | — | — | — | — |
| Handwriting Baseline | — | — | — | — |
| Gait Baseline | — | — | — | — |
| Early Fusion | — | — | — | — |
| Late Fusion | — | — | — | — |
| Hybrid Fusion | — | — | — | — |

*Table 4.2 is populated from `results/latency_report.csv` after running `python -m src.evaluation.latency_benchmark`.*

**Acceptance threshold:** p95 latency < 1000 ms per sample (near-real-time screening criterion).

**Hardware profile:** Tables 4.1 and 4.2 are generated on a standard CPU deployment target. The hardware specification (CPU model, RAM, operating system, framework versions) is printed to the console and included in the generated latency report.

---

## 4.8 Summary

This chapter has described the complete design and implementation of a multimodal Parkinson's Disease detection system consisting of six model configurations across three modalities and three fusion strategies.

The five project objectives are addressed as follows:

| Objective | Addressed in |
|---|---|
| 1. Data collection and preprocessing with subject-wise 70/15/15 split | Section 4.4 |
| 2. Unimodal baselines + three fusion architectures (Early, Late, Hybrid) | Sections 4.5 and 4.6 |
| 3. Performance evaluation (Accuracy, Precision, Recall, F1, ROC-AUC) | Section 4.7.1 |
| 4. Inference latency benchmark (p95 < 1000 ms threshold) | Section 4.7.2 |
| 5. Prototype inference pipeline | Section 4.2 (Tier 4), `src/inference/pipeline.py` |

The system architecture is implemented across 20 Python source files organised into five sub-packages: `data`, `models`, `training`, `evaluation`, and `inference`. All configuration parameters are centralised in `src/config.py` with a fixed random seed of 42 for full reproducibility.

Chapter Five will present a comprehensive testing strategy, discuss the findings from the evaluation, compare results against related work in the literature, and provide conclusions and recommendations for future improvements to the system.

---

*Note: Metric values in Tables 4.1 and 4.2 marked with "—" are to be populated with actual experimental results after training all six models. Run the following commands in order:*

```
python -m src.training.train_speech
python -m src.training.train_handwriting
python -m src.training.train_gait
python -m src.training.train_late_fusion
python -m src.training.train_early_fusion
python -m src.training.train_hybrid_fusion
python -m src.evaluation.evaluate_all
python -m src.evaluation.latency_benchmark
```
