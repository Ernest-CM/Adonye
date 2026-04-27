# Revised Alignment Blueprint for Chapter 4 Readiness — Second Pass
### Revalidation Agent Score: 28/100 → Target: 100/100
### All FATAL, DANGER, and WARNING issues addressed in this revision.

---

## 1) Locked Project Scope (Authoritative — Unchanged)

**Title:** Early Detection of Parkinson's Disease Using Multimodal Analysis
**Author:** Mohammed Folajimi Joshua
**Final modality scope:** Speech + Handwriting + Gait only
**Fusion strategies in scope:** Early Fusion (feature-level concatenation), Late Fusion (decision-level probability averaging), Hybrid Fusion (cross-attention mechanism)
**Implementation frameworks:** TensorFlow/Keras and PyTorch
**Out-of-scope (permanently removed):** MRI, EEG, PET, SPECT, fMRI, neuroimaging pipelines, biological marker fusion, PPMI database

This scope is locked. Any sentence in any chapter that references neuroimaging modalities, neuroimaging databases, or neuroimaging-based fusion must be deleted or replaced. No exceptions.

---

## 2) Canonical Aim and Objectives — AUTHORITATIVE VERSION (Replaces All Prior Versions)

> **CRITICAL:** The revalidation agent confirmed that Chapter 1 and Chapter 3 currently contain two completely different, non-overlapping objective sets. The five objectives below are the single authoritative version. They must be copied verbatim into Chapter 1 AND into every objectives table or traceability section in Chapter 3. No paraphrasing. No subset. All five.

### Aim
Develop and evaluate an optimised multimodal fusion architecture for early detection of Parkinson's Disease using speech, handwriting, and gait data, with implementation as a near-real-time prototype inference pipeline.

### Objectives (Five — Measurable and Locked)

1. Collect and preprocess speech (mPower/UCI-compatible PD voice dataset), handwriting (HandPD/NewHandPD), and gait (PhysioNet gait-in-PD) datasets for binary Parkinson's Disease versus Healthy Control classification, applying subject-wise stratified splits (70% train / 15% validation / 15% test), and documenting final sample counts per class, inclusion and exclusion criteria, and missing-data handling rules for each modality.

2. Develop and train unimodal deep learning baseline models for each of the three modalities (speech, handwriting, and gait) and implement three multimodal fusion architectures: Early Fusion (feature-level concatenation of modality embeddings before a shared classifier), Late Fusion (decision-level averaging of independent modality prediction probabilities), and Hybrid Fusion (cross-attention mechanism applied to intermediate modality embeddings before final classification), using TensorFlow/Keras and PyTorch.

3. Evaluate and compare model performance across all unimodal baselines and all three multimodal fusion configurations using Accuracy, Precision, Recall, F1-score, ROC-AUC, and confusion matrix, and report statistical confidence as mean ± standard deviation across validation folds where applicable.

4. Benchmark inference latency of the complete multimodal pipeline under a defined CPU hardware profile by conducting at least 100 repeated inference trials per modality configuration and reporting mean, median, and p95 latency, with an acceptance threshold for near-real-time screening of p95 < 1000 ms per sample.

5. Implement and demonstrate a prototype inference pipeline that accepts speech, handwriting, and/or gait inputs, executes preprocessing and multimodal prediction, outputs a Parkinson's Disease risk classification, and logs per-run latency for feasibility validation.

---

## 3) Objective-to-Method-to-Deliverable Traceability (Synced with Objectives Above)

> **ACTION REQUIRED IN CHAPTER 3:** The traceability table in Chapter 3 must be replaced entirely with the table below. The old table used a different, incompatible objective set and must be deleted.

| Objective | Chapter 3 Method | Chapter 4 Deliverable |
|---|---|---|
| 1. Data collection and preprocessing | Final dataset selection (mPower/HandPD/PhysioNet), cleaning, normalisation, segmentation, feature extraction per modality, subject-wise stratified split | Reproducible data pipeline with documented input/output specs, sample counts, split ratios, and missing-data handling per modality |
| 2. Unimodal baselines + three fusion architectures | Train speech, handwriting, gait baselines; train Early Fusion, Late Fusion, and Hybrid Fusion (cross-attention) models | Baseline vs fusion experiment results for all five model configurations |
| 3. Performance evaluation with specific metrics | Fixed evaluation protocol using Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix; mean ± std reporting across folds | Tables and figures comparing all unimodal and multimodal configurations on the same held-out test partition |
| 4. Inference latency benchmarking | 100+ repeated inference trials per modality condition on defined hardware; mean, median, p95 computation | Latency results table with hardware specification and pass/fail against p95 < 1000 ms threshold |
| 5. Prototype pipeline implementation | End-to-end inference workflow integrating preprocessing, prediction, and latency logging | Working prototype demonstration with test cases and logged latency output |

---

## 4) Reproducible Technical Specification (Implementation Contract)

### 4.1 Dataset Specification (Final — PPMI Removed)

> **FATAL FIX:** Chapter 3 currently references PPMI (Parkinson's Progression Markers Initiative), which is a neuroimaging database and is explicitly out of scope. Every reference to PPMI in Chapter 3 must be deleted and replaced with the three approved datasets below.

| Modality | Approved Dataset | Inclusion Criteria | Target Label | Mandatory Reporting |
|---|---|---|---|---|
| Speech | mPower study dataset or UCI-compatible PD voice dataset (e.g., Oxford Parkinson's Disease Detection Dataset via UCI ML Repository) | Adult subjects with valid sustained vowel or connected speech recordings | PD vs Healthy Control | Final dataset name and version, sample count per class, recording protocol |
| Handwriting | HandPD or NewHandPD dataset | Samples with complete spiral, meander, or signature handwriting task signals or images | PD vs Healthy Control | Final task type selected, sample count per class, image/signal resolution |
| Gait | PhysioNet Gait in Neurodegenerative Disease Database or equivalent validated gait dataset | Subjects with complete gait cycle records at consistent sampling frequency | PD vs Healthy Control | Sampling rate, stride sequence length, harmonisation rule for varying lengths |

**Mandatory additions in Chapter 4 dataset section:**
- Exact dataset names and version or access date
- Final sample size: number of PD subjects and Healthy Controls per modality
- Inclusion and exclusion criteria written explicitly
- Missing-data handling rule (imputation strategy or exclusion threshold)
- Confirmation that PPMI and all neuroimaging databases are not used

### 4.2 Data Split and Validation Protocol (Mandatory — Must Appear in Chapter 3)

> **DANGER FIX:** Chapter 3 currently contains no data split definition, no subject-wise leakage prevention, and no validation protocol. This section must be written into Chapter 3 verbatim or in direct paraphrase.

- **Split type:** Subject-wise partition. A single subject's data must appear in only one of train, validation, or test. No sample from a subject in the training set may appear in validation or test. This prevents data leakage.
- **Default split ratio:** 70% training / 15% validation / 15% test, stratified by class label (PD vs Healthy Control) to maintain class balance across all three partitions.
- **Alternative (small datasets only):** If the dataset is too small for a stable holdout (fewer than 100 subjects per modality), use 5-fold stratified cross-validation on the training set, then evaluate once on a fixed held-out test set.
- **Random seeds:** A fixed random seed must be set and reported for all split operations to ensure reproducibility.
- **Leakage check:** Before training, confirm that subject IDs in the test set do not appear in the training or validation sets.

### 4.3 Model Configuration (Fixed Reporting Template)

For each model (speech baseline, handwriting baseline, gait baseline, Early Fusion model, Late Fusion model, Hybrid Fusion model), report:
- Input dimension and shape
- Layer structure in order (layer type, units or filters per layer)
- Activation functions per layer
- Loss function: binary cross-entropy
- Optimiser (Adam recommended) and learning rate
- Batch size and number of training epochs
- Regularisation strategy (dropout rate, L2 weight decay, or early stopping patience)
- Checkpoint selection rule: save the model checkpoint achieving the best validation F1-score or ROC-AUC

### 4.4 Fusion Design (Locked — Three Strategies, No Neuroimaging)

> **FATAL FIX:** Chapter 3 Section 3.5 must be rewritten. The current text describes feature-level fusion with MRI and EEG, which are out of scope. The replacement text is provided below and must replace the existing Section 3.5 content entirely.

**Replacement content for Chapter 3 Section 3.5:**

This study evaluates three fusion strategies for combining the speech, handwriting, and gait modalities. All fusion occurs exclusively across these three non-invasive modalities. No neuroimaging data is included at any fusion stage.

**Early Fusion (Feature-Level Concatenation):** Feature vectors extracted from the final feature extraction layer of each modality-specific sub-network are concatenated into a single unified feature vector. This concatenated representation is passed into a shared fully-connected classification head that produces the final binary PD versus Healthy Control prediction. Early fusion allows the classifier to learn joint cross-modal feature interactions from the outset.

**Late Fusion (Decision-Level Probability Averaging):** Each modality sub-network is trained independently and produces its own class probability output for PD versus Healthy Control. The final prediction is obtained by averaging (or weighted averaging) of the three modality-specific probability outputs. Late fusion preserves the independence of each modality model and is robust to missing modality inputs at inference time.

**Hybrid Fusion (Cross-Attention Mechanism):** Intermediate learned embeddings from each modality sub-network are passed through a cross-attention layer that models pairwise inter-modal relationships. The attention mechanism allows each modality's representation to attend to and be influenced by the representations of the other two modalities. The attended embeddings are aggregated and passed to a final classification layer. This strategy is implemented in PyTorch and follows the multi-head cross-modal attention paradigm.

All three fusion strategies are trained and evaluated on the same data splits using the same evaluation protocol defined in Section 3.4, enabling direct and fair comparison.

### 4.5 Evaluation Workflow

1. Preprocess all three modality datasets and confirm subject-wise split integrity.
2. Train and evaluate each unimodal baseline model on the same validation and test partition.
3. Train and evaluate all three multimodal fusion models on the same partitions.
4. Compare all five configurations (three unimodal, three fusion) using the fixed metric set.
5. Run inference latency benchmarking on the complete multimodal pipeline.
6. Report all results in structured tables with statistical confidence intervals.

---

## 5) Inference Latency and Real-Time Claim (Methodologically Supported)

### 5.1 Deployment Target
- Prototype platform: standard laptop or desktop CPU environment.
- Optional secondary profile: GPU, if available, reported separately.
- Runtime stack covers: preprocessing at inference time + model forward pass + decision output.

### 5.2 Latency Definition
- **Inference latency (ms per sample):** measured from model-ready preprocessed input to prediction output.
- Offline training time is excluded.
- Preprocessing steps required at inference (normalisation, feature extraction) are included.

### 5.3 Benchmark Protocol
- Minimum 100 repeated inference trials per modality configuration.
- Report: mean latency, median latency, p95 latency.
- Report hardware specification: CPU model, RAM, operating system, framework version.
- **Acceptance threshold for near-real-time screening: p95 < 1000 ms per sample.**

### 5.4 Prototype Deliverable
A runnable pipeline that accepts available speech, handwriting, and/or gait inputs, executes preprocessing, predicts PD risk class, logs latency per run, and outputs results in a readable format.

---

## 6) Chapter 3 Mandatory Corrections Summary

> This section consolidates every change required in Chapter 3 in one place for the revision pass.

| Location | Current Problem | Required Correction |
|---|---|---|
| Section 3.5 (Fusion Strategy) | Describes feature-level fusion with MRI and EEG | Delete entirely. Replace with the three-strategy fusion description provided in Section 4.4 of this document |
| Objectives Table / Traceability Section | Uses the original 5 old objectives, completely inconsistent with Chapter 1 | Delete entirely. Replace with the 5-objective traceability table in Section 3 of this document |
| Dataset Table | References PPMI (neuroimaging database) as a primary data source | Remove PPMI. Replace with mPower/UCI PD voice (speech), HandPD/NewHandPD (handwriting), PhysioNet gait-in-PD (gait) |
| Data Split Section (missing) | No subject-wise split, no leakage prevention, no validation protocol defined anywhere in Chapter 3 | Add the full data split and validation protocol from Section 4.2 of this document as a dedicated subsection in Chapter 3 |
| Any remaining MRI/EEG/neuroimaging references | Scope conflict with Chapter 1 | Search Chapter 3 for MRI, EEG, PET, SPECT, neuroimaging, PPMI and delete or replace every instance |

---

## 7) Chapter 1 Research Questions — Corrected Version (Section 1.4)

> **WARNING FIX:** Chapter 1 Section 1.4 currently asks about comparing early/late/hybrid fusion, which conflicts with the revised objectives if they were written without this comparison. The research questions below are aligned with the five canonical objectives and must replace the existing Section 1.4 questions.

**RQ1:** Can a multimodal fusion of speech, handwriting, and gait features achieve higher classification performance — measured by F1-score and ROC-AUC — than any individual single-modality baseline model for early Parkinson's Disease detection?

**RQ2:** Among the three fusion strategies evaluated — Early Fusion (feature-level concatenation), Late Fusion (decision-level probability averaging), and Hybrid Fusion (cross-attention mechanism) — which strategy achieves the highest discriminative performance for binary Parkinson's Disease versus Healthy Control classification?

**RQ3:** Can the complete multimodal inference pipeline meet a near-real-time screening requirement, defined as a p95 inference latency below 1000 milliseconds per sample, under a standard CPU deployment profile?

These three research questions map directly onto Objectives 3, 2, and 4 respectively, creating a coherent thread from problem statement through to evaluation.

---

## 8) Chapter 2 Mandatory Cleanup (Neuroimaging Removal + Synthesis Paragraph)

### 8.1 Neuroimaging Content Removal

> **DANGER FIX:** The revalidation agent confirmed 66 or more mentions of neuroimaging, MRI, EEG, PET, and SPECT in Chapter 2. None of these are implemented in this project. Every mention must be handled as follows:

- **Delete:** Any paragraph, sentence, or citation whose primary subject is neuroimaging modalities (MRI, fMRI, DaTscan, SPECT, PET, EEG) as detection inputs.
- **Reframe (where relevant):** If a sentence discusses neuroimaging only in the context of establishing why non-invasive modalities are preferred or more accessible, it may be retained if reworded to make that contrast explicit.
- **Retain:** Literature on speech biomarkers, handwriting biomarkers, and gait biomarkers for Parkinson's Disease. Literature on multimodal fusion methods for non-invasive PD detection. Literature on deep learning architectures used for audio, image, or time-series classification.
- **After cleanup:** Run a word search for MRI, EEG, PET, SPECT, fMRI, neuroimaging, DaTscan, PPMI in the Chapter 2 text. Every remaining instance must have a documented justification for why it was retained.

### 8.2 Synthesis Paragraph (Must Be Added to Chapter 2)

> **WARNING FIX:** A paragraph justifying the selection of the three modalities must be added to Chapter 2, preferably at the end of the literature review as a concluding synthesis before the chapter summary. The paragraph below must be included verbatim or as a close paraphrase:

---

Speech, handwriting, and gait have been selected as the three core modalities for this study because each captures a distinct and complementary dimension of Parkinson's Disease symptomatology that can be observed non-invasively in clinical or community settings. Speech signals reflect dysarthria, hypophonia, and vocal tremor arising from dopaminergic motor pathway dysfunction, with measurable acoustic features detectable in sustained phonation tasks. Handwriting captures fine motor degradation manifested as micrographia, increased pen pressure irregularity, reduced writing velocity, and tremor-induced stroke distortion. Gait encodes large-scale motor impairment including reduced stride length, increased gait asymmetry, decreased cadence, and episodic freezing. Crucially, these three modalities are non-invasive, do not require clinical infrastructure or specialist equipment, and each exhibits measurable discriminative signal in the early stages of Parkinson's Disease, before severe motor symptoms appear. Their integration in a multimodal fusion framework is therefore both clinically motivated and practically deployable, and the combination provides richer discriminative information than any single modality alone.

---

---

## 9) Proofreading — Mandatory Corrections for Chapter 1 Revised Objectives

> **WARNING FIX:** The following spelling errors were identified by the revalidation agent in the revised Chapter 1 objectives text. Each must be corrected before resubmission.

| Incorrect | Correct |
|---|---|
| parkisons disease | Parkinson's Disease |
| perfomance | performance |
| resposnse | response |
| statisticss | statistics |

In addition, perform a full proofread of Chapter 1, Chapter 2, and Chapter 3 before drafting Chapter 4. Pay particular attention to the revised objectives text, as it will be referenced repeatedly by assessors.

---

## 10) Final Consistency Checklist — Updated

Before drafting Chapter 4, every item below must be confirmed as complete:

- [ ] Chapter 1 scope statement confirms only speech, handwriting, and gait — no neuroimaging
- [ ] Chapter 1 contains exactly the five measurable objectives listed in Section 2 of this document, worded precisely, with fusion types, metric names, and latency threshold included
- [ ] Chapter 1 Section 1.4 research questions are replaced with the three corrected questions in Section 7 of this document
- [ ] Chapter 1 objectives text is fully proofread with all spelling errors corrected
- [ ] Chapter 2 neuroimaging mentions (MRI, EEG, PET, SPECT, fMRI, DaTscan, PPMI) are removed or reframed as documented
- [ ] Chapter 2 includes the three-modality synthesis paragraph from Section 8.2 of this document
- [ ] Chapter 3 Section 3.5 fusion strategy text has been deleted and replaced with the Early/Late/Hybrid Fusion descriptions in Section 4.4 of this document
- [ ] Chapter 3 objectives/traceability table has been deleted and replaced with the five-objective traceability table in Section 3 of this document
- [ ] Chapter 3 dataset table no longer references PPMI; replaced with mPower/HandPD/PhysioNet as specified in Section 4.1
- [ ] Chapter 3 contains a dedicated data split subsection defining subject-wise 70/15/15 split with leakage prevention, as specified in Section 4.2
- [ ] No mention of MRI, EEG, PET, SPECT, neuroimaging, or PPMI remains anywhere in Chapters 1, 2, or 3 without documented justification
- [ ] All five objectives appear identically in Chapter 1 and in the Chapter 3 traceability table
- [ ] Chapter 3 model configuration template specifies architecture details for all six model variants (three unimodal baselines + three fusion models)
- [ ] Fusion method descriptions are bounded to speech, handwriting, and gait inputs only
- [ ] Runtime target (p95 < 1000 ms) and latency benchmarking protocol are explicitly defined in Chapter 3
- [ ] Random seeds are documented for all data split and training operations

---

## 11) Expected Chapter 4 Outcome After This Revision

With all corrections applied, Chapter 4 can be written as a coherent and fully traceable implementation chapter containing:

- A reproducible multimodal Parkinson's Disease detection pipeline operating on speech, handwriting, and gait data
- Six model configurations: three unimodal baselines and three fusion variants (Early, Late, Hybrid)
- Comparative performance tables using Accuracy, Precision, Recall, F1-score, ROC-AUC, and confusion matrix across all configurations
- A latency benchmarking section with pass/fail result against the p95 < 1000 ms threshold
- A working prototype demonstration with logged latency output
- A coherent and unbroken line from: problem statement → aim → objectives → research questions → datasets → methodology → results
