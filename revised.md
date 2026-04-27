# Revised Alignment Blueprint for Chapter 4 Readiness

## 1) Locked Project Scope (Authoritative)

**Title:** Early Detection of Parkinson's Disease Using Multimodal Analysis  
**Final modality scope:** **Speech + Handwriting + Gait only**  
**Out-of-scope (removed):** MRI, EEG, neuroimaging pipelines, biological marker fusion

This revision fixes the scope conflict by keeping the system strictly aligned with Chapters 1 and the original proposed system boundary.

---

## 2) Unified Aim and Objectives (Frozen)

### Aim
Develop and evaluate an optimized multimodal fusion architecture for early detection of Parkinson’s Disease using speech, handwriting, and gait data.

### Objectives
1. Collect and preprocess speech, handwriting, and gait datasets suitable for binary PD classification (PD vs Healthy Control).  
2. Build unimodal baseline models for each modality and a multimodal fusion model.  
3. Evaluate model performance using Accuracy, Precision, Recall, F1-score, ROC-AUC, and confusion matrix.  
4. Evaluate runtime performance (inference latency) under a defined deployment profile.  
5. Implement a prototype inference pipeline that performs near-real-time early screening from available modality inputs.

This removes objective drift by preserving both model-quality and runtime/prototype goals in one consistent objective set.

---

## 3) Objective-to-Method-to-Deliverable Traceability

| Objective | Chapter 3 Method | Chapter 4 Deliverable |
|---|---|---|
| 1. Data collection & preprocessing | Final dataset selection, cleaning, normalization, segmentation, feature extraction per modality | Reproducible data pipeline with documented input/output for each modality |
| 2. Build unimodal + multimodal models | Train speech, handwriting, gait baselines; train fusion architecture | Baseline vs fusion experiment results |
| 3. Accuracy-focused evaluation | Fixed evaluation protocol and metric computation | Tables/figures for metrics and comparative performance |
| 4. Runtime evaluation | Latency benchmarking protocol with fixed hardware/profile | Response-time results and compliance against target threshold |
| 5. Prototype implementation | End-to-end inference workflow and integration logic | Working prototype demo flow with test cases |

---

## 4) Reproducible Technical Specification (Implementation Contract)

## 4.1 Dataset Specification (Final)

| Modality | Dataset Source | Inclusion | Target Label | Notes |
|---|---|---|---|---|
| Speech | mPower / UCI-compatible PD voice dataset | Adult subjects with valid voice recordings | PD vs Healthy | Use one finalized source or harmonized subset; document final sample counts |
| Handwriting | HandPD / NewHandPD / UCI handwriting equivalent | Samples with complete handwriting signals/images | PD vs Healthy | Use consistent task type (spiral/meander/signature) |
| Gait | PhysioNet gait-in-PD or equivalent validated gait dataset | Subjects with complete gait cycle records | PD vs Healthy | Harmonize sampling units and sequence lengths |

**Mandatory additions in Chapter 4 tables:**  
- Final selected dataset names (exact versions)  
- Final sample size per class (PD, Healthy) per modality  
- Inclusion/exclusion criteria  
- Missing-data handling rule

## 4.2 Data Split and Validation Protocol

- Subject-wise split to avoid leakage across train/validation/test.  
- Recommended fixed split: **70% train / 15% validation / 15% test** (stratified by class).  
- If sample size is small, use **5-fold stratified cross-validation** on training set, then final holdout test once.  
- Set and report random seeds for reproducibility.

## 4.3 Model Configuration (Fixed Reporting Template)

For each model (speech baseline, handwriting baseline, gait baseline, fusion model), report:
- Input dimension/shape  
- Layer structure (type and order)  
- Hidden dimensions  
- Activation functions  
- Loss function (binary cross-entropy)  
- Optimizer (e.g., Adam) and learning rate  
- Batch size  
- Number of epochs  
- Regularization (dropout/L2/early stopping)  
- Checkpoint selection rule (best validation F1 or AUC)

## 4.4 Fusion Design (Locked)

- Baseline fusion method: **Late Fusion (decision-level)** combining modality-specific prediction probabilities.  
- Optional secondary experiment: **Intermediate fusion** (concatenate learned embeddings before final classifier).  
- No MRI/EEG/imaging inputs allowed in fusion experiments.

## 4.5 Evaluation Workflow

1. Train unimodal models.  
2. Evaluate unimodal models on the same test partition.  
3. Train/evaluate multimodal fusion model.  
4. Compare unimodal vs multimodal results using fixed metrics.  
5. Report statistical confidence (e.g., mean ± std across folds where applicable).

---

## 5) Real-Time Claim Operationalization (Now Methodologically Supported)

## 5.1 Deployment Target

- Prototype platform: laptop/desktop CPU environment (and optional GPU profile if available).  
- Runtime stack: preprocessing + model inference + decision output pipeline.

## 5.2 Latency Definition

- **Inference latency (ms/sample)** measured from model-ready input to prediction output.  
- Exclude offline training time; include preprocessing needed at inference stage.

## 5.3 Benchmark Protocol

- Run at least 100 repeated inference trials per modality condition.  
- Report mean, median, p95 latency.  
- Report hardware details (CPU/GPU/RAM).  
- Define acceptance threshold for near-real-time screening (example: p95 < 1000 ms per sample).

## 5.4 Prototype Deliverable

- A runnable pipeline that accepts available speech/handwriting/gait inputs, executes preprocessing, predicts PD risk class, and logs latency per run.

---

## 6) Chapter 2 Refinement Guidance (For Coherence with Chapter 4)

1. Retain literature directly supporting speech, handwriting, gait biomarkers and multimodal fusion in PD detection.  
2. Move/remove neuroimaging-heavy discussions that are not implemented in this project scope.  
3. Add a short synthesis paragraph explicitly justifying why these three modalities are complementary for early detection.  
4. Proofread and remove garbled/repeated text artifacts before final submission.

---

## 7) Final Consistency Checklist Before Drafting Chapter 4

- [ ] Scope states only speech + handwriting + gait in all chapters  
- [ ] Objectives are identical across Chapters 1 and 3  
- [ ] Dataset table includes exact sources and class counts  
- [ ] Split/validation protocol is fixed and leakage-safe  
- [ ] Model architectures and hyperparameters are fully specified  
- [ ] Fusion method(s) are clearly bounded and reproducible  
- [ ] Runtime target and latency benchmarking method are explicitly defined  
- [ ] Chapter 2 literature and wording are cleaned and aligned to implementation scope

---

## 8) Expected Chapter 4 Outcome After This Revision

With this revision, Chapter 4 can be written as a clear implementation chapter containing:
- A reproducible multimodal PD detection pipeline (speech, handwriting, gait),
- Comparative unimodal vs multimodal performance evidence,
- Explicit runtime/latency validation for prototype feasibility,
- A coherent, traceable line from problem → aim → objectives → method → results.
