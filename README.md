# Chest X-ray Classification: Normal vs Abnormal (Transfer Learning with DenseNet121)

A deep-learning pipeline that classifies chest X-rays as **Normal** or **Abnormal**
(Pneumonia + Lung Opacity). It started as a CNN trained from scratch and was rebuilt around
a pre-trained DenseNet121, with synthetic image generation to balance the classes,
explainability (Grad-CAM) and subgroup analysis.

> Research and portfolio project only. This is not a validated diagnostic tool.

## Dataset
- About 22,000 chest X-rays in two folders: `Normal/` and `Abnormal/`.
  `Abnormal/` contains two sub-folders, `Pneumonia/` and `Lung_Opacity/`.
- Labels: Normal = 0, Abnormal = 1 (Abnormal is the positive class, so recall = sensitivity).
- Dataset Link : https://www.kaggle.com/datasets/lyesaitk/clifford-3-class-dataset

## Pipeline

**1. Data split.** The split is done once on file paths, stratified by class: 70% train,
20% validation, 10% test. Each image belongs to exactly one split. (The first version split a
reshuffling `tf.data` pipeline, which lets images drift between splits.) The file listing is
recursive, so images in the Abnormal sub-folders are found automatically.

**2. Synthetic images for class balance.** The minority class (Normal, 7,134 vs 8,359 Abnormal
training images) is topped up with about 1,225 synthetic images. Each one starts from a random
real training image and gets a random combination of rotation (±10°), shift (±5%), zoom
(0.92–1.08), a smooth elastic deformation, gamma/contrast/brightness changes, light noise and
optional blur. Design rules:
- No horizontal flips (they would put the heart on the wrong side).
- Borders are filled by edge replication, not black, to avoid a "black corners = synthetic" cue.
- Synthetic images come only from training images and are added only to the training set, so
  validation and test stay 100% real and the metrics are not inflated.

**3. Input pipeline.** Images are decoded once, resized to 224×224, cached in RAM as uint8,
batched and prefetched. Normalisation lives inside the model, so training, evaluation and
inference cannot get out of sync.

**4. Model.** `augmentation (training only) -> backbone-specific preprocessing -> DenseNet121
(ImageNet weights) -> global average pooling -> dropout 0.3 -> Dense(1, sigmoid)`.
DenseNet121 was chosen because it is the architecture behind CheXNet, has a strong record on
chest X-rays and is parameter-efficient. The backbone is swappable (ResNet50V2, EfficientNetB0,
VGG16) through one config value.

**5. Two-phase training.**
- Phase 1: backbone frozen, only the new head trains (lr 1e-3, 5 epochs).
- Phase 2: the top 120 backbone layers are unfrozen and fine-tuned at lr 5e-6 with BatchNorm
  layers kept frozen. Callbacks: checkpoint on validation AUC, early stopping with best-weights
  restore, learning-rate reduction on plateau.

**6. Speed.** Training uses `MirroredStrategy` on 2 GPUs (global batch 64), mixed precision
(float16 compute, float32 output) and the RAM cache, so epochs after the first read from memory.

**7. Decision threshold.** The threshold is chosen on the **validation** set, where sensitivity
is about equal to specificity (0.471 in my run), then applied once to the test set. Choosing it on
validation keeps the test results honest.

**8. Evaluation and explainability.**
- Test metrics: accuracy, ROC-AUC, sensitivity, specificity, confusion matrix, classification report.
- **Per-subclass analysis:** accuracy for Normal, Pneumonia and Lung Opacity separately, with
  95% Wilson confidence intervals.
- **Grad-CAM:** heatmaps for correct predictions and for the missed cases, to check that the model
  looks at the lungs and not at text markers or devices.

## Results (test set, real images only; update with your final run)
| Metric | Value |
|---|---|
| Accuracy | 0.945 |
| ROC-AUC | 0.986 |
| Sensitivity / Specificity (threshold 0.471) | 0.944 / 0.943 |
| Pneumonia correctly flagged | 100% (622/622) |
| Lung Opacity correctly flagged | 88.3% (95% CI 85.4–90.7) |
| Normal correctly classified | 94.3% |

All missed Abnormal cases were Lung Opacity. Those images tend to show subtle findings.

## Limitations
- The split is by image, not by patient. If the dataset has several images per patient, results
  may be slightly optimistic.
- Perfect Pneumonia accuracy may partly reflect differences in image source or acquisition
  between classes. This has not been verified.
- The synthetic images are all Normal, so generation artifacts exist in only one class. The test
  set is real-only, but an ablation without synthetic images would confirm they help.
- Grad-CAM attention is sometimes outside the lung fields on missed cases.

## Reproducing
Kaggle notebook with a GPU (T4 ×2 recommended) and Internet on (to download ImageNet weights).
Set `DATA_DIR` and run all cells in order.
