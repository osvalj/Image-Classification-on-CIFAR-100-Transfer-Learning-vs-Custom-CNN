# Image Classification on CIFAR-100: Transfer Learning vs Custom CNN

Deep learning project comparing two approaches to image classification on CIFAR-100 — a dataset with 60,000 images across 100 categories, where the images are only 32×32 pixels and there are just 500 training examples per class.

Developed as part of a deep learning course (Master's in Data Science & AI — Nuclio Digital School) by Osval Hernández, Jorge Espina, Jon Mikel García Zurutuza, and Ronal Cuesta.

---

## The problem

CIFAR-100 is genuinely difficult. The images are small enough that a lot of visual information is simply not there, and with 500 examples per class the risk of overfitting is real. The goal wasn't just to get a good accuracy number — it was to understand what each approach offers and when you'd choose one over the other.

---

## Project structure

```
├── Entregable_DL_grupo4.ipynb   # Main notebook
├── tl_best.keras                # Saved Transfer Learning model (generated on run)
└── custom_best.keras            # Saved Custom CNN model (generated on run)
```

---

## Two methods

### Method 1 — Transfer Learning + Fine Tuning

We started with EfficientNetB0 and MobileNetV2, both pretrained on ImageNet, and compared them under identical conditions (same data, same epochs, same callbacks) with the backbone frozen. The winner moved to Fine Tuning; the loser was removed from memory.

**Fine Tuning**: we unfroze the last 30 layers of the winning backbone — enough to adapt to CIFAR-100's specific visual patterns without losing the low-level features (edges, textures) learned from ImageNet. We kept BatchNorm layers frozen during fine tuning to avoid destabilizing the normalization statistics.

Training setup: Adam optimizer, EarlyStopping on val_accuracy, ReduceLROnPlateau, ModelCheckpoint. Images upscaled from 32×32 to 96×96 because the pretrained models expect larger inputs.

### Method 2 — Custom ResNet-style CNN from scratch

A custom residual network designed specifically for 32×32 images — no upscaling needed. The architecture uses residual connections to allow training deeper networks without vanishing gradients.

Key design decisions:
- BatchNorm after every convolution, before activation — stabilizes training and allows higher learning rates
- Dropout after dense layers — prevents the network from depending on specific neurons
- L2 regularization on dense layers — penalizes large weights
- Swish activations throughout — empirically stronger than ReLU on image tasks
- SGD with Nesterov momentum + Cosine Decay learning rate schedule — works better than Adam here because it's more sensitive to the learning rate schedule
- Conservative data augmentation: horizontal flip, small crop, subtle brightness/contrast shifts — aggressive transforms would destroy information in such small images

---

## Results

| Metric | Transfer Learning (EfficientNetB0) | Custom CNN (ResNet-style) |
|--------|-------------------------------------|---------------------------|
| **Top-1 Accuracy (Test)** | 69.83% | ✅ 70.01% |
| **Top-5 Accuracy (Test)** | ✅ 92.45% | 91.15% |
| **Loss (Test)** | ✅ 1.0821 | ⚠️ 1.6471 |
| **Parameters** | 4.76M | ✅ 4.42M |

Both methods reached nearly identical Top-1 accuracy, but through very different paths. Transfer Learning converged faster and with lower, more calibrated loss — its probability estimates are more reliable. The custom CNN matched it on Top-1 but with a notably higher loss, which suggests it's less confident in its correct predictions.

**What the error analysis revealed:**  
- Both models struggle with fine-grained categories (distinguishing a wolf from a fox, or different vehicle subtypes)
- Transfer Learning has better Top-5 accuracy, meaning even when it's wrong, the correct answer tends to appear in its top 5 predictions
- Some errors are shared across both models — these likely reflect genuinely ambiguous images rather than model-specific weaknesses

---

## When to use each

**Transfer Learning** when you have limited data, limited compute, or need well-calibrated probability outputs. It converges faster and the loss is more meaningful.

**Custom CNN from scratch** when you have full control over architecture decisions, the input domain is very different from ImageNet (e.g. medical imaging, satellite imagery), or you specifically want to understand what the network is learning.

---

## Stack

- Python · TensorFlow 2.x · Keras
- EfficientNetB0 · MobileNetV2 (pretrained on ImageNet)
- NumPy · Matplotlib · Seaborn · Pandas

---

## How to run

1. Install dependencies: `pip install tensorflow numpy matplotlib seaborn pandas`
2. CIFAR-100 loads automatically via `keras.datasets.cifar100` — no manual download needed
3. Run the notebook top to bottom. GPU recommended; training times vary by hardware (the notebook was developed on Apple M4 with tensorflow-metal)
