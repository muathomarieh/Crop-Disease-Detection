# 🌿 Crop Disease Detection — Deep Learning Image Classifier

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Model](https://img.shields.io/badge/Model-MobileNetV3--Small-4CAF50)
![Accuracy](https://img.shields.io/badge/Val%20Accuracy-96.3%25-brightgreen)

A convolutional neural network that identifies **39 classes of plant leaf diseases and healthy leaves** from a single photo.
Built with **transfer learning on MobileNetV3-Small**, a lightweight architecture chosen so the model can later run **on-device inside a mobile app** (Core ML / TensorFlow Lite).

## 📊 Results

| Metric | Value |
|---|---|
| Dataset | PlantVillage (without augmentation) — **55,446 images, 39 classes** |
| Validation accuracy | **96.3%** |
| Training epochs | 4 (early stopping, best checkpoint restored) |
| Input size | 256 × 256 RGB |
| Hardware | Apple Silicon (M4) GPU via `tensorflow-metal` |

Validation accuracy per epoch: **90.4% → 94.3% → 95.8% → 96.3%**

## 🧠 Approach

1. **Data pipeline** — `image_dataset_from_directory` with a seeded **80/20 train/validation split** (~44k / ~11k images) + `tf.data` `cache()`, `shuffle()` and `prefetch(AUTOTUNE)` for fast GPU feeding.
2. **Transfer learning (feature extraction)** — frozen ImageNet-pretrained **MobileNetV3-Small** backbone (global average pooling) + trainable `Dense(256, ReLU) → Dropout(0.2) → Dense(39)` head.
3. **Training** — Adam (lr = 1e-4), categorical cross-entropy, `ModelCheckpoint` on best `val_accuracy`, `EarlyStopping(patience=5)`.
4. **Inference** — load the saved `.keras` model and classify any leaf image.

## 📁 Project Structure

```
CropDisease/
├── crop_disease_detection.ipynb   # data pipeline, training, evaluation, inference
├── crop_disease_detection.keras   # trained model
└── best_model.h5                  # best checkpoint
```

## 🚀 Run it

```bash
pip install tensorflow matplotlib numpy
# macOS / Apple Silicon GPU:
pip install tensorflow-metal
jupyter notebook CropDisease/crop_disease_detection.ipynb
```

Quick prediction:

```python
import numpy as np, tensorflow as tf
model = tf.keras.models.load_model("CropDisease/crop_disease_detection.keras")
img = tf.keras.utils.load_img("leaf.jpg", target_size=(256, 256))
pred = model.predict(np.expand_dims(tf.keras.utils.img_to_array(img), 0))
print(classes[int(np.argmax(pred))])
```

## 🔭 Next Steps

- Convert to **Core ML** and ship inside an iOS app (camera → diagnosis on-device).
- Add a held-out test set, confusion matrix and per-class precision/recall.
- Plug in the prepared augmentation block (flip / rotation / zoom) and fine-tune upper backbone layers for extra accuracy.

---
👤 **Muath Omarieh** — CS student @ University of Jordan · [GitHub](https://github.com/muathomarieh) · [LinkedIn](https://www.linkedin.com/in/muath-omarieh-550679297/)
