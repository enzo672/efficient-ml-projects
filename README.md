# Drone Vision AI — EfficientNet & Mask R-CNN

This repository gathers the main machine learning workflows developed for **drone-based image analysis**.  
The objective is to automatically detect and evaluate **mud gauge heights** using deep learning techniques combining **Mask R-CNN** and **EfficientNet** architectures.

---

## 📂 Repository Structure

```
├── 1_maskrcnn_detection.ipynb
├── 2_dataset_preparation.ipynb
├── 3_gauge_height_estimation.ipynb
└── README.md
```

### `MASKRCNN.ipynb`
Trains and evaluates a **Mask R-CNN** model to isolate the **mud gauge** from raw drone images.  
This segmentation step allows extracting precise regions of interest that will later be used for supervised regression tasks.

### `Dataset_Intermediaire.ipynb`
Processes the outputs from the Mask R-CNN model to build an **intermediate dataset**.  
This dataset includes cropped and labeled gauge regions, ready to be used for training the EfficientNet model.

### `Evaluation_hauteur_de_boue.ipynb`
Trains an **EfficientNet-based regression model** to estimate the **gauge height** from the segmented images.  
This notebook includes preprocessing, training, and evaluation stages, providing insights into model performance and error distribution.

---

## 🧠 Technical Overview

- **Frameworks:** PyTorch, TensorFlow, OpenCV, scikit-learn  
- **Models:** Mask R-CNN (segmentation), EfficientNet (regression)  
- **Tasks:**
  - Object detection & segmentation (Mask R-CNN)
  - Dataset structuring and augmentation
  - Gauge height prediction (EfficientNet)
  - Model evaluation and visualization

---

## ⚙️ Environment Setup

To reproduce the experiments:

```bash
git clone https://github.com/yourusername/drone-vision-ai.git
cd drone-vision-ai
pip install -r requirements.txt
```

Then open the notebooks in order:

1. `1_maskrcnn_detection.ipynb`
2. `2_dataset_preparation.ipynb`
3. `3_gauge_height_estimation.ipynb`

---

## 📊 Results

The workflow enables a robust and modular pipeline for:
- Isolating physical indicators (mud gauges) in field drone footage  
- Training an interpretable model for automated height estimation  
- Preparing datasets reusable across different missions or environments  

---

## Future Work

- Fine-tuning EfficientNet with transfer learning on larger datasets  
- Exploring UNet-based segmentation as an alternative to Mask R-CNN  
- Deployment via an inference API (Flask / FastAPI) for real-time analysis  

---

## Author

**Enzo Kara**  
Master’s in Engineering – Télécom Physique Strasbourg  
Machine Learning & Computer Vision  
[LinkedIn](https://www.linkedin.com/in/enzo-kara-b15178263) | [GitHub](https://github.com/enzo672)
