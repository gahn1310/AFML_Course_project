# **Neuro-Symbolic Atrial Fibrillation Detection using CRNN \+ LNN with Uncertainty Quantification**

This project implements a **multimodal neuro-symbolic pipeline** for detecting **Atrial Fibrillation (AF)** from **single-lead ECG signals**, combining:

* **CRNN (Convolutional Recurrent Neural Network)** for waveform feature extraction  
* **LNN (Linear Neural Network)** for rhythm/tabular features  
* **Uncertainty Quantification (MC Dropout)**  
* **Hierarchical fusion-based classifier \+ SVM refinement**  
* **Explainability-friendly design suited for clinical interpretation**

This work is part of the **Advanced Foundations for Machine Learning (AFML)** course project.

---

## **📌 Features**

* End-to-end ECG processing pipeline  
* Band-pass filtering (0.5–40 Hz)  
* Pan–Tompkins R-peak detection  
* RR-interval \+ HRV statistics extraction  
* CRNN model for morphological features  
* LNN model for tabular features  
* Fusion architecture for multimodal prediction  
* Hierarchical classification (AF vs All → N vs Others → O vs Noisy)  
* SVM post-processing for improved separation  
* Model uncertainty estimation (MC Dropout)  
* Visualizations for ECG, peaks, windows, predictions

---

## **📁 Project Structure**

`├── AF_CODE.ipynb            # Main Jupyter Notebook (full pipeline)`  
`├── AFML_CourseProject.pdf   # PPT summarizing approach & results`  
`├── Research_Paper.pdf       # IEEE-style report`  
`├── README.md                # Project documentation`  
`└── data/                    # PhysioNet data (user must download)`

---

## **📊 Dataset**

**Source:** PhysioNet CinC Challenge 2017  
**Samples:** 8,528 ECG recordings  
**Duration:** 30–60 seconds  
**Sampling Rate:** 300 Hz  
**Classes:**

* N – Normal  
* AF – Atrial Fibrillation  
* O – Other rhythm  
* \~ – Noisy

Link: [https://physionet.org/challenge/2017/](https://physionet.org/challenge/2017/)

---

## **🧠 Model Architecture**

### **1️⃣ CRNN Branch**

* Conv1D → MaxPool (×3)  
* LSTM layers (×2)  
* Dense representation (64 units)  
* Learns morphological and rhythmic ECG patterns.

### **2️⃣ LNN Branch**

* Dense(64 → 32\)  
* Dropout(0.2)  
* Learns:  
  * RR statistics  
  * HRV metrics  
  * Noise indices  
  * Signal quality features

### **3️⃣ Fusion Module**

`CRNN embedding  ─┐`  
                 `├─> Dense(32) → Softmax`  
`LNN embedding   ─┘`

### **4️⃣ Hierarchical Classification**

1. AF vs All  
2. N vs (O \+ Noisy)  
3. O vs Noisy

### **5️⃣ SVM Refinement**

Adds decision-boundary sharpness for difficult classes.

---

## **🧪 Experimental Results (Summary)**

| Task | Model | F1-Score | Balanced Accuracy |
| ----- | ----- | ----- | ----- |
| AF vs All | CRNN | 0.856 | 0.83 |
| AF vs All | CRNN+LNN | 0.872 | 0.88 |
| N vs Others | Fusion+SVM | 0.888 | 0.89 |
| O vs Noisy | Fusion+SVM | 0.849 | 0.88 |

The multimodal \+ SVM approach outperforms all baselines.

---

## **🛠 How to Run**

### **1\. Install Dependencies**

`pip install numpy scipy matplotlib sklearn tensorflow wfdb`

### **2\. Download Dataset**

Download PhysioNet 2017 dataset locally and place inside:

`/data/training2017/`

### **3\. Open Notebook**

Run the pipeline:

`AF_CODE.ipynb`

The notebook contains:

* Preprocessing  
* Feature extraction  
* Model training  
* Evaluation  
* Plots

---

## **📌 Key Learnings**

* Integrating neural \+ symbolic reasoning improves interpretability.  
* RR-interval and HRV features significantly enhance AF detection.  
* Hierarchical classification reduces confusion among “Other” vs “Noisy”.  
* MC dropout provides uncertainty estimates essential for clinical trust.

---

## **⚠️ Current Limitations**

* Handling extremely noisy signals remains difficult.  
* "Other" class has high intra-class variability.  
* Real-time hospital deployment would require additional calibration.

---

## **🚀 Future Work**

* Replace LSTM with Transformers for longer temporal modeling.  
* Add SHAP/LIME explanations for clinical interpretability.  
* Train on multi-lead ECG datasets.  
* Real-time streaming ECG inference.

---

## **👩‍💻 Authors**

* Surabhi M – PES1UG23AM325  
* Smrithi A S – PES1UG23AM306  
* Gahnavi B – PES1UG23AM900  
* Tanuu Shree M – PES1UG23AM336

