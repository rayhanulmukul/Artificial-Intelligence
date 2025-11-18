# CSE4132 – Artificial Intelligence Laboratory

This repository hosts the full practical workflow for **CSE4132 – Artificial Intelligence Lab**, covering neural network pipelines, transfer learning, numerical reasoning with neural models, and heuristic search strategies.

All experiments adhere to the official departmental lab specification and are structured for reproducible execution, academic assessment, and portfolio deployment.

---

## Repository Structure

```
Artificial-Intelligence-Lab/
│
├── 01_FCNN_Image_Classification/
├── 02_CNN_Image_Classification/
├── 03_MobileNet_Backbone_CNN/
├── 04_TransferLearning_vs_FineTuning/
├── 05_Root_Finding_NeuralNetwork/
├── 06_Heuristic_Search/
├── 07_A_Star_Search/
├── 08_BestFirst_Search/
│
└── Question.pdf
├── ProblemSet_A/   # FCNN vs CNN, Fashion dataset, MobileNet, Figure-1 & Figure-2 architectures, 2-stage training
└── ProblemSet_B/   # Polynomial DNN (32→64→128), normalization, train/val/test split, accuracy & loss curves
```

---

## Lab Portfolio

### Additional Problem Set

**Problem Set A**

* FCNN and CNN on Fashion dataset
* MobileNet backbone comparison
* Architecture replication from Figure‑1 and Figure‑2
* Two‑stage training with layer freezing, checkpointing, custom image dataset

**Problem Set B**

* DNN for solving cubic polynomial
* Architecture: 32 → 64 → 128 hidden neurons
* Data normalization to [-1,1]
* Train/val/test split: 90% / 5% / 5%
* Accuracy & error curve visualization

### **01 – Fully Connected Neural Network (FCNN) Classifier**

* Multi-layer perceptron for 10-class image recognition

### **02 – Convolutional Neural Network (CNN) Classifier**

* End-to-end CNN pipeline for the same dataset

### **03 – CNN with MobileNet Backbone**

* Transfer learning classifier using pretrained MobileNet

### **04 – Transfer Learning vs Fine-Tuning**

* Dataset: CIFAR‑10
* Comparative accuracy & loss profiling

### **05 – Neural Network Based Root Finding**

* Polynomial root approximation via regression-based neural inference

### **06 – Heuristic Search Algorithm**

* Implementation & performance demonstration

### **07 – A* Search Algorithm**

* Optimal pathfinding + cost-based evaluation

### **08 – Best‑First Search Algorithm**

* Greedy state‑space exploration

---

## Technology Stack

Python, TensorFlow / PyTorch, NumPy, Matplotlib, Scikit‑Learn

---

## Execution

```
cd 0X_Module
python main.py
```

---

## Academic Attribution

Department of Computer Science & Engineering
University of Rajshahi
