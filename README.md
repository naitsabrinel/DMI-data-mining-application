# DMI DATA MINING APP

## 📊 Overview

DMI is an interactive data mining application developed in Python that allows users to visually and quantitatively analyze data using both clustering and classification algorithms.
It provides a complete pipeline from data loading → preprocessing → modeling (clustering & classification) → evaluation → visualization, all through an intuitive graphical interface.
---

## 🎯 Objectives

*Compare multiple clustering and classification algorithms
*Visualize results in 2D and 3D
*Evaluate model performance using metrics
*Provide an easy-to-use interface for data analysis
*Support custom and predefined datasets
---

## 🛠️ Technologies Used

* **Language:** Python 3.x
* **GUI:** Tkinter, ttk
* **Data Processing:** pandas, numpy
* **Machine Learning:** scikit-learn, scipy, pyclustering
* **Visualization:** matplotlib, mpl_toolkits.mplot3d

---

## 🧩 Features

### 🔍 Data Preview

* Load custom CSV files or predefined datasets (Iris, Wine, Blobs)
* Display first rows of data
* View dataset structure (types, missing values)
* Statistical summary (mean, median, quartiles, etc.)

---

### ⚙️ Preprocessing

* **Categorical Encoding**

  * One-Hot Encoding
  * Label Encoding
* **Missing Values Handling**

  * Mean / Median / Mode imputation
  * Drop columns with many missing values
* **Duplicate Detection & Removal**
* **Feature Scaling**

  * Min-Max Scaling
  * Standardization (Z-score)

---

### 📈 Clustering Algorithms(Unsupervised Learning)

* K-Means
* K-Medoids
* Agglomerative (AGNES)
* DIANA
* DBSCAN
  
### Classification Algorithms (Supervised Learning)
*k-Nearest Neighbors (k-NN)
*Naive Bayes
*Support Vector Machine (SVM)
---

### 📊 Analysis Tools

* **Elbow Method** → Find optimal number of clusters
* **Dendrogram** → Visualize hierarchical clustering
* **Model Evaluation Metrics** → Assess classification performance

---

### 📤 Results Export

* Export clustering labels to CSV
* Save results for further analysis

---

## 🖥️ Application Structure

The app is organized into 5 main tabs:

1. Welcome
2. Preview
3. Preprocessing
4. Modeling (Clustering & Classification)
5. Results

---

## 📌 Project Context

This project was developed as part of a **Data Mining lab (Software Engineering – 4th year)**.

---

## 👩‍💻 Authors

* Sabrinel Nait Cherif
* Mesgui Alae

---

## 📈 Conclusion

DMI bridges the gap between **theoretical data mining concepts and practical experimentation**, offering a powerful tool for exploring clustering techniques in an interactive way.
