# 📊 DWLR Data Analysis using Principal Component Analysis (PCA)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Technique-PCA-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Project Overview

This project applies **Principal Component Analysis (PCA)** on the **DWLR Dataset (2023)** to reduce dimensionality and extract meaningful patterns from high-dimensional data.

PCA helps in:

- Reducing redundant features  
- Preserving maximum variance  
- Improving computational efficiency  
- Enhancing data visualization  
- Preparing data for machine learning models  

---

## 🎯 Objectives

- Preprocess and normalize the DWLR dataset  
- Apply PCA using Scikit-learn  
- Analyze explained variance ratio  
- Generate Scree Plot  
- Visualize reduced 2D feature space  
- Interpret principal components  

---

## 🧠 Mathematical Background

Let dataset be represented as:

\[
X = \{x_1, x_2, ..., x_n\}
\]

### 1️⃣ Standardization

\[
Z = \frac{X - \mu}{\sigma}
\]

### 2️⃣ Covariance Matrix

\[
C = \frac{1}{n-1} X^T X
\]

### 3️⃣ Eigen Decomposition

\[
C v = \lambda v
\]

Where:

- \( \lambda \) = Eigenvalues (variance explained)  
- \( v \) = Eigenvectors (principal components)

### 4️⃣ Projection to Lower Dimension

\[
Z_{new} = XW
\]

Where:

- \(W\) = Top \(k\) eigenvectors  
- \(Z_{new}\) = Reduced dataset  

---

## 📂 Project Structure

```bash
DWLR-Data-Analysis-using-Principal-Component-Analysis
├── Data
│   └── DWLR_Dataset_2023.csv
│
├── Notebooks
│   └── PCA_on_DWLR.ipynb
│
├── Results
│   ├── Explained_Variance.png
│   ├── Pca_Scatter.png
│   └── Scree_Plot.png
│
├── LICENSE
└── README.md
```

---

## 🛠️ Technologies Used

- Python 3.x  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Jupyter Notebook  

---

## ⚙️ Installation & Setup

Clone the repository:

```bash
git clone https://github.com/your-username/DWLR-Data-Analysis-using-Principal-Component-Analysis.git
cd DWLR-Data-Analysis-using-Principal-Component-Analysis
```

Install required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter scipy
```

Run Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
Notebooks/PCA_on_DWLR.ipynb
```

---

## 📊 Workflow

1. Load `DWLR_Dataset_2023.csv`  
2. Perform data cleaning  
3. Standardize features  
4. Apply PCA  
5. Compute explained variance ratio  
6. Generate visualizations:
   - Explained Variance Plot  
   - Scree Plot  
   - PCA Scatter Plot  

---

## 📈 Results

The **Results** folder contains:

- **Explained_Variance.png** → Shows variance explained by each component  
- **Scree_Plot.png** → Helps determine optimal number of components  
- **Pca_Scatter.png** → Visualizes data in reduced 2D space  

### Key Findings

- First principal component captures maximum variance  
- Significant dimensionality reduction achieved  
- Clear visualization of feature distribution  
- PCA effectively removes redundant dimensions  

---

## 🚀 Future Enhancements

- Apply Kernel PCA  
- Compare with t-SNE and UMAP  
- Perform clustering on reduced data  
- Build interactive dashboard  
- Deploy as a web-based analytics tool  

---

## 📜 License

This project is licensed under the MIT License.
