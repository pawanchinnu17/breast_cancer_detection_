# 🩺 Breast Cancer Detection Using Machine Learning

This project focuses on early detection of breast cancer using a supervised machine learning model. It leverages the **Wisconsin Breast Cancer Dataset** to predict whether a tumor is **benign** or **malignant** based on various cellular characteristics.

---

## 📊 Project Highlights

- 🔍 Applied **exploratory data analysis (EDA)** to understand the distribution and correlation between features.
- 🧪 Preprocessed data including scaling and label encoding.
- 🧠 Built and evaluated multiple ML models: Logistic Regression, SVM, Random Forest, and KNN.
- ✅ Achieved high accuracy, precision, and recall for early-stage cancer detection.
- 📈 Includes data visualizations such as heatmaps and class distributions for better interpretation.
- 🚀 Model can be deployed for real-time predictions using web frameworks like **Streamlit**.

---

## 🧪 Technologies Used

- **Python** (NumPy, Pandas, Matplotlib, Seaborn)
- **Scikit-learn** (classification models, train-test split, metrics)
- **Jupyter Notebook**
- Optional: **Streamlit** for UI deployment

---

## 📂 Dataset

The dataset used is the famous **Breast Cancer Wisconsin (Diagnostic) Dataset**, available from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).

| Feature | Description |
|---------|-------------|
| `radius_mean`, `texture_mean`, `perimeter_mean`, ... | Mean values of different cell nucleus properties |
| `diagnosis` | Target variable (B = Benign, M = Malignant) |

---

## 📈 Exploratory Data Analysis (EDA)

- Distribution of malignant vs benign cases
- Feature correlation matrix
- Pair plots of significant features
- Box plots to identify outliers

---

## 🤖 Machine Learning Models Used

| Model              | Accuracy |
|-------------------|----------|
| Logistic Regression | ✅ High |
| Support Vector Machine | ✅ High |
| K-Nearest Neighbors | ✅ High |
| Random Forest | ✅ High |

- Used `train_test_split` for validation
- Compared models using **accuracy**, **precision**, **recall**, and **F1-score**

---

## 🏁 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/breast_cancer_detection_.git
cd breast-cancer-detection
