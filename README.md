# 💳 Data Driven Credit Risk Assessment System

An interactive dashboard built with Streamlit to analyze and compare the performance of multiple classification models on the UCI Credit Approval Dataset.
This platform automates the end-to-end ML pipeline, from data fetching and preprocessing to model training and interactive evaluation.

**🔗 Live:** [Data Driven Credit Risk Assessment System View the Dashboard on Streamlit](https://data-driven-credit-risk-assessment-system.streamlit.app)

---

## 🚀 Key Features

* **Automated Data Pipeline :** Fetches the Credit Approval dataset directly from the UCI Machine Learning Repository.
  
* **Comprehensive Preprocessing :** Handles categorical encoding (Label Encoding), missing value imputation (Mean), and feature scaling (StandardScaler) automatically.
  
* **Multi-Model Benchmarking :** Compares 6 different classification algorithms:
    * Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), Gradient Boosting Machines (GBM), Neural Network (MLP)
      
* **Interactive Visualizations :**
    * Global accuracy comparisons using Plotly bar charts.
    * Detailed model metrics (Precision, Recall, F1-Score).
    * Interactive Confusion Matrices using Plotly heatmaps.

## 🛠 Technical Stack

| Category | Tools |
| :--- | :--- |
| **Framework** | Streamlit |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn |
| **Data API** | ucimlrepo (University of California Irvine Machine Learning Repository API) |
| **Visualization** | Plotly Express, Matplotlib |

---
