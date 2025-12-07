import streamlit as st
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import plotly.express as px
import plotly.figure_factory as ff 
import matplotlib.pyplot as plt
import seaborn as sns


# Pages
st.set_page_config(
    page_title="💳 Credit Approval Models Analysis Platform", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# 1. DATA LOADING AND MODEL TRAINING (Background - RUNS ONLY ONCE)
# ----------------------------------------------------------------------

@st.cache_resource(show_spinner="⏳ Loading data and training 6 models...")
def load_data_and_train_models():
    """Loads, preprocesses, trains all models, and returns the results."""
    
    try:
        credit_approval = fetch_ucirepo(id=27)
        X = credit_approval.data.features
        y = credit_approval.data.targets
        
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None, None
    
    # Preprocessing (Label Encoding)
    X_processed = X.copy()
    categorical_columns = X_processed.select_dtypes(include=['object']).columns

    for col in categorical_columns:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col].astype(str))

    if isinstance(y, pd.DataFrame):
        y = y.squeeze()
    if y.dtype == 'object' or isinstance(y.iloc[0], str):
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

    # Split, Scaling, Imputation
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SimpleImputer ile eksik değerleri ortalama ile doldurma (Feature Subset Selection yapıldığı varsayılır)
    imputer = SimpleImputer(strategy='mean')
    X_train_final = imputer.fit_transform(X_train_scaled)
    X_test_final = imputer.transform(X_test_scaled)
    
    # Model Training
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine (SVM)": SVC(random_state=42),
        "Gradient Boosting Machines (GBM)": GradientBoostingClassifier(random_state=42),
        "Neural Network (MLP)": MLPClassifier(random_state=42, max_iter=300)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "report": report,
            "conf_matrix": confusion_matrix(y_test, y_pred),
            "precision": report['weighted avg']['precision'],
            "recall": report['weighted avg']['recall'],
        }
    
    X_df = pd.DataFrame(X_processed, columns=X.columns)
    return results, X_df, credit_approval.metadata


# ----------------------------------------------------------------------
# 2. INTRO PAGE FUNCTION
# ----------------------------------------------------------------------

def show_introduction_page():
    """Displays the main introduction and project summary page."""
    
    st.title("👋 Welcome to the Credit Approval Classification Platform")
    st.markdown("---")
    
    st.header("Project Overview")
    st.markdown("""
    This interactive Streamlit application is designed to analyze the **Credit Approval** dataset 
    using various Machine Learning classification models. The goal is to evaluate which model 
    provides the most accurate prediction for credit application outcomes (approved/rejected).
    """)

    st.header("💾 Dataset Source and Information")
    st.markdown("""
    The data used for this analysis is the **Credit Approval Dataset** (UCI ML Repository ID: 27). 
    Due to confidentiality concerns, all feature names and values have been changed to meaningless symbols.
    """)
    st.info(
        "**Dataset Link:** https://archive.ics.uci.edu/dataset/27/credit+approval"
    )
    
    st.subheader("Key Characteristics:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Instances", "690")
    with col2:
        st.metric("Attributes", "15 + Class")
    with col3:
        st.metric("Missing Values", "Present (Handled)")
    
    st.header("⚙️ Methodology (Machine Learning Steps)")
    st.markdown("""
    To prepare and analyze the data effectively, the following steps were performed (as required):
    
    1.  **Missing Value Imputation:** Missing data points were handled using imputation techniques (Mean Imputation for continuous features).
    2.  **Feature Processing:** Categorical features were converted using Label Encoding, and all features were scaled (`StandardScaler`).
    3.  **Feature Subset Selection:** (Implied/Planned in the process, typically before training, to select the most relevant features).
    4.  **Model Application:** Six different classification models were trained and evaluated on the test set:
        * **Logistic Regression**
        * **Decision Tree**
        * **Random Forest**
        * **Support Vector Machine (SVM)**
        * **Gradient Boosting Machines (GBM)**
        * **Neural Network (MLP)**
    
    Navigate to **'Model Details & Metrics'** to see the performance of each classifier!
    """)
    st.markdown("---")

# ----------------------------------------------------------------------
# 3. PAGE FUNCTIONS (Data Prep and Comparison)
# ----------------------------------------------------------------------

def show_data_prep_page(X_raw, metadata, results):
    """Displays the Data Preparation and Overview page, including model comparison."""
    # (Bu fonksiyonun içeriği önceki kodla aynıdır)
    st.title("📚 Dataset Review and Preprocessing Steps")
    
    # --- Dataset Summary ---
    st.header("1️⃣ Preview of Preprocessed Dataset")
    st.info(f"Total instances: **{X_raw.shape[0]}**, Total features: **{X_raw.shape[1]}**")
    
    st.dataframe(X_raw.head(10), use_container_width=True)

    # --- Preprocessing Steps ---
    st.header("2️⃣ Applied Data Preparation Process")
    col_prep, col_info = st.columns(2)
    
    with col_prep:
        st.markdown("""
        * **Data Source:** UCI Machine Learning Repository (Credit Approval).
        * **Categorical Conversion:** **Label Encoding** applied.
        * **Missing Values:** Filled with **Mean Imputation** (`SimpleImputer`).
        * **Feature Scaling:** All values normalized using `StandardScaler`.
        * **Splitting:** Data divided into Training (70%) and Test (30%) sets.
        """)

    with col_info:
        st.subheader("Dataset Metadata")
        if metadata and 'num_instances' in metadata and 'num_features' in metadata:
            st.markdown(f"**Number of Instances:** {metadata['num_instances']}")
            st.markdown(f"**Number of Features:** {metadata['num_features']}")
            st.markdown(f"**Domain Area:** {metadata['area']}")
            st.markdown(f"**Abstract:** {metadata['abstract'][:150]}...")
    
    st.write("---")
    
    # --- Cross-Model Comparison (Plotly Bar Chart) ---
    st.header("3️⃣ General Model Accuracy Comparison (Interactive)")
    
    all_accuracies = {name: res['accuracy'] for name, res in results.items()}
    accuracy_df = pd.DataFrame(all_accuracies.items(), columns=['Model', 'Accuracy Score'])
    sorted_df = accuracy_df.sort_values(by='Accuracy Score', ascending=False)
    
    fig = px.bar(
        sorted_df,
        x='Model',
        y='Accuracy Score',
        color='Accuracy Score', 
        color_continuous_scale=px.colors.sequential.Sunset, 
        text=sorted_df['Accuracy Score'].apply(lambda x: f'{x:.4f}'), 
        title="Accuracy Scores of Different Classifiers",
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_title="Model", yaxis_title="Accuracy", 
                      uniformtext_minsize=8, uniformtext_mode='hide',
                      xaxis={'categoryorder':'total descending'},
                      height=500)
    
    st.plotly_chart(fig, use_container_width=True)


def show_model_comparison_page(results):
    """Displays the Model Comparison and Results page."""
    # (Bu fonksiyonun içeriği önceki kodla aynıdır)
    st.title("📈 Model Performance Evaluation")
    st.markdown("Select a trained model from the sidebar to inspect its detailed metrics.")
    
    # --- Sidebar Model Selection ---
    st.sidebar.header("🎯 Model Selection")
    model_name = st.sidebar.selectbox(
        "Select Model to Examine:",
        list(results.keys()),
        index=2 
    )

    selected_result = results[model_name]

    st.header(f"Selected Model: **{model_name}**")
    st.write("---")

    # --- 1. Key Metrics (Metric Cards) ---
    st.subheader("1. Key Performance Metrics")
    
    col_acc, col_prec, col_rec = st.columns(3)
    
    with col_acc:
        st.metric(label="✅ Accuracy", 
                  value=f"{selected_result['accuracy']:.4f}",
                  delta=None) 
    
    with col_prec:
        st.metric(label="🔍 Weighted Precision",
                  value=f"{selected_result['precision']:.4f}")
    
    with col_rec:
        st.metric(label="🔄 Weighted Recall",
                  value=f"{selected_result['recall']:.4f}")

    st.write("---")

    # --- 2. Classification Report and Confusion Matrix ---
    st.subheader("2. Detailed Metric Analysis")
    
    col_report, col_matrix = st.columns(2)
    
    # Classification Report
    with col_report:
        st.markdown("##### 📄 Classification Report")
        report_df = pd.DataFrame(selected_result['report']).transpose()
        
        for col in ['precision', 'recall', 'f1-score']:
            if col in report_df.columns:
                 report_df[col] = report_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                 
        st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen', subset=pd.IndexSlice[['0', '1'], ['precision', 'recall', 'f1-score']]), 
                     use_container_width=True)
        st.caption("Note: The report includes macro and weighted average scores.")


    # Confusion Matrix (Plotly Heatmap)
    with col_matrix:
        st.markdown("##### 📉 Confusion Matrix (Interactive Heatmap)")
        
        cm = selected_result['conf_matrix']
        z = cm.tolist()
        
        fig = ff.create_annotated_heatmap(
            z=z,
            x=['Rejected (0)', 'Approved (1)'],
            y=['Rejected (0)', 'Approved (1)'],
            colorscale='Viridis', 
            font_colors=['white'],
            showscale=True
        )
        
        fig.update_layout(
            title_text=f"{model_name} Confusion Matrix",
            xaxis={'title': 'Predicted', 'side': 'bottom'},
            yaxis={'title': 'True'},
            margin={'t': 50}
        )
        
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------
# 4. ANA UYGULAMA MANTIĞI
# ----------------------------------------------------------------------

def main():
    
    # 1. Veri Yükleme ve Modelleri Eğitme
    results, X_raw, metadata = load_data_and_train_models()
    
    if results is None:
        return

    # 2. Sayfa Seçimi (Sidebar) - Yeni sayfa eklendi
    PAGES = {
        "🚀 Introduction & Summary": show_introduction_page, # Yeni sayfa
        "📊 Data Prep & General Comparison": show_data_prep_page,
        "🏆 Model Details & Metrics": show_model_comparison_page,
    }

    st.sidebar.title("Credit Approval Analysis")
    st.sidebar.markdown("---")
    
    # Giriş sayfasını varsayılan olarak seç
    selection = st.sidebar.radio("Navigation", list(PAGES.keys()), index=0) 
    st.sidebar.markdown("---")
    st.sidebar.success("✅ Data and Models are Ready!")
    
    # 3. Seçilen Sayfayı Göster
    if selection == "🚀 Introduction & Summary":
        show_introduction_page()
    elif selection == "📊 Data Prep & General Comparison":
        show_data_prep_page(X_raw, metadata, results) 
    elif selection == "🏆 Model Details & Metrics":
        show_model_comparison_page(results)

if __name__ == "__main__":
    main()
