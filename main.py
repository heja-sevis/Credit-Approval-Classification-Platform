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
import matplotlib.pyplot as plt
import seaborn as sns

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Kredi Onayı Modelleri (Tek Dosya)", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# 1. Veri Yükleme ve Model Eğitimi
# ----------------------------------------------------------------------

@st.cache_resource(show_spinner="Veri yükleniyor ve tüm modeller eğitiliyor...")
def load_data_and_train_models():  
    # Veri Yükleme (data Subset.ipynb)
    try:
        credit_approval = fetch_ucirepo(id=27)
        X = credit_approval.data.features
        y = credit_approval.data.targets
        
    except Exception as e:
        st.error(f"Veri yüklenirken hata: {e}")
        return None, None, None
    
    # Ön İşleme (Label Encoding)
    X_processed = X.copy()
    categorical_columns = X_processed.select_dtypes(include=['object']).columns

    for col in categorical_columns:
        le = LabelEncoder()
        # Eksik değerleri Label Encoding yapmadan önce doldurmak için str'ye dönüştür
        X_processed[col] = le.fit_transform(X_processed[col].astype(str))

    if isinstance(y, pd.DataFrame):
        y = y.squeeze()
    if y.dtype == 'object' or isinstance(y.iloc[0], str):
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

    # Split, Scaling, Imputation (data Imputation.ipynb ve data Classifiers.ipynb)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SimpleImputer ile eksik değerleri ortalama ile doldurma
    imputer = SimpleImputer(strategy='mean')
    X_train_final = imputer.fit_transform(X_train_scaled)
    X_test_final = imputer.transform(X_test_scaled)
    
    # Model Eğitimi (data Classifiers.ipynb)
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
        
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "report": classification_report(y_test, y_pred, output_dict=True),
            "conf_matrix": confusion_matrix(y_test, y_pred),
        }
    
    # Kullanılacak verileri döndür
    X_df = pd.DataFrame(X_processed, columns=X.columns)
    return results, X_df, credit_approval.metadata


# ----------------------------------------------------------------------
# 2. SAYFA FONKSİYONLARI
# ----------------------------------------------------------------------

def show_data_prep_page(X_raw, metadata):
    """Veri Hazırlığı ve Giriş sayfasını gösterir."""
    st.title("📊 Veri Hazırlığı ve Giriş")
    st.markdown("Bu sayfada kullanılan veri setinin (UCI Credit Approval) ön izlemesi ve ön işleme adımları gösterilmektedir.")

    # Veri setini gösterme
    st.subheader("1. Ön İşleme Yapılmış Veri Seti Ön İzlemesi")
    st.dataframe(X_raw.head(10), use_container_width=True)
    st.caption(f"Toplam örnek sayısı: **{X_raw.shape[0]}**, Özellik sayısı: **{X_raw.shape[1]}**")

    # Ön İşleme Adımları
    st.subheader("2. Uygulanan Ön İşleme Adımları")
    st.markdown("""
    * **Veri Yükleme:** Veri seti UCI Machine Learning Repository'den çekildi.
    * **Label Encoding:** Tüm kategorik özellikler sayısallaştırıldı (**data Subset.ipynb**).
    * **Eksik Değer Doldurma (Imputation):** Eksik değerler ortalama (`SimpleImputer(strategy='mean')`) kullanılarak dolduruldu (**data Imputation.ipynb**).
    * **Ölçeklendirme (Scaling):** Tüm özellikler `StandardScaler` kullanılarak standartlaştırıldı.
    * **Veri Bölme:** Veriler Eğitim (%70) ve Test (%30) setlerine ayrıldı.
    """)

def show_model_comparison_page(results):
    """Model Karşılaştırma ve Sonuçlar sayfasını gösterir."""
    st.title("🏆 Sınıflandırma Modelleri Karşılaştırması")
    st.markdown("Eğitilmiş modellerden birini seçerek detaylı performans metriklerini (Doğruluk, Rapor, Karmaşıklık Matrisi) inceleyin.")
    
    # --- Sidebar Model Seçimi ---
    st.sidebar.header("Model Seçimi")
    model_name = st.sidebar.selectbox(
        "İncelenecek Modeli Seçin:",
        list(results.keys()),
        index=2 
    )

    selected_result = results[model_name]

    st.header(f"Seçilen Model: {model_name}")
    st.write("---")

    col1, col2 = st.columns([1, 2])
    
    # 1. Doğruluk (Accuracy) ve Karşılaştırma
    with col1:
        st.subheader("Doğruluk Skoru")
        st.metric(label="Test Seti Doğruluğu", 
                  value=f"{selected_result['accuracy']:.4f}")
        
        # Tüm model doğruluklarını gösteren tablo
        all_accuracies = {name: res['accuracy'] for name, res in results.items()}
        accuracy_df = pd.DataFrame(all_accuracies.items(), columns=['Model', 'Doğruluk Skoru'])
        accuracy_df['Doğruluk Skoru'] = accuracy_df['Doğruluk Skoru'].map('{:.4f}'.format)
        
        st.markdown("##### Tüm Modellerin Doğruluk Karşılaştırması")
        st.dataframe(accuracy_df.set_index('Model').sort_values(by='Doğruluk Skoru', ascending=False), 
                     use_container_width=True)

    # 2. Sınıflandırma Raporu
    with col2:
        st.subheader("Sınıflandırma Raporu (Precision, Recall, F1-Score)")
        report_df = pd.DataFrame(selected_result['report']).transpose()
        # Sayısal formatı düzenleme
        for col in ['precision', 'recall', 'f1-score']:
            if col in report_df.columns:
                 report_df[col] = report_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                 
        st.dataframe(report_df, use_container_width=True)

    st.write("---")

    # 3. Karmaşıklık Matrisi Görseli
    st.subheader("Karmaşıklık Matrisi Görselleştirmesi")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = selected_result['conf_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Onaylanmadı (0)', 'Onaylandı (1)'], 
                yticklabels=['Onaylanmadı (0)', 'Onaylandı (1)'],
                ax=ax)
    ax.set_title(f"{model_name} Karmaşıklık Matrisi")
    ax.set_xlabel("Tahmin Edilen")
    ax.set_ylabel("Gerçek")
    st.pyplot(fig)


# ----------------------------------------------------------------------
# 3. ANA UYGULAMA MANTIĞI
# ----------------------------------------------------------------------

def main():
    
    # 1. Veri Yükleme ve Modelleri Eğitme
    # Bu fonksiyon, @st.cache_resource sayesinde sadece ilk seferde çalışır.
    results, X_raw, metadata = load_data_and_train_models()
    
    if results is None:
        return

    # 2. Sayfa Seçimi (Sidebar)
    PAGES = {
        "Veri Hazırlığı ve Giriş": show_data_prep_page,
        "Model Karşılaştırma ve Sonuçlar": show_model_comparison_page,
    }

    st.sidebar.title("Kredi Onayı Analizi")
    st.sidebar.markdown("---")
    
    selection = st.sidebar.radio("Sayfa Seçimi", list(PAGES.keys()))
    st.sidebar.markdown("---")
    st.sidebar.info("Model eğitimi tamamlandı ve sonuçlar önbelleğe alındı.")
    
    # 3. Seçilen Sayfayı Göster
    if selection == "Veri Hazırlığı ve Giriş":
        PAGES[selection](X_raw, metadata)
    elif selection == "Model Karşılaştırma ve Sonuçlar":
        PAGES[selection](results)

if __name__ == "__main__":
    main()
