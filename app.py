import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Chargement des données
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", 
    "exang", "oldpeak", "slope", "ca", "thal", "target"
]
data = pd.read_csv(url, names=column_names, na_values="?")
data = data.dropna()
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

# Séparation des caractéristiques et de la cible
X = data.drop('target', axis=1)
y = data['target']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entraînement du modèle (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Interface Streamlit
st.set_page_config(page_title="Prédiction de maladie cardiaque", )

# Titre et sous-titre
st.title(" Prédiction de maladie cardiaque")
st.markdown("""
    <style>
    .big-font {
        font-size: 20px !important;
        color: #2E86C1;
    }
    </style>
    <div class="big-font">
    """, unsafe_allow_html=True)

# Section pour saisir les informations du patient
st.sidebar.header("Renseignements")

# Création des champs de saisie
col1, col2 = st.sidebar.columns(2)
with col1:
    age = st.number_input("Âge", min_value=0, max_value=120, value=50)
    sex = st.selectbox("Sexe", options=[("Homme", 1), ("Femme", 0)], format_func=lambda x: x[0])[1]
    cp = st.selectbox("Type de douleur thoracique", options=[(0, "Typical Angina"), (1, "Atypical Angina"), (2, "Non-anginal Pain"), (3, "Asymptomatic")], format_func=lambda x: x[1])[0]
    trestbps = st.number_input("Pression artérielle au repos (en mm Hg)", min_value=0, max_value=200, value=120)
    chol = st.number_input("Cholestérol sérique (en mg/dl)", min_value=0, max_value=600, value=200)
    fbs = st.selectbox("Glycémie à jeun > 120 mg/dl", options=[("Non", 0), ("Oui", 1)], format_func=lambda x: x[0])[1]

with col2:
    restecg = st.selectbox("Résultats électrocardiographiques au repos", options=[(0, "Normal"), (1, "Anomalie onde ST-T"), (2, "Hypertrophie ventriculaire gauche probable")], format_func=lambda x: x[1])[0]
    thalach = st.number_input("Fréquence cardiaque maximale atteinte", min_value=0, max_value=220, value=150)
    exang = st.selectbox("Angine induite par l'exercice", options=[("Non", 0), ("Oui", 1)], format_func=lambda x: x[0])[1]
    oldpeak = st.number_input("Dépression du segment ST induite par l'exercice", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Pente du segment ST au pic de l'exercice", options=[(0, "Montante"), (1, "Plate"), (2, "Descendante")], format_func=lambda x: x[1])[0]
    ca = st.number_input("Nombre de gros vaisseaux colorés par fluoroscopie", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thalassémie", options=[(3, "Normal"), (6, "Défaut fixe"), (7, "Défaut réversible")], format_func=lambda x: x[1])[0]

# Bouton pour faire la prédiction
if st.sidebar.button("Prédire", key="predict_button"):
    # Créer un DataFrame avec les informations du patient
    patient_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    })

    # Normaliser les données du patient
    patient_data_scaled = scaler.transform(patient_data)

    # Faire la prédiction
    prediction = model.predict(patient_data_scaled)
    prediction_proba = model.predict_proba(patient_data_scaled)

    # Afficher le résultat
    st.subheader(" Résultat de la prédiction")
    if prediction[0] == 1:
        st.error(" présence de maladie cardiaque.")
    else:
        st.success("absence de maladie cardiaque.")

    # Afficher les probabilités
    st.write(f"absence de maladie cardiaque : {prediction_proba[0][0]:.2f}")
    st.write(f"présence de maladie cardiaque : {prediction_proba[0][1]:.2f}")