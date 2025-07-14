import streamlit as st
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import OrdinalEncoder
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="ND_FA_Biostatistique", page_icon="🎢", layout="centered")
st.title("Prédition de la survenue instantanée de décès après le traitement")
st.markdown(
    "**Nous avons utilisre le modèle de Cox pour effectuer nos préduction, car il est spécifiquement choisi**"
    "**pour analyser les données de survie, où l'objectif est de modéliser le temps jusqu'à l'arrivée de**"
    "**l'événement d'intérêt (décès). Il nous permet aussi de prend en compte à la fois les événements**"
    "**observés et les données censurées, c'est-à-dire les individus pour lesquels l'événement**"
    "**n'est pas survenu à la fin de l'étude.**"
)

# Colletion des données d'entré
st.sidebar.header("Caractéristiques du patien")


def patient():
    Premiers_Signe = st.sidebar.slider(
        "Premiers Signe - Admission à l'hopital", min_value=1, max_value=20, step=1
    )
    Admission_hopital = st.sidebar.slider(
        "Admission à l'hopital - Prise en charge medicale",
        min_value=1,
        max_value=20,
        step=1,
    )
    Temps_Suivi = st.sidebar.slider(
        "Temps de Suivi après traitement (en jours)", min_value=1, max_value=366, step=1
    )
    # SEXE = st.sidebar.selectbox("Sexe", ("Homme", "Femme"))
    # Cardiopathie = st.sidebar.selectbox("Cardiopathie", ("NON", "OUI"))
    hémiplégie = st.sidebar.selectbox("Hémiplégie", ("NON", "OUI"))
    Paralysie_faciale = st.sidebar.selectbox("Paralysie faciale", ("NON", "OUI"))
    Aphasie = st.sidebar.selectbox("Aphasie", ("NON", "OUI"))
    # Hémiparésie = st.sidebar.selectbox("Hémiparésie", ("NON", "OUI"))
    Inondation_Ventriculaire = st.sidebar.selectbox(
        "Inondation Ventriculaire", ("NON", "OUI")
    )
    Traitement = st.sidebar.selectbox("Traitement", ("Thrombolyse", "Chirurgie"))
    donne = {
        "Premiers_Signe": Premiers_Signe,
        "Admission_hopital": Admission_hopital,
        "hémiplégie": hémiplégie,
        "Paralysie_faciale": Paralysie_faciale,
        "Aphasie": Aphasie,
        "Inondation_Ventriculaire": Inondation_Ventriculaire,
        "Traitement": Traitement,
        "Temps_Suivi": Temps_Suivi,
    }
    donneePatient = pd.DataFrame(donne, index=[0])
    return donneePatient


donne2 = patient()


# Tranformation des données d'entré
# Fonction de chargement du jeu de données
@st.cache_data(persist=True)
def chargement():
    donnee = pd.read_excel("Donnnées_Projet_M2SID2023_2024.xlsx")
    return donnee


df = chargement()
donne1 = df.drop(columns=["Evolution"])
donne1.columns = [
    "AGE",
    "SEXE",
    "Premiers_Signe",
    "Admission_hopital",
    "Hypertension_Arterielle",
    "Diabete",
    "Cardiopathie",
    "hémiplégie",
    "Paralysie_faciale",
    "Aphasie",
    "Hémiparésie",
    "Engagement_Cerebral",
    "Inondation_Ventriculaire",
    "Traitement",
    "Temps_Suivi",
]
donne1.drop(
    columns=[
        "SEXE",
        "AGE",
        "Hypertension_Arterielle",
        "Diabete",
        "Engagement_Cerebral",
        "Hémiparésie",
        "Cardiopathie",
    ],
    axis=1,
    inplace=True,
)
donnee_entre = pd.concat([donne2, donne1], axis=0)
# Encodage des variables d'entrées
varQual = donnee_entre.select_dtypes(include="object").columns.tolist()
categories_order = [
    ["NON", "OUI"],
    ["NON", "OUI"],
    ["NON", "OUI"],
    ["NON", "OUI"],
    ["Thrombolyse", "Chirurgie"],
]
# instanciation
encoder = OrdinalEncoder(categories=categories_order)
donnee_entre.loc[:, varQual] = encoder.fit_transform(donnee_entre[varQual])
for var in varQual:
    donnee_entre[var] = donnee_entre[var].apply(lambda x: int(x))

# Récupération de la première ligne
donnee_entre = donnee_entre[:1]

# Affichage des données transformé
# st.write(donnee_entre)
# if st.sidebar.button("Prediction"):
# Importation du moèle
chargement_modele = joblib.load("projet_biostatistique.h5")

# Prévision
prevision = chargement_modele.predict_survival_function(donnee_entre)

# Affichage du prévision
st.subheader("Résultat de la prévision")
# st.text(prevision)
prevision.plot()
plt.title("Courbe de prévision de survie du patient après le traitement")
plt.xlabel("Durée de survie en mois")
plt.ylabel("Probabilité de survie")
st.pyplot()