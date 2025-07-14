import streamlit as st
import pandas as pd

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
st.sidebar.header("Caractéristiques du patient")
def patient():
    Cardiopathie = st.sidebar.selectbox("Cardiopathie", ("NON", "OUI"))
    Ulcere_gastrique = st.sidebar.selectbox("Ulcere Gastrique", ("NON", "OUI"))
    Ulcero_bourgeonnant = st.sidebar.selectbox("Ulcero-bourgeonnant", ("NON", "OUI"))
    Constipation = st.sidebar.selectbox("Constipation", ("NON", "OUI"))
    Denitrution = st.sidebar.selectbox("Denitrution", ("NON", "OUI"))
    Tabac = st.sidebar.selectbox("Tabac", ("NON", "OUI"))
    Tubuleux = st.sidebar.selectbox("Tubuleux", ("NON", "OUI"))
    Infiltrant = st.sidebar.selectbox("Infiltrant", ("NON", "OUI"))
    Stenosant = st.sidebar.selectbox("Stenosant", ("NON", "OUI"))
    Metastases = st.sidebar.selectbox("Metastases", ("NON", "OUI"))
    Adenopathie = st.sidebar.selectbox("Adenopathie", ("NON", "OUI"))
    donne = {
        "Cardiopathie": Cardiopathie,
        "Ulcere_gastrique": Ulcere_gastrique,
        "Ulcero-bourgeonnant": Ulcero_bourgeonnant,
        "Constipation": Constipation,
        "Denitrution": Denitrution,
        "Tabac": Tabac,
        "Tubuleux": Tubuleux,
        "Infiltrant": Infiltrant,
        "Stenosant": Stenosant,
        "Metastases": Metastases,
        "Adenopathie": Adenopathie
    }
    donneePatient = pd.DataFrame(donne, index=[0])
    return donneePatient
