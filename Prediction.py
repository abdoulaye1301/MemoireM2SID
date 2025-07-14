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
    colon=st.sidebar.columns(2)
    # Collecte des données du patient

    Cardiopathie = colon[0].sidebar.selectbox("Cardiopathie", ("NON", "OUI"))
    Ulcere_gastrique = colon[1].sidebar.selectbox("Ulcere Gastrique", ("NON", "OUI"))
    Ulcero_bourgeonnant = colon[0].sidebar.selectbox("Ulcero-bourgeonnant", ("NON", "OUI"))
    Constipation = colon[1].sidebar.selectbox("Constipation", ("NON", "OUI"))
    Denitrution = colon[0].sidebar.selectbox("Denitrution", ("NON", "OUI"))
    Tabac = colon[1].sidebar.selectbox("Tabac", ("NON", "OUI"))
    Tubuleux = colon[0].sidebar.selectbox("Tubuleux", ("NON", "OUI"))
    Infiltrant = colon[1].sidebar.selectbox("Infiltrant", ("NON", "OUI"))
    Stenosant = colon[0].sidebar.selectbox("Stenosant", ("NON", "OUI"))
    Metastases = colon[1].sidebar.selectbox("Metastases", ("NON", "OUI"))
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
