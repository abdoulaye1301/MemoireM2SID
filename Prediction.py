import streamlit as st
import pandas as pd

# Colletion des données d'entré

def patient():
    # Collecte des données du patient
    
    colon=st.sidebar.columns(2)
    colon[1].subheader("du patient")
    colon[0].subheader(" Caractéristiques")
    Ulcere_gastrique = colon[1].selectbox("Ulcere Gastrique", ("NON", "OUI"))
    Cardiopathie =colon[0].selectbox("Cardiopathie", ("NON", "OUI"))
    Constipation = colon[1].selectbox("Constipation", ("NON", "OUI"))
    Denitrution = colon[0].selectbox("Denitrution", ("NON", "OUI"))
    Tabac = colon[1].selectbox("Tabac", ("NON", "OUI"))
    Tubuleux = colon[0].selectbox("Tubuleux", ("NON", "OUI"))
    Infiltrant = colon[1].selectbox("Infiltrant", ("NON", "OUI"))
    Stenosant = colon[0].selectbox("Stenosant", ("NON", "OUI"))
    Metastases = colon[1].selectbox("Metastases", ("NON", "OUI"))
    Adenopathie =  colon[0].selectbox("Adenopathie", ("NON", "OUI"))
    Ulcero_bourgeonnant = st.sidebar.selectbox("Ulcero-bourgeonnant", ("NON", "OUI"))
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
