
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
# from PIL import Image


st.set_page_config(page_title="ND_FA_Biostatistique", page_icon="🎢", layout="centered")
st.sidebar.success("Selectionnez une page")


# Definition de la fonction principale
def main():
    st.title(
        "Application de prédiction de la survie des patients atteints de cansaire"
    )
    st.subheader("Auteurs : Abdoulaye NDAO, Amadou BA")
    st.markdown(
        "**Cette étude consiste à mettre en place un modèle de machine learning ou statistique qui permet de faire un pronostique sur la survenue instantanée de décès après le traitement.**"
        "**Pour la construction de ce modèle, nous allons utiliser les données des patients atteints**"
        "**d’accident cérébral vasculaire (AVC), traités et suivis.**"
    )
    st.text("   ")
    st.text("   ")
    st.image("biostatistique.jpg", use_column_width=True)
    st.set_page_config(page_title="ND_BA_Biostatistique", page_icon="🎢", layout="centered")


    # Fonction de chargement du jeu de données
    @st.cache_data(persist=True)
    def chargement():
        # Upload du fichier Excel et csv
        Chargement = st.sidebar.file_uploader(" 📁 Charger un fichier Excel", type=["xlsx","csv"])
        if Chargement is not None:
            if Chargement.name.endswith('.xlsx'):
                donnee = pd.read_excel(Chargement)
            elif Chargement.name.endswith('.csv'):
                donnee = pd.read_csv(Chargement)
            else:
                st.error("Format de fichier non supporté. Veuillez charger un fichier Excel (.xlsx) ou CSV (.csv).")
                return None
        else:
            st.warning("Veuillez charger un fichier pour continuer.")
            return None
        return donnee


    df = chargement()


if __name__ == "__main__":
    main()
