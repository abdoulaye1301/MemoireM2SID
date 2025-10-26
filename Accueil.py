import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#import joblib
from SSVM import ssvm
from RSF import rsf
from sklearn.preprocessing import OrdinalEncoder
#from sksurv.metrics import concordance_index_censored
#import numpy as np
#from sksurv.util import Surv

# from PIL import Image

st.set_page_config(page_title="ND_FA_Biostatistique", page_icon="🎢", layout="centered")
@st.cache_data
def chargement():
    donnee = pd.read_excel("GastricCancerData (2).xlsx", engine='openpyxl')
    # Modification des noms des variables
    donnee.columns = ['N° Patient','AGE', 'SEXE', 'Cardiopathie', 'Ulcere_gastrique', 'Douleur_epigastrique',
       'Ulcero-bourgeonnant', 'Constipation', 'Denitrution', 'Tabac',
       'Mucineux', 'Tubuleux', 'Infiltrant', 'Stenosant', 'Metastases',
       'Adenopathie', 'Traitement', 'Tempsdesuivi (Mois)', 'Deces']
    donnee.drop(["Douleur_epigastrique","AGE", "SEXE","Mucineux","Ulcere_gastrique","Constipation","Denitrution",
                 'Tempsdesuivi (Mois)', 'Traitement','Deces'],axis=1,inplace=True)
    return donnee

# Definition de la fonction principale
def main():
    st.subheader(
        "Application de prédiction de la survie des patients atteints de cancer"
    )
    st.text("   ")
    st.text("   ")

    df = chargement().iloc[1:].reset_index(drop=True)
    # Collecte des données du patient
    
    colonnes=st.sidebar.columns(2)
    colonnes[1].subheader("du patient")
    colonnes[0].subheader(" Caractéristiques")
    df_final= df.copy()
    df_final.sort_values(by='N° Patient', ascending=False,inplace=True)
    numPatient=st.sidebar.selectbox("Sélectionner le patient", df_final['N° Patient'].unique())
    donneePatient=df[df_final['N° Patient']==numPatient]
   
    #==================================================================#
    choix=st.selectbox("Navigation", ["RSF", "SSVM"], key="navigation")
    if choix=="RSF":
        colon=st.sidebar.columns(2)
        #Ulcere_gastrique = colon[1].selectbox("Ulcere Gastrique", ("NON", "OUI"))
        colon[0].write(f"**Cardiopathie** : {donneePatient['Cardiopathie'].values[0]}")
        colon[0].text("   ")
        colon[0].text("   ")
        #Constipation = colon[1].selectbox("Constipation", ("NON", "OUI"))
        #Denitrution = colon[0].selectbox("Denitrution", ("NON", "OUI"))
        colon[1].write(f"**Tabac :** {donneePatient['Tabac'].values[0]}")
        #Tubuleux = colon[0].selectbox("Tubuleux", ("NON", "OUI"))
        colon[1].text("   ")
        colon[1].text("   ")
        colon[0].write(f"**Ulcero-bourgeonnant :** {donneePatient['Ulcero-bourgeonnant'].values[0]}")

        colon[0].text("   ")
        colon[1].write(f"**Infiltrant :** {donneePatient['Infiltrant'].values[0]}")
        colon[1].text("   ")
        colon[1].text("   ")
        colon[0].write(f"**Stenosant :** {donneePatient['Stenosant'].values[0]}")
        colon[0].text("   ")
        colon[0].text("   ")
        colon[1].write(f"**Metastases :** {donneePatient['Metastases'].values[0]}")
        colon[1].text("   ")
        colon[1].text("   ")
        colon[0].write(f"**Adenopathie :** {donneePatient['Adenopathie'].values[0]}")
        colon[0].text("   ")
        colon[0].text("   ")
        colon[1].write(f"**Tubuleux :** {donneePatient['Tubuleux'].values[0]}")




        #donne2 = patient()
        #donnee_entre = pd.concat([donne2,df], axis=0)
        donnee_entre = donneePatient.drop(columns=['N° Patient'])
        # Encodage des variables d'entrées
        varQuali = donnee_entre.columns.tolist()
        #categories_order = [['NON','OUI']] * len(varQuali)
        categories_order = [
                        ['NON','OUI'] ,
                        ['NON','OUI'] ,
                        ['NON','OUI']  ,
                        ['NON','OUI'] ,
                        ['NON','OUI']  ,
                        ['NON','OUI']  ,
                        ['NON','OUI']  ,
                        ['NON','OUI']  
                    ]
        encoder = OrdinalEncoder(categories=categories_order)
        donnee_entre.loc[:, varQuali] = encoder.fit_transform(donnee_entre[varQuali])
        donnee_entre = donnee_entre.astype(int)

        # Récupération de la première ligne (nouveau patient)
        donnee_entre = donnee_entre[:1]
        rsf(donnee_entre)
    elif choix=="SSVM":
        colon=st.sidebar.columns(2)
        #Ulcere_gastrique = colon[1].selectbox("Ulcere Gastrique", ("NON", "OUI"))
        colon[0].write(f"**Cardiopathie** : {donneePatient['Cardiopathie'].values[0]}")
        colon[0].text("   ")
        colon[0].text("   ")
        #Constipation = colon[1].selectbox("Constipation", ("NON", "OUI"))
        #Denitrution = colon[0].selectbox("Denitrution", ("NON", "OUI"))
        colon[1].write(f"**Tabac :** {donneePatient['Tabac'].values[0]}")
        colon[0].text("   ")
        colon[1].write(f"**Infiltrant :** {donneePatient['Infiltrant'].values[0]}")
        colon[1].text("   ")
        colon[1].text("   ")
        colon[0].write(f"**Stenosant :** {donneePatient['Stenosant'].values[0]}")
        colon[0].text("   ")
        colon[0].text("   ")
        colon[1].write(f"**Metastases :** {donneePatient['Metastases'].values[0]}")
        colon[1].text("   ")
        colon[1].text("   ")
        colon[0].write(f"**Adenopathie :** {donneePatient['Adenopathie'].values[0]}")

        #donne2 = patient()
        #donnee_entre = pd.concat([donne2,df], axis=0)
        donnee_entre = donneePatient.drop(columns=['N° Patient'])
        # Encodage des variables d'entrées
        varQuali = donnee_entre.columns.tolist()
        #categories_order = [['NON','OUI']] * len(varQuali)
        categories_order = [
                        ['NON','OUI'] ,
                        ['NON','OUI'] ,
                        ['NON','OUI']  ,
                        ['NON','OUI'] ,
                        ['NON','OUI']  ,
                        ['NON','OUI']  ,
                        ['NON','OUI']  ,
                        ['NON','OUI']  
                    ]
        encoder = OrdinalEncoder(categories=categories_order)
        donnee_entre.loc[:, varQuali] = encoder.fit_transform(donnee_entre[varQuali])
        donnee_entre = donnee_entre.astype(int)

        # Récupération de la première ligne (nouveau patient)
        donnee_entre = donnee_entre[:1]
        donnee_entre.drop('Ulcero-bourgeonnant', axis=1,inplace=True)
        donnee_entre.drop('Tubuleux', axis=1, inplace=True)
        #st.write(donnee_entre)
        ssvm(donnee_entre)
        
    # Chargement du CSS
    fichier_css = "style.css"
    with open(fichier_css) as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

if __name__ == "__main__":
    main()


