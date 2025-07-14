
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from Prediction import patient
from sklearn.preprocessing import OrdinalEncoder
# from PIL import Image


st.set_page_config(page_title="ND_FA_Biostatistique", page_icon="🎢", layout="centered")

@st.cache_data
def chargement():
    donnee = pd.read_excel("GastricCancerData (2).xlsx", engine='openpyxl')
    # Modification des noms des variables
    donnee.columns = ['AGE', 'SEXE', 'Cardiopathie', 'Ulcere_gastrique', 'Douleur_epigastrique',
       'Ulcero-bourgeonnant', 'Constipation', 'Denitrution', 'Tabac',
       'Mucineux', 'Tubuleux', 'Infiltrant', 'Stenosant', 'Metastases',
       'Adenopathie', 'Traitement', 'Tempsdesuivi (Mois)', 'Deces']
    donnee.drop(["Douleur_epigastrique","Douleur_epigastrique","AGE", "SEXE","Mucineux","Traitement",'Tempsdesuivi (Mois)', 'Deces'],axis=1,inplace=True)
    return donnee

# Definition de la fonction principale
def main():
    st.subheader(
        "Application de prédiction de la survie des patients atteints de cancer"
    )
    st.text("   ")
    st.text("   ")
    #st.image("biostatistique.jpg", use_column_width=True)
    st.set_page_config(page_title="ND_BA_Biostatistique", page_icon="🎢", layout="centered")
    df = chargement().iloc[1:].reset_index(drop=True)
    donne2 = patient()
    donnee_entre = pd.concat([donne2,df], axis=0)
    # Encodage des variables d'entrées
    varQuali = donnee_entre.columns.tolist()
    categories_order = [
                    ['NON','OUI'] ,
                    ['NON','OUI'] ,
                    ['NON','OUI']  ,
                    ['NON','OUI'] ,
                    ['NON','OUI']  ,
                   ['NON','OUI']  ,
                    ['NON','OUI']  ,
                    ['NON','OUI']  ,
                    ['NON','OUI']  ,
                   ['NON','OUI'] ,
                    ['NON','OUI']  
                ]
    # instanciation
    encoder = OrdinalEncoder(categories=categories_order)
    donnee_entre.loc[:, varQuali] = encoder.fit_transform(donnee_entre[varQuali])
    for var in varQuali:
        donnee_entre[var] = donnee_entre[var].apply(lambda x: int(x))

    # Récupération de la première ligne
    donnee_entre = donnee_entre[:1]

    # Importation du moèle
    chargement_modele = joblib.load("random_survival_forest.pkl")

    #Prédictions de prédiction de la fonction de survie:
    fig, ax = plt.subplots()
    survival_functions = chargement_modele.predict_survival_function(donnee_entre)
    for i, sf in enumerate(survival_functions):
        ax.step(sf.x, sf.y, where="post")
    plt.ylabel("Probabilité de survie")
    plt.xlabel("Temps de survie en mois")
    plt.legend()
    plt.show()
    st.pyplot(fig)
    fichier_css = "style.css"
    with open(fichier_css) as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
if __name__ == "__main__":
    main()
