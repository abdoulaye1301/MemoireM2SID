import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from Prediction import patient
from sklearn.preprocessing import OrdinalEncoder
from sksurv.metrics import concordance_index_censored
import numpy as np
import shap
from sksurv.util import Surv

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
    donnee.drop(["Douleur_epigastrique","Douleur_epigastrique","AGE", "SEXE","Mucineux","Traitement",
                 'Tempsdesuivi (Mois)', 'Deces'],axis=1,inplace=True)
    return donnee

# Definition de la fonction principale
def main():
    st.subheader(
        "Application de prédiction de la survie des patients atteints de cancer"
    )
    st.text("   ")
    st.text("   ")

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

    # Récupération de la première ligne (nouveau patient)
    donnee_entre = donnee_entre[:1]

    # Importation du modèle et des données de test
    chargement_modele = joblib.load("random_survival_forest.pkl")
    X_test = joblib.load("X_test.pkl")
    Y_test = joblib.load("Y_test.pkl")

    #==================================================================#
    # Créer un tableau structuré pour le temps de survie et l'événement
    Y_test_structure = Surv.from_arrays(event=Y_test.astype(bool), time=X_test['Tempsdesuivi (Mois)'])

    # Supprimer la colonne 'Tempsdesuivi (Mois)' des caractéristiques X car elle est maintenant dans Y
    X_test = X_test.drop('Tempsdesuivi (Mois)', axis=1)

    # Prédictions de la fonction de survie
    fig, ax = plt.subplots()
    survival_functions = chargement_modele.predict_survival_function(donnee_entre)
    for i, sf in enumerate(survival_functions):
        ax.step(sf.x, sf.y, where="post")
    plt.title("Fonction de survie du patient")
    plt.ylabel("Probabilité de survie")
    plt.xlabel("Temps de survie en mois")
    plt.legend()
    plt.show()
    st.pyplot(fig)

    # ==================================================================#
    # INTERPRÉTATION DU MODÈLE AVEC VIM
    st.header("Calcul de l'importance des variables")
    st.write("Calcul en cours, veuillez patienter...")

    # --- Étape 1 : Conversion de Y_test en tableau structuré ---
    try:
        dt = np.dtype([('event', 'bool'), ('time', 'float')])
        y_test_structured = np.empty(len(Y_test_structure), dtype=dt)
        y_test_structured['event'] = Y_test_structure['event'].astype(bool)
        y_test_structured['time'] = Y_test_structure['time'].astype(float)
    except Exception as e:
        st.error(f"Erreur inattendue lors du formatage de Y_test: {e}")
        st.stop()

    # --- Étape 2 : Calculer le C-index de base sur le jeu de test ---
    try:
        predictions = chargement_modele.predict(X_test)

        events = y_test_structured['event']
        times = y_test_structured['time']

        c_index_baseline = concordance_index_censored(events, times, predictions)[0]
    except Exception as e:
        st.error(f"Erreur lors du calcul de la performance de base : {e}. "
                 f"Assurez-vous que X_test et Y_test sont bien formatés.")
        st.stop()

    # --- Étape 3 : Calcul des VIM par permutation ---
    importances = []
    feature_names = X_test.columns

    for i in range(X_test.shape[1]):
        X_permuted = X_test.copy()
        X_permuted.iloc[:, i] = np.random.permutation(X_permuted.iloc[:, i])

        pred_permuted = chargement_modele.predict(X_permuted)
        c_index_permuted = concordance_index_censored(events, times, pred_permuted)[0]

        importance = c_index_baseline - c_index_permuted
        importances.append(importance)

    # --- Étape 4 : Affichage des résultats et du graphique ---
    df_vim = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)

    st.success("Calcul terminé !")
    st.dataframe(df_vim)

    st.header("Représentation Graphique des VIM")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=df_vim, ax=ax)
    plt.title('Importance des variables par Permutation')
    plt.xlabel('Baisse du C-index')
    plt.ylabel('Variable')
    plt.tight_layout()
    st.pyplot(fig)

    # Chargement du CSS
    fichier_css = "style.css"
    with open(fichier_css) as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
