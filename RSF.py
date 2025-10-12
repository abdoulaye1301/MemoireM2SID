import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sksurv.metrics import concordance_index_censored
import numpy as np
from sksurv.util import Surv

# from PIL import Image
donnee_entre = []
def rsf(donnee_entre):
    # Récupération de la première ligne (nouveau patient)

    # Importation du modèle et des données de test
    chargement_modele = joblib.load("random_survival_forest.pkl")
    X_test = joblib.load("X_test.pkl")
    Y_test = joblib.load("Y_test.pkl")

    #==================================================================#
    # Créer un tableau structuré pour le temps de survie et l'événement
    Y_test_structure = Surv.from_arrays(event=Y_test.astype(bool), time=X_test['Tempsdesuivi (Mois)'])

    # Supprimer la colonne 'Tempsdesuivi (Mois)' des caractéristiques X car elle est maintenant dans Y
    X_test = X_test.drop('Tempsdesuivi (Mois)', axis=1)



    #==================================================================#
    # GRAPHIQUE DE SURVIE DU PATIENT
   # st.header("Fonction de survie prédite du patient")
   # fig, ax = plt.subplots(figsize=(10,6))

#    sf_patient = chargement_modele.predict_survival_function(donnee_entre)
 #   for i, sf in enumerate(sf_patient):
  #      ax.step(sf.x, sf.y, where="post", label="Patient")

  #  ax.set_xlabel("Temps de survie (mois)")
 #   ax.set_ylabel("Probabilité de survie")
  #  ax.set_title("Fonction de survie prédite du patient")
   # ax.legend()
  #  plt.tight_layout()
   # st.pyplot(fig)



    #==================================================================#
    # GRAPHIQUE DE SURVIE AVEC IMPACT DE CHAQUE VARIABLE
    #st.header("Fonction de survie avec impact des variables")
    fig, ax = plt.subplots(figsize=(10,6))

    # Fonction de survie du patient original
    sf_base = chargement_modele.predict_survival_function(donnee_entre)
    for i, sf in enumerate(sf_base):
        ax.step(sf.x, sf.y, where="post", label="Patient original")

    # Impact des variables (on inverse les binaires pour voir effet)
    variables = donnee_entre.columns
    for var in variables:
        patient_var = donnee_entre.copy()
        patient_var[var] = 1 - patient_var[var].iloc[0]  # inversion 0/1
        sf_var = chargement_modele.predict_survival_function(patient_var)
        for i, sf in enumerate(sf_var):
            ax.step(sf.x, sf.y, where="post", label=f"{var} modifié")

    ax.set_xlabel("Temps de survie (mois)")
    ax.set_ylabel("Probabilité de survie")
    ax.set_title("Fonction de survie avec impact des variables sur la survie")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)
    #st.write("**Note :** Chaque courbe représente l'impact de la modification d'une variable binaire sur la fonction de survie du patient.")
    #st.write("Par exemple, si la courbe diminue plus rapidement lorsque 'Cardiopathie' est modifié, cela suggère que la présence de cardiopathie réduit la survie.")
    #st.write("Inversement, si la courbe diminue plus lentement, cela suggère que l'absence de cette condition améliore la survie.")
    #st.write("Cela permet de visualiser l'importance relative de chaque variable sur la survie du patient.")
    #st.write("Cependant, pour une interprétation plus rigoureuse de l'importance des variables, veuillez vous référer à la section suivante sur l'importance des variables par permutation (VIM).")
    # ==================================================================#
    # INTERPRÉTATION DU MODÈLE AVEC VIM
    #st.header("Calcul de l'importance des variables")
    #st.write("Calcul en cours, veuillez patienter...")

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

   # st.success("Calcul terminé !")
    #st.dataframe(df_vim)

    st.header("Représentation Graphique des VIM")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=df_vim, ax=ax)

    # Ajouter les valeurs au bout de chaque barre
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.001,  # un petit décalage
                p.get_y() + p.get_height() / 2,
                f'{width:.3f}',  # 3 décimales
                va='center')

    plt.title('Importance des variables par Permutation')
    plt.xlabel('Baisse du C-index')
    plt.ylabel('Variable')
    plt.tight_layout()
    st.pyplot(fig)