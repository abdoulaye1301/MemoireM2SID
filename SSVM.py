import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sksurv.metrics import concordance_index_censored
import numpy as np
from sksurv.util import Surv
#from sksurv.nonparametric import kaplan_meier
# from sksurv.linear_model import CoxPHSurvivalAnalysis # Pour l'entraînement de la base de risque

# Données d'entrée pour le patient (doit être un DataFrame à une ligne)
# donnee_entre = ... (sera fourni par l'application Streamlit)

def ssvm(donnee_entre):
    # ==================================================================#
    # 1. Chargement des modèles et des données de test
    # ==================================================================#
    try:
        # Modèle SSVM (prédit le score de risque)
        chargement_modele_ssvm = joblib.load("survival_svm.pkl")
        # Modèle CPH pour la base de risque (ou juste la base de risque précalculée)
        # ASSUREZ-VOUS QUE CE FICHIER EXISTE
        chargement_modele_cph = joblib.load("cox_ph_baseline.pkl")
        
        X_test = joblib.load("X_test.pkl")
        Y_test = joblib.load("Y_test.pkl")
        
    except FileNotFoundError as e:
        st.error(f"Erreur de chargement du fichier : {e}. Assurez-vous que 'survival_svm.pkl', 'cox_ph_baseline.pkl', 'X_test.pkl' et 'Y_test.pkl' existent.")
        return
    
    # Préparation des données de test
    try:
        # Créez un tableau structuré pour le temps de survie et l'événement
        # Y_test_structure sera utilisé pour l'évaluation (C-index)
        Y_test_structure = Surv.from_arrays(event=Y_test.astype(bool), time=X_test['Tempsdesuivi (Mois)'])
        X_test = X_test.drop('Tempsdesuivi (Mois)', axis=1)
        
    except KeyError:
        st.error("La colonne 'Tempsdesuivi (Mois)' est manquante. Impossible de préparer Y_test_structure.")
        return

    # ==================================================================#
    # 2. Prédiction des scores et fonction de survie du patient
    # ==================================================================#
    
    # Prédiction du score de risque par le SSVM pour le nouveau patient
    try:
        # SSVM prédit le score de risque (plus le score est élevé, plus le risque est grand)
        score_risque_patient = chargement_modele_ssvm.predict(donnee_entre)[0]
    except Exception as e:
        st.error(f"Erreur lors de la prédiction du score de risque SSVM : {e}")
        return

    # Base de survie S0(t) obtenue à partir du modèle CPH (ou précalculée)
    # Note : Le modèle CPH fournit directement la fonction de survie de base (S0)
    # Les temps auxquels S0(t) est évaluée et les probabilités S0(t)
    base_survie = chargement_modele_cph.baseline_survival_function_
    temps = base_survie.x
    s0_t = base_survie.y
    
    # Calcul du risque relatif (Hazard Ratio)
    # L'interprétation du score SSVM comme un risque relatif est une approximation
    # pour permettre l'estimation de la fonction de survie avec Kalbfleisch-Prentice.
    risque_relatif = np.exp(score_risque_patient)
    
    # Calcul de la fonction de survie du patient : S(t|x) = S0(t)^(exp(score))
    # S_patient est la probabilité de survie à chaque temps 'temps'
    s_patient = s0_t ** risque_relatif

    # ==================================================================#
    # 3. GRAPHIQUE DE SURVIE DU PATIENT
    # ==================================================================#
    st.header("Fonction de survie prédite du patient (via SSVM Score et Base CPH)")
    fig, ax = plt.subplots(figsize=(10,6))

    # Courbe de survie du patient
    ax.step(temps, s_patient, where="post", label=f"Patient (Score de Risque: {score_risque_patient:.2f})", color='red')
    
    # Courbe de survie de base (HR=1)
    ax.step(temps, s0_t, where="post", label="Survie de Base (HR=1)", linestyle='--', color='gray')

    ax.set_xlabel("Temps de survie (mois)")
    ax.set_ylabel("Probabilité de survie")
    ax.set_title("Fonction de survie prédite du patient")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown(f"**Score de Risque (SSVM) :** `{score_risque_patient:.3f}` (Plus le score est haut, plus le risque est élevé)")
    
    # ==================================================================#
    # 4. GRAPHIQUE DE SURVIE AVEC IMPACT DE CHAQUE VARIABLE
    # ==================================================================#
    st.header("Fonction de survie avec impact des variables")
    fig2, ax2 = plt.subplots(figsize=(12,8))

    # Fonction de survie du patient original
    ax2.step(temps, s_patient, where="post", label="Patient original", color='red', linewidth=3, linestyle='-')

    # Impact des variables (on inverse les binaires pour voir effet)
    variables = donnee_entre.columns
    for var in variables:
        if donnee_entre[var].iloc[0] in [0, 1]: # Seulement pour les variables binaires
            patient_var = donnee_entre.copy()
            # Inversion 0/1 pour voir l'impact de l'absence/présence de la condition
            patient_var[var] = 1 - patient_var[var].iloc[0] 
            
            # Nouveau score SSVM prédit
            score_var = chargement_modele_ssvm.predict(patient_var)[0]
            risque_relatif_var = np.exp(score_var)
            
            # Nouvelle fonction de survie
            s_var = s0_t ** risque_relatif_var
            
            ax2.step(temps, s_var, where="post", label=f"{var} (Modifié à {patient_var[var].iloc[0]}) - Score: {score_var:.2f}", linestyle='--')
        # On pourrait ajouter un traitement pour les variables continues (e.g., +1 SD / -1 SD)

    ax2.set_xlabel("Temps de survie (mois)")
    ax2.set_ylabel("Probabilité de survie")
    ax2.set_title("Impact de la modification des variables sur la Survie")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    st.pyplot(fig2)

    # Le reste du code pour le VIM (Permutation Importance) est conservé et utilise le score SSVM
    # ==================================================================#
    # INTERPRÉTATION DU MODÈLE AVEC VIM
    # (Le code VIM ci-dessous est conservé de votre version précédente et est fonctionnel avec le SSVM)
    # ==================================================================#
    
    st.header("Calcul de l'importance des variables (VIM)")

    # --- Étape 1 : Conversion de Y_test en tableau structuré (vérification) ---
    try:
        dt = np.dtype([('event', 'bool'), ('time', 'float')])
        y_test_structured = np.empty(len(Y_test_structure), dtype=dt)
        y_test_structured['event'] = Y_test_structure['event'].astype(bool)
        y_test_structured['time'] = Y_test_structure['time'].astype(float)
    except Exception as e:
        st.error(f"Erreur inattendue lors du formatage de Y_test pour VIM: {e}")
        return

    # --- Étape 2 : Calculer le C-index de base sur le jeu de test ---
    try:
        predictions = chargement_modele_ssvm.predict(X_test)
        events = y_test_structured['event']
        times = y_test_structured['time']
        c_index_baseline = concordance_index_censored(events, times, predictions)[0]
        st.write(f"C-index de base du modèle SSVM : {c_index_baseline:.3f}")
    except Exception as e:
        st.error(f"Erreur lors du calcul de la performance de base : {e}.")
        return

    # --- Étape 3 : Calcul des VIM par permutation ---
    importances = []
    feature_names = X_test.columns

    for i in range(X_test.shape[1]):
        X_permuted = X_test.copy()
        X_permuted.iloc[:, i] = np.random.permutation(X_permuted.iloc[:, i].values)
        pred_permuted = chargement_modele_ssvm.predict(X_permuted)
        c_index_permuted = concordance_index_censored(events, times, pred_permuted)[0]
        importance = c_index_baseline - c_index_permuted
        importances.append(importance)

    # --- Étape 4 : Affichage des résultats et du graphique ---
    df_vim = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)

    st.header("Représentation Graphique des VIM")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=df_vim, ax=ax3)

    for p in ax3.patches:
        width = p.get_width()
        ax3.text(width + 0.001, 
                 p.get_y() + p.get_height() / 2,
                 f'{width:.3f}', 
                 va='center')

    plt.title('Importance des variables par Permutation (SSVM)')
    plt.xlabel('Baisse du C-index')
    plt.ylabel('Variable')
    plt.tight_layout()
    st.pyplot(fig3)