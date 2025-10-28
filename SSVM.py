import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
import shap


def ssvm(donnee_entre):
    # ==================================================================#
    # 1. Chargement du modèle et des données de test
    # ==================================================================#
    try:
        chargement_modele_ssvm = joblib.load("survival_svm.pkl")
        X_test = joblib.load("X_test_ssvm.pkl")
        Y_test = joblib.load("Y_test_ssvm.pkl")
    except FileNotFoundError as e:
        st.error(f"Erreur de chargement du fichier : {e}")
        return

    # ==================================================================#
    # 2. Préparation des données
    # ==================================================================#
    try:
        Y_test_structure = Surv.from_arrays(
            event=Y_test.astype(bool), 
            time=X_test['Tempsdesuivi (Mois)']
        )
        temps_max = X_test['Tempsdesuivi (Mois)'].max()
        X_test = X_test.drop('Tempsdesuivi (Mois)', axis=1)
    except KeyError:
        st.error("La colonne 'Tempsdesuivi (Mois)' est manquante dans X_test.")
        return

    # ==================================================================#
    # 3. Prédiction du score de risque
    # ==================================================================#
    try:
        score_risque_patient = chargement_modele_ssvm.predict(donnee_entre)[0]
        st.subheader("🔍 Score de risque estimé du patient")
        st.write(f"**Score de risque SSVM :** {score_risque_patient:.3f}")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        return

    
    # ==================================================================#
    # 4. Courbe de survie approximative
    # ==================================================================#
    """
    st.header("📈 Courbe de survie approximative")
    temps = np.linspace(0, temps_max, 200)
    survie_approx = np.exp(-np.exp(score_risque_patient) * temps / temps_max)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(temps, survie_approx, color='blue', lw=2)
    ax.set_xlabel("Temps (mois)")
    ax.set_ylabel("Probabilité de survie (approx.)")
    ax.set_title("Courbe de survie approximée - FastSurvivalSVM")
    st.pyplot(fig)

    """
    
    # ==================================================================#
    # 5. C-index et importance globale des variables
    # ==================================================================#
    try:
        dt = np.dtype([('event', 'bool'), ('time', 'float')])
        y_test_structured = np.empty(len(Y_test_structure), dtype=dt)
        y_test_structured['event'] = Y_test_structure['event']
        y_test_structured['time'] = Y_test_structure['time']

        predictions = chargement_modele_ssvm.predict(X_test)
        events = y_test_structured['event']
        times = y_test_structured['time']
        c_index_baseline = concordance_index_censored(events, times, predictions)[0]
       # st.write(f"📊 **C-index du modèle SSVM : {c_index_baseline:.3f}**")
    except Exception as e:
        st.error(f"Erreur lors du calcul du C-index : {e}")
        return

    importances = []
    feature_names = X_test.columns

    for i in range(X_test.shape[1]):
        X_permuted = X_test.copy()
        X_permuted.iloc[:, i] = np.random.permutation(X_permuted.iloc[:, i].values)
        pred_permuted = chargement_modele_ssvm.predict(X_permuted)
        c_index_permuted = concordance_index_censored(events, times, pred_permuted)[0]
        importances.append(c_index_baseline - c_index_permuted)

    df_vim = pd.DataFrame({'Variable': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)

    st.header(" Importance globale des variables (Permutation Importance)")
    fig_vim, ax_vim = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Variable', data=df_vim, ax=ax_vim, palette="viridis")
    ax_vim.set_title('Importance des variables (SSVM)')
    st.pyplot(fig_vim)

    # ==================================================================#
    # 6. Impact individuel des variables sur la survie du patient
    # ==================================================================#
    st.header("Impact individuel des variables sur la survie du patient")

    impact_vars = []
    base_risk = score_risque_patient
    features = donnee_entre.columns

    for feature in features:
        donnee_mod = donnee_entre.copy()

        # si binaire 0/1 : inversion
        if donnee_mod[feature].iloc[0] in [0, 1]:
            donnee_mod[feature] = 1 - donnee_mod[feature]
        else:
            # sinon : variation de +10 %
            donnee_mod[feature] = donnee_mod[feature] + 0.1 * donnee_mod[feature]

        new_risk = chargement_modele_ssvm.predict(donnee_mod)[0]
        variation = new_risk - base_risk

        impact_vars.append({
            "Variable": feature,
            "Variation_risque": variation,
            "Effet": "🔺 Diminue la survie" if variation > 0 else "🟩 Améliore la survie"
        })

    df_impact = pd.DataFrame(impact_vars).sort_values("Variation_risque", ascending=False)
    st.dataframe(df_impact.style.background_gradient(subset=['Variation_risque'], cmap='RdYlGn_r'))

    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
    colors = df_impact["Variation_risque"].apply(lambda x: "red" if x > 0 else "green")
    ax_imp.barh(df_impact["Variable"], df_impact["Variation_risque"], color=colors)
    ax_imp.set_xlabel("Variation du score de risque")
    ax_imp.set_ylabel("Variable")
    ax_imp.set_title("Impact de chaque variable sur la survie du patient")
    st.pyplot(fig_imp)

        # ==================================================================#
    # 5. INTERPRÉTATION SHAP (impact de chaque variable sur le risque)
    # ==================================================================#


    st.header("Interprétation SHAP : impact des variables sur le risque du patient")

    try:
        # Créer un explainer SHAP pour ton modèle SSVM
        explainer = shap.Explainer(chargement_modele_ssvm.predict, X_test)

        # Calculer les valeurs SHAP pour le patient courant
        shap_values = explainer(donnee_entre)

        # Afficher les valeurs SHAP sous forme de dataframe
        shap_df = pd.DataFrame({
            'Variable': X_test.columns,
            'Valeur_SHAP': shap_values.values[0],
            'Valeur_patient': donnee_entre.values[0]
        }).sort_values('Valeur_SHAP', ascending=False)

        # Interprétation directionnelle
        shap_df['Effet'] = shap_df['Valeur_SHAP'].apply(
            lambda x: "⬆️ Augmente le risque" if x > 0 else "⬇️ Diminue le risque"
        )

        st.dataframe(shap_df, use_container_width=True)

        # --- Graphique SHAP bar plot ---
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = shap_df['Valeur_SHAP'].apply(lambda x: 'red' if x > 0 else 'green')
        ax.barh(shap_df['Variable'], shap_df['Valeur_SHAP'], color=colors)
        ax.set_xlabel("Valeur SHAP (impact sur le score de risque)")
        ax.set_ylabel("Variable")
        ax.set_title("Impact des variables sur la prédiction du patient (SHAP)")
        plt.gca().invert_yaxis()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erreur lors du calcul des valeurs SHAP : {e}")

        # --- WATERFALL PLOT SHAP POUR UN PATIENT --- #
    """
    
    try:
        st.subheader("Interprétation détaillée : contribution cumulée des variables (Waterfall Plot)")

        # Créer la figure waterfall (force plot)
        shap_values_single = shap_values[0]

        # matplotlib backend pour Streamlit
        fig_wf = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values_single, show=False)
        st.pyplot(fig_wf)
    """
       # st.caption("""
        #🔍 **Interprétation :**
        #- Le point de départ (zéro) correspond au score moyen du modèle.
        #- Les barres rouges ➕ indiquent les variables qui **augmentent le risque**.
        #- Les barres bleues/vertes ➖ montrent celles qui **diminuent le risque**.
        #- La somme de tous ces effets donne le **score de risque final** du patient.
        #""")
    
    #except Exception as e:
    #    st.error(f"Erreur lors de la génération du graphique waterfall SHAP : {e}")
    