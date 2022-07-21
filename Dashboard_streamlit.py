import warnings

import lightgbm

warnings.filterwarnings("ignore")
from sklearn.naive_bayes import GaussianNB as gnb
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import requests
import shap
import joblib
from lightgbm import LGBMClassifier
from urllib.request import urlopen
import plotly.express as px
st.set_option('deprecation.showPyplotGlobalUse', False)


st.write("""
	# Pret A Depenser - Credit Dashboard
	""")

#Chargement des dataset
PATH = r'C:\Users\nha5600\Desktop\BUREAU\FORMATION\Suite\P7\Predictions\\'
data = pd.read_csv(PATH+r'test_df_sample_.csv')
df = data.drop('TARGET', axis = 1)
df = df.drop('Unnamed: 0', axis = 1)
cust_list = df['SK_ID_CURR'].unique()[:10]
main_feat = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'PAYMENT_RATE', 'EXT_SOURCE_3']
print(df.columns.tolist())
description = pd.read_csv(PATH + 'HomeCredit_columns_description.csv',
                          usecols=['Row', 'Description'], \
                          index_col=0, encoding='unicode_escape')

print(df.shape)


#Chargement du modèle
load_clf = joblib.load(PATH+r"trained_model_sample.joblib")


@st.cache
def get_client_info(data, id_client):
    client_info = data[data['SK_ID_CURR'] == int(id_client)]
    return client_info


with st.sidebar:
    st.header("Prêt à dépenser")

    st.write("## ID Client")
    id_list = df["SK_ID_CURR"].tolist()
    id_client = st.selectbox(
        "Sélectionner l'identifiant du client", id_list)

    st.write("## Actions à effectuer")
    show_credit_decision = st.checkbox("Afficher la décision de crédit")
    show_client_details = st.checkbox("Afficher les informations du client")
    shap_general = st.checkbox("Afficher la feature importance globale")
    graphiques_bivariés = st.checkbox("Afficher les graphiques bivariés")
    recherche = st.checkbox("Afficher le positionnement du client")

    if (st.checkbox("Aide description des features")):
        list_features = description.index.to_list()
        list_features = list(dict.fromkeys(list_features))
        feature = st.selectbox('Sélectionner une variable', \
                               sorted(list_features))

        desc = description['Description'].loc[description.index == feature][:1]
        st.markdown('**{}**'.format(desc.iloc[0]))

    default_list = ["GENRE", "AGE", "STATUT FAMILIAL", "NB ENFANTS", "REVENUS", "MONTANT CREDIT"]

    infospersos = {
        'CODE_GENDER': "GENRE",
        'DAYS_BIRTH': "AGE",
        'DAYS_EMPLOYED': "NB ANNEES EMPLOI",
        'AMT_CREDIT': "MONTANT CREDIT",
        'AMT_ANNUITY': "MONTANT ANNUITES",
        'EXT_SOURCE_1': "EXT_SOURCE_1",
        'EXT_SOURCE_2': "EXT_SOURCE_2",
        'EXT_SOURCE_3': "EXT_SOURCE_3",
    }

    #Afficher l'ID Client sélectionné
    st.write("ID Client Sélectionné :", id_client)
    if (int(id_client) in id_list):
        client_info = get_client_info(df, id_client)

    if (show_client_details):
        st.header('Informations relatives au client')
        with st.spinner('Chargement des informations relatives au client...'):

            personal_info = client_info[list(infospersos.keys())]
            personal_info.rename(columns=infospersos, inplace=True)

            personal_info["AGE"] = int(round(personal_info["AGE"] / 365 * (-1)))
            personal_info["NB ANNEES EMPLOI"] = \
                int(round(personal_info["NB ANNEES EMPLOI"] / 365 * (-1)))

            filtered = st.multiselect("Choisir les informations à afficher", \
                                      options=list(personal_info.columns), \
                                      default='GENRE')
            df_info = personal_info[filtered]
            df_info['SK_ID_CURR'] = client_info['SK_ID_CURR']
            df_info = df_info.set_index('SK_ID_CURR')

            st.table(df_info.astype(str).T)
            show_all_info = st \
                .checkbox("Afficher toutes les informations (dataframe brute)")
            if (show_all_info):
                st.dataframe(client_info)

#Afficher la décision de crédit

if (show_credit_decision):
    st.header('Scoring et décision du modèle')

#Appel de l'API :
    API_url = "http://127.0.0.1:5000/credit/" + str(id_client)

#Prédiction du score du client
    with st.spinner('Chargement du score du client...'):
        json_url = urlopen(API_url)
        API_data = json.loads(json_url.read())
        classe_predite = API_data['prediction']
        if classe_predite == 1:
            decision = 'Crédit refusé'
        else:
            decision = 'Crédit accordé'
        proba = 1 - API_data['proba']

#Calcul du score du client
        client_score = round(proba * 100, 2)

        left_column, right_column = st.columns((1, 2))

        left_column.markdown('Risque de défaut: **{}%**'.format(str(client_score)))
        left_column.markdown('Seuil par défaut du modèle: **50%**')

        if classe_predite == 1:
            left_column.markdown(
                'Décision: <span style="color:red">**{}**</span>'.format(decision), \
                unsafe_allow_html=True)
        else:
            left_column.markdown(
                'Décision: <span style="color:green">**{}**</span>' \
                    .format(decision), \
                unsafe_allow_html=True)

# Afficher la feature importance globale
Feature_imp = PATH+r'Feature_importance.png'
with open(PATH+'val_file.pkl', 'rb') as f:
    shap_values = joblib.load(f)

if (shap_general):
    st.write('Le graphique suivant indique les variables ayant le plus contribué au modèle.')
    beeswarm = shap.plots.beeswarm(shap_values, max_display=10)
    st.pyplot(beeswarm)

    st.button("Recommencer")

# Affichage de graphique bivariés avec regression linéaire
if (graphiques_bivariés):
    features = st.multiselect("Choisissez deux variables", list(df.columns))
    if len(features) != 2 :
        st.error("Sélectionnez deux variables")
    else :
        st.write("## Graphique bi-varié")
        # Graphique
        chart = sns.jointplot(x=df[features[0]], y=df[features[1]], height=10)
        # Regression linéaire
        sns.regplot(x=df[features[0]], y=df[features[1]], scatter=False, ax=chart.ax_joint)
        st.pyplot(chart)
    st.button("Recommencer")

# Mise en évidence du profil du client (et comparaison avec les autres)
if (recherche):
    features = st.multiselect("Choisissez deux variables", list(df.columns))
    if len(features) != 2 :
        st.error("Sélectionnez deux variables")
    else :
        profile_ID = st.multiselect("Choisissez un ou plusieurs profils à mettre en évidence", list(data['SK_ID_CURR']), default = 398791)
        temp_data_chart = df
        temp_data_chart['HIGHLIGHT'] = temp_data_chart['SK_ID_CURR'].apply(lambda x : True if x in profile_ID else False)
        st.write("## Graphique interactif avec profils choisis en couleur")
        # Graphique
        chart = px.scatter(temp_data_chart, x=features[0], y=features[1], color='HIGHLIGHT')
        st.plotly_chart(chart)
    st.button("Recommencer")