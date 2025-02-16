import streamlit as st
import numpy as np
import joblib
import pandas as pd
import pickle
from Build_LogisticRegression import LogisticRegression
from Build_RandomForestClassifier import RandomForestClassifier
from Build_SVC import SVC
from Build_PCA import PCA


def set_bg_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        .big-font {{
            font-size: 18px;
            margin-bottom: 0px;
        }}

        [data-testid="stSidebar"] {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )



set_bg_image("https://img.freepik.com/premium-vector/abstract-dna-structure-biotechnology-design-concept-with-hexagonal-texture-blue-background_858705-229.jpg")


# Page title
st.title("Lung Cancer Gene Expression Analysis")

st.sidebar.header("Lung cancer Analysis")
choice = st.sidebar.radio(    
            "Select an option:",  # Non-empty label for accessibility
            ["Prediction", "Model Analysis"],
            label_visibility="collapsed"  # This hides the label visually
        )


if choice=="Prediction":
    if "isTumor" not in st.session_state:
        st.session_state.isTumor = None

    st.markdown('<p class="big-font"> Choose a Classification Model: </p>', unsafe_allow_html=True)
    classification_model_choice = st.selectbox( "Select an option:", 
                                                ("None", "Logistic Regression", "Support Vector Classifier", "Random forest Classifier"),
                                                label_visibility="collapsed" )

    # File uploader for CSV input
    st.markdown('<p class="big-font"> Upload the sample gene expression to test (CSV file) </p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Select an option:", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        if classification_model_choice=="Logistic Regression":
            with open('models/logistic_regression_model.pkl', 'rb') as file:
                lr_model = pickle.load(file)
            if df.isnull().values.any():
                st.warning("The data contains missing values. Please handle them before proceeding.")
            else:
                input_array = df.to_numpy()
                #st.write(input_array)
                predictions = lr_model.predict(input_array)
                st.write(f'<p class="big-font"> Predicted output: {predictions[0]} </p>', unsafe_allow_html=True)
                if predictions[0]==0:
                    st.write('<p class="big-font"> The given sample is classified as <b>NORMAL</b> </p>', unsafe_allow_html=True)
                    isTumor = False
                elif predictions[0]==1:
                    st.write('<p class="big-font"> The given sample is classified as <b>TUMOR</b> </p>', unsafe_allow_html=True)
                    isTumor = True



        elif classification_model_choice=="Support Vector Classifier":
            with open('models/svc_model.pkl', 'rb') as file:
                svc_model = pickle.load(file)
            if df.isnull().values.any():
                st.warning("The data contains missing values. Please handle them before proceeding.")
            else:
                input_array = df.to_numpy()
                #st.write(input_array)
                predictions = svc_model.predict(input_array.reshape(1,-1))
                
                st.write(f'<p class="big-font"> Predicted output: {predictions[0]} </p>', unsafe_allow_html=True)
                if predictions[0]==0:
                    st.write('<p class="big-font"> The given sample is classified as <b>NORMAL</b> </p>', unsafe_allow_html=True)
                    isTumor = False
                elif predictions[0]==1:
                    st.write('<p class="big-font"> The given sample is classified as <b>TUMOR</b> </p>', unsafe_allow_html=True)
                    isTumor = True



        elif classification_model_choice=="Random forest Classifier":
            with open('models/random_forest_classifier.pkl', 'rb') as file:
                rfc_model = pickle.load(file)
                if df.isnull().values.any():
                    st.warning("The data contains missing values. Please handle them before proceeding.")
                else:
                    input_array = df.to_numpy()
                    #st.write(input_array)
                    predictions = rfc_model.predict(input_array.reshape(1, -1))

                    st.write(f'<p class="big-font"> Predicted output: {predictions[0]} </p>', unsafe_allow_html=True)
                    if predictions[0]==0:
                        st.write('<p class="big-font"> The given sample is classified as <b>NORMAL</b> </p>', unsafe_allow_html=True)
                        isTumor = False
                    elif predictions[0]==1:
                        st.write('<p class="big-font"> The given sample is classified as <b>TUMOR</b> </p>', unsafe_allow_html=True)
                        isTumor = True



        if isTumor==True:
            st.markdown('<br>', unsafe_allow_html=True)
            st.write("###### Do you want to know about the type of tumor?")
            clustering_needed = st.radio("Select an option", 
                                         ["NO", "YES"],
                                         label_visibility="collapsed")

            if clustering_needed == "YES":
                st.write("Using the K-means clustering model (k=2)...")
                
                pca_loaded = joblib.load('./models/pca_model.pkl')
                with open('./models/feature_names.pkl', 'rb') as f:
                    feature_names = pickle.load(f)
                with open('./models/selected_features.pkl', 'rb') as f:
                    selected_features = pickle.load(f)

                test = pd.DataFrame(input_array, columns=feature_names)
                test = test[selected_features]
                test = test.values.reshape(1, -1)
                test_pca = pca_loaded.transform(test)
                
                kmeans_loaded = joblib.load('./models/kmeans_model.pkl')
                predicted_labels = kmeans_loaded.predict(test_pca)
                st.write(f'<p class="big-font"> Predicted Label for new sample: Tumor of <b>type-{predicted_labels[0]}</b> </p>', unsafe_allow_html=True)


elif choice=="Model Analysis":
    st.subheader("Model Analysis")
    st.write("##### Classification model accuracies")
    st.image("./plots/accuracies.png", width=600)

    st.markdown('<br>', unsafe_allow_html=True)
    st.write("##### Elbow plot for Selecting k-value in kmeans clustering")
    st.image("./plots/elbow_method_plot.png", width=600)

    st.markdown('<br>', unsafe_allow_html=True)
    st.write("##### K-means Clustering")
    st.image("./plots/clustering.png", width=600)

    st.markdown('<br>', unsafe_allow_html=True)
    with open("./models/clustering_metrics.txt", "r") as file:
        lines = file.readlines()  # Read all lines into a list
        for line in lines:
            st.write(f"##### {line.strip()}") 
    

# To run the code:
# streamlit run Streamlit.py