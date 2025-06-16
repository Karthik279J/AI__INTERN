import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix, mean_squared_error, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

st.markdown("""
    <h1 style='text-align: center; color: #f1c40f;'>🛡️ Cybersecurity Intrusions Dashboard</h1>
    <p style='text-align: center; color: #ecf0f1;'>Analyze, Visualize, and Understand Attacks</p>
    <hr style='border: 1px solid #555;' />
""", unsafe_allow_html=True)

def set_bg_hack(main_bg):
    main_bg_ext = "png"

    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{main_bg}) no-repeat center center fixed;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

import base64

def load_bg_img():
    with open("bgimg.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

set_bg_hack(load_bg_img())

st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: rgba(15, 32, 39, 0.6) !important;  /* 60% opacity dark */
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #fefce8 !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        background-color: rgba(255, 255, 255, 0.10) !important;
        color: #111;
        font-weight: bold;
        border-radius: 8px;
        transition: 0.3s;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(255, 255, 255, 1.0) !important;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)



#Start mapping the logic to built the interface

def main():
    st.sidebar.title("⚙️ Control Panel")
    st.sidebar.markdown("🪄 Start Performance")

    @st.cache_data(persist = True)
    def load(): #Data Loading
        data=pd.read_csv("cybersecurity_intrusion_data.csv")
        label = LabelEncoder() #you do with OneHotEncoder
        for i in data.columns:
            data[i] = label.fit_transform(data[i])
        return data
    df = load()#call the function
    if st.sidebar.checkbox("Display data",False):
        st.subheader("Data is displayed!")
        st.write(df)

    @st.cache_data(persist = True)
    def split(df):
        y = df['attack_detected']
        x = df.drop(columns=['attack_detected'])
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,
                                                         random_state = 42)
        return x_train,x_test,y_train,y_test
    x_train,x_test,y_train,y_test = split(df)

    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model,x_test,y_test, ax=ax)
            st.pyplot(fig)
        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model,x_test,y_test, ax=ax)
            st.pyplot(fig)
        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_estimator(model,x_test,y_test, ax=ax)
            st.pyplot(fig)
    class_names = ['non-attacked','attacked']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier",("Logistic Regression","Random Forest"))
    if classifier == "Logistic Regression":
        st.sidebar.subheader("Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)",0.01,10.0,
                                    step = 0.01,key="C_LR")
        max_iter = st.sidebar.slider("Maximum iterations",100,500,key="max_iter")
        metrics = st.sidebar.multiselect("What Metrics to plot?",
                                         ("Confusion Matrix","ROC Curve","Precision-Recall Curve"))
        if st.sidebar.button("Classify",key = "classify"):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C,max_iter =max_iter)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ",accuracy)
            st.write("Precision:",precision_score(y_test,y_pred,
                                                  labels = class_names))
            st.write("Recall:",recall_score(y_test,y_pred,
                                            labels = class_names))
            plot_metrics(metrics)
            
    if classifier == "Random Forest":
        st.sidebar.subheader("Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the \
                                               Forest",100,5000,step=10,key="n_estimators")
        max_depth = st.sidebar.number_input("The maximum depth of the \
                                            tree" ,1,20,step = 1,key="max_depth")
        metrics = st.sidebar.multiselect("What Metrics to plot?",
                                         ("Confusion Matrix","ROC Curve","Precision-Recall Curve"))
        if st.sidebar.button("Classify",key = "classify"):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators = n_estimators,
                                           max_depth = max_depth,n_jobs=-1)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ",accuracy)
            st.write("Precision:",precision_score(y_test,y_pred,
                                                  labels = class_names))
            st.write("Recall:",recall_score(y_test,y_pred,
                                            labels = class_names))
            plot_metrics(metrics)


                 
if __name__ == "__main__":
    main()
        