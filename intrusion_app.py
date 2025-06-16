import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st
import base64
import io


st.set_page_config(page_title="Cybersecurity Intrusions Dashboard", layout="wide")

# Background styling
st.markdown("""
    <style>
    .stApp {
        background: url(data:image/png;base64,""" + base64.b64encode(open("bgimg.jpg", "rb").read()).decode() + """) no-repeat center center fixed;
        background-size: cover;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(15, 32, 39, 0.6) !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    [data-testid="stSidebar"] * {
        color: #fefce8 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align: center; color: #f1c40f;'>🛡 Cybersecurity Intrusions Dashboard</h1>
    <p style='text-align: center; color: #ecf0f1;'>Analyze, Visualize, and Understand Attacks</p>
    <hr style='border: 1px solid #555;' />
""", unsafe_allow_html=True)

# Load and preprocess dataset

od = pd.read_csv("cybersecurity_intrusion_data.csv")
od.drop(columns=['session_id'], inplace=True, errors='ignore')
@st.cache_data
def load_data():
    df = pd.read_csv("cybersecurity_intrusion_data.csv")
    
    df.fillna(method='ffill', inplace=True)
    df['unusual_time_access'] = df['unusual_time_access'].astype(bool)
    df['attack_detected'] = df['attack_detected'].astype(bool)
    df['packet_per_second'] = df['network_packet_size'] / df['session_duration'].replace(0, 0.001)
    df['login_failure_ratio'] = df['failed_logins'] / df['login_attempts'].replace(0, 1)
    df['encryption_strength'] = df['encryption_used'].map({'None': 0, 'DES': 1, 'AES': 2})
    df['reputation_risk'] = 1 - df['ip_reputation_score']
    label_cols = ['protocol_type', 'browser_type', 'encryption_used']
    for col in label_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

df = load_data()
X = df.drop(['session_id', 'attack_detected'], axis=1)
y = df['attack_detected']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Sidebar controls
st.sidebar.title("📂 Sections")
section = st.sidebar.radio("Navigate", [
    "Data Overview", "Visualizations", "Model: Logistic Regression",
    "Model: Polynomial Regression", "Model: Random Forest", "KMeans Clustering", "Prediction Panel"])

# Data Overview
if section == "Data Overview":
    st.subheader("🧾 Data Preview")
    st.write(od)
    st.subheader('Data Description')
    st.dataframe(df.describe(include='all'))
    st.write("Shape:", df.shape)
    st.write("Missing Values:")
    st.dataframe(df.isnull().sum().reset_index().rename(columns={0: 'Missing Count', 'index': 'Column'}))

elif section == "Visualizations":
    st.subheader("📊 Visualization Section")

    viz_option = st.sidebar.selectbox(
        "Choose a visualization:",
        ("Attack Distribution","login_attempts by Attack", "Protocol Types by Attack", "Packet Size vs Duration", "Feature Correlation Heatmap","encryption_used by Attack")
    )

    if viz_option == "Attack Distribution":
        st.subheader("📊 Attack Distribution")
        fig1 = px.pie(df, names='attack_detected', title='Attack vs Normal')
        st.plotly_chart(fig1, use_container_width=True, key="pie")

    elif viz_option == "Protocol Types by Attack":
        st.subheader("📊 Protocol Types by Attack")
        fig2 = px.histogram(df, x='protocol_type', color='attack_detected', title='Protocols by Attack')
    #now i want to replace 0,1,2 with TCP, UDP, ICMP
        fig2.update_xaxes(tickvals=[0, 1, 2], ticktext=['TCP', 'UDP', 'ICMP'])
        st.plotly_chart(fig2, use_container_width=True, key="hist")
    elif viz_option == "encryption_used by Attack":
        st.subheader("📊 Encryption Used by Attack")
        fig2 = px.histogram(df, x='encryption_used', color='attack_detected', title='Encryption Used by Attack')
        fig2.update_xaxes(tickvals=[0, 1, 2], ticktext=['None', 'DES', 'AES'])
        st.plotly_chart(fig2, use_container_width=True, key="hist_encryption")
    elif viz_option =="login_attempts by Attack":
        st.subheader("📊 Login Attempts by Attack")
        fig2 = px.histogram(df, x='failed_logins', color='attack_detected', title='Login Attempts by Attack')
        st.plotly_chart(fig2, use_container_width=True, key="hist_login")

    elif viz_option == "Packet Size vs Duration":
        st.subheader("📊 Packet Size vs Duration")
        fig3 = px.scatter(df, x='session_duration', y='network_packet_size', color='attack_detected')
        st.plotly_chart(fig3, use_container_width=True, key="scatter")

    elif viz_option == "Feature Correlation Heatmap":
        st.subheader("📊 Feature Correlation Heatmap")
        corr_matrix = df.drop('session_id', axis=1).corr()
        fig4 = px.imshow(corr_matrix, text_auto=True, aspect="auto")
        st.plotly_chart(fig4, use_container_width=True, key="heatmap")


# Logistic Regression
elif section == "Model: Logistic Regression":
    st.subheader("📈 Logistic Regression")
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)

    st.text(classification_report(y_test, y_pred))
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, x=['Normal', 'Attack'], y=['Normal', 'Attack'])
    st.plotly_chart(fig_cm, use_container_width=True, key="log_cm")
#kMeans Clustering for elbow method kneed for user input
elif section == "KMeans Clustering":
    st.subheader("🤖 KMeans Clustering")
    from sklearn.cluster import KMeans
    from kneed import KneeLocator

    kmeans = KMeans(random_state=42)
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans.set_params(n_clusters=k)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    kn = KneeLocator(k_range, inertia, curve='convex', direction='decreasing')
    optimal_k = kn.elbow
    st.write(f"Optimal number of clusters: {optimal_k}")
    fig_kmeans = px.line(x=k_range, y=inertia, labels={'x': 'Number of Clusters', 'y': 'Inertia'})
    fig_kmeans.add_vline(x=optimal_k, line_dash="dash", line_color="red", annotation_text="Optimal K")
    st.plotly_chart(fig_kmeans, use_container_width=True, key="kmeans_elbow")
    # Polynomial Regression
elif section == "Model: Polynomial Regression":
    st.subheader("🧮 Polynomial Logistic Regression")
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    poly_model = LogisticRegression(max_iter=1000)
    poly_model.fit(X_train_poly, y_train)
    y_poly_pred = poly_model.predict(X_test_poly)

    st.text(classification_report(y_test, y_poly_pred))
    st.write(f"Accuracy: {accuracy_score(y_test, y_poly_pred):.2f}")

    cm_poly = confusion_matrix(y_test, y_poly_pred)
    fig_poly = px.imshow(cm_poly, text_auto=True, x=['Normal', 'Attack'], y=['Normal', 'Attack'])
    st.plotly_chart(fig_poly, use_container_width=True, key="poly_cm")

# Random Forest
elif section == "Model: Random Forest":
    st.subheader("🌲 Random Forest Classifier")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_rf_pred = rf.predict(X_test)

    st.text(classification_report(y_test, y_rf_pred))
    st.write(f"Accuracy: {accuracy_score(y_test, y_rf_pred):.2f}")

    cm_rf = confusion_matrix(y_test, y_rf_pred)
    fig_rf = px.imshow(cm_rf, text_auto=True, x=['Normal', 'Attack'], y=['Normal', 'Attack'])
    st.plotly_chart(fig_rf, use_container_width=True, key="rf_cm")

    st.subheader("📊 Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    fig_imp = px.bar(feature_importance, x='Importance', y='Feature', orientation='h')
    st.plotly_chart(fig_imp, use_container_width=True, key="rf_importance")

# Prediction Panel
elif section == "Prediction Panel":
    st.subheader("🎯 Predict Intrusion Based on Input")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            network_packet_size = st.number_input("Packet Size", min_value=0.0)
            protocol_type = st.selectbox("Protocol", options=[('TCP', 0), ('UDP', 1), ('ICMP', 2)], format_func=lambda x: x[0])
            login_attempts = st.number_input("Login Attempts", min_value=0)
            session_duration = st.number_input("Session Duration (seconds)", min_value=0.0)
            encryption_used = st.selectbox("Encryption", options=[('None', 0), ('DES', 1), ('AES', 2)], format_func=lambda x: x[0])

        with col2:
            ip_reputation_score = st.slider("IP Reputation Score", 0.0, 1.0, 0.5, 0.01)
            failed_logins = st.number_input("Failed Logins", min_value=0)
            browser_type = st.selectbox("Browser", options=[('Chrome', 0), ('Firefox', 1), ('Edge', 2), ('Safari', 3), ('Unknown', 4)], format_func=lambda x: x[0])
            unusual_time_access = st.checkbox("Unusual Time Access")

        submitted = st.form_submit_button("Predict Attack")

    if submitted:
        user_data = pd.DataFrame({
            'network_packet_size': [network_packet_size],
            'protocol_type': [protocol_type[1]],
            'login_attempts': [login_attempts],
            'session_duration': [session_duration],
            'encryption_used': [encryption_used[1]],
            'ip_reputation_score': [ip_reputation_score],
            'failed_logins': [failed_logins],
            'browser_type': [browser_type[1]],
            'unusual_time_access': [int(unusual_time_access)],
            'packet_per_second': [network_packet_size / max(session_duration, 0.001)],
            'login_failure_ratio': [failed_logins / max(login_attempts, 1)],
            'encryption_strength': [encryption_used[1]],
            'reputation_risk': [1 - ip_reputation_score]
        })

             # Shared model setup
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(X_train, y_train)

        poly = PolynomialFeatures(degree=2)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        poly_model = LogisticRegression(max_iter=1000)
        poly_model.fit(X_train_poly, y_train)

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)


        user_data_scaled = scaler.transform(user_data)
        lr_pred = log_reg.predict(user_data_scaled)[0]
        lr_prob = log_reg.predict_proba(user_data_scaled)[0][1]
        poly_user_scaled = poly.transform(user_data_scaled)
        poly_pred = poly_model.predict(poly_user_scaled)[0]
        poly_prob = poly_model.predict_proba(poly_user_scaled)[0][1]
        rf_pred = rf.predict(user_data_scaled)[0]
        rf_prob = rf.predict_proba(user_data_scaled)[0][1]

        st.subheader("Model Predictions")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Logistic Regression", "Attack" if lr_pred else "Normal", f"{lr_prob:.2%}")
        with col2:
            st.metric("Polynomial Regression", "Attack" if poly_pred else "Normal", f"{poly_prob:.2%}")
        with col3:
            st.metric("Random Forest", "Attack" if rf_pred else "Normal", f"{rf_prob:.2%}")

        final_vote = sum([lr_pred, poly_pred, rf_pred]) >= 2
        if final_vote:
            st.error("⚠ Final Verdict: ATTACK DETECTED!")
        else:
            st.success("✅ Final Verdict: Normal traffic")
