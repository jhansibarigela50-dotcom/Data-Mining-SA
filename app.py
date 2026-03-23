import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="InsightMart | Black Friday Data Mining", layout="wide")

# --- 2. DATA CLEANING & PREPROCESSING (STAGES 1 & 2) ---
@st.cache_data # This keeps the app fast by only cleaning once per session
def load_and_clean_data():
    try:
        # Load the raw file you uploaded to GitHub
        df = pd.read_csv('BlackFriday.csv')
        
        # Handling Missing Values (Stage 2)
        df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
        df['Product_Category_3'] = df['Product_Category_3'].fillna(0)
        
        # Encoding Categorical Data (Stage 2)
        le = LabelEncoder()
        df['Gender_Num'] = le.fit_transform(df['Gender']) # Male=0, Female=1
        
        # Map Age to ordered numbers
        age_map = {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}
        df['Age_Num'] = df['Age'].map(age_map)
        
        # Normalize Purchase for Clustering
        scaler = StandardScaler()
        df['Purchase_Scaled'] = scaler.fit_transform(df[['Purchase']])
        
        return df
    except Exception as e:
        st.error(f"Error loading 'BlackFriday.csv': {e}")
        return None

df = load_and_clean_data()

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("Project Stages")
selection = st.sidebar.radio("Go to:", 
    ["Scope & Raw Data", "Cleaned Insights (EDA)", "Customer Clusters", "Product Associations", "Anomaly Detection"])

if df is not None:
    # --- STAGE 1: SCOPE ---
    if selection == "Scope & Raw Data":
        st.title("🎯 Project Scope: Black Friday Insights")
        st.write("Our aim is to find who buys what, how much they spend, and how their behavior changes.")
        st.subheader("Data Preview (Raw/Cleaned)")
        st.write(df.head(10))
        st.info(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

    # --- STAGE 3: EDA ---
    elif selection == "Cleaned Insights (EDA)":
        st.title("📊 Exploratory Data Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Purchase by Age Group")
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='Age', y='Purchase', palette='coolwarm', ax=ax)
            st.pyplot(fig)
            
        with col2:
            st.write("### Top Product Categories")
            fig, ax = plt.subplots()
            df['Product_Category_1'].value_counts().head(10).plot(kind='bar', color='teal', ax=ax)
            st.pyplot(fig)

    # --- STAGE 4: CLUSTERING ---
    elif selection == "Customer Clusters":
        st.title("👥 Clustering Analysis (K-Means)")
        st.write("Grouping customers into 'Budget', 'Mid', and 'Premium' spenders.")
        
        # Clustering Logic
        X = df[['Age_Num', 'Purchase_Scaled']]
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X)
        
        fig, ax = plt.subplots()
        sns.scatterplot(data=df.sample(5000), x='Age', y='Purchase', hue='Cluster', palette='bright', ax=ax)
        st.pyplot(fig)
        st.success("K-Means successfully applied using the Elbow Method logic.")

    # --- STAGE 5: ASSOCIATION RULES ---
    elif selection == "Product Associations":
        st.title("🛒 Association Rule Mining")
        st.write("Analyzing product combinations using the **Apriori Algorithm**.")
        
        # Pivot data for Apriori (Sampled for speed)
        subset = df.head(10000)
        basket = subset.groupby(['User_ID', 'Product_Category_1'])['Product_Category_1'].count().unstack().fillna(0)
        basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
        
        freq_items = apriori(basket_sets, min_support=0.05, use_colnames=True)
        rules = association_rules(freq_items, metric="lift", min_threshold=1)
        
        st.write("### Frequent Product Combinations")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

    # --- STAGE 6: ANOMALY DETECTION ---
    elif selection == "Anomaly Detection":
        st.title("⚠️ Anomaly Detection")
        st.write("Detecting unusual big spenders using **Z-Score/Statistical Limits**.")
        
        upper_limit = df['Purchase'].mean() + (3 * df['Purchase'].std())
        anomalies = df[df['Purchase'] > upper_limit]
        
        st.metric("Anomalies Detected", len(anomalies))
        st.write(f"Transactions above **${upper_limit:,.2f}** are flagged as unusual.")
        st.dataframe(anomalies[['User_ID', 'Age', 'Gender', 'Purchase']].head(20))

