import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules

# --- 1. SETTINGS & DATA LOADING ---
st.set_page_config(page_title="InsightMart | Black Friday AI", layout="wide")

@st.cache_data
def load_and_clean():
    try:
        df = pd.read_csv('BlackFriday.csv')
        # Stage 2: Cleaning
        df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
        df['Product_Category_3'] = df['Product_Category_3'].fillna(0)
        # Stage 2: Encoding
        le = LabelEncoder()
        df['Gender_Num'] = le.fit_transform(df['Gender'])
        age_map = {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}
        df['Age_Num'] = df['Age'].map(age_map)
        # Stage 2: Scaling
        scaler = StandardScaler()
        df['Purchase_Scaled'] = scaler.fit_transform(df[['Purchase']])
        return df
    except:
        return None

df = load_and_clean()

# --- 2. SIDEBAR NAVIGATION ---
st.sidebar.title("Project Stages")
page = st.sidebar.radio("Navigate to:", 
    ["1. Project Scope", "2. Data Preprocessing", "3. Market EDA", "4. Customer Clustering", "5. Product Associations", "6. Anomaly Detection"])

if df is not None:
    # --- STAGE 1: PROJECT SCOPE ---
    if page == "1. Project Scope":
        st.title("🎯 Stage 1: Define Project Scope")
        
        # Using columns to organize the "Game Plan"
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Objective")
            st.write("""
            The aim is to analyze the Black Friday dataset to understand:
            * **Who** buys what (Demographics).
            * **How much** they spend (Purchase patterns).
            * **Product associations** for better combo offers.
            """)
        with col2:
            st.subheader("Target Outcomes")
            st.success("✔️ Identify Shopping Behaviors")
            st.success("✔️ Group Customers into Clusters")
            st.success("✔️ Detect Unusual Big Spenders")

        st.divider()
        st.subheader("Raw Dataset Preview")
        # Showing only the main columns to keep it organized
        main_cols = ['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category', 'Purchase']
        st.dataframe(df[main_cols].head(10), use_container_width=True)

    # --- STAGE 2: PREPROCESSING ---
    elif page == "2. Data Preprocessing":
        st.title("🧼 Stage 2: Data Cleaning & Preprocessing")
        
        st.write("To make the data ready for AI models, we performed the following:")
        c1, c2, c3 = st.columns(3)
        c1.metric("Missing Values Handled", "Cat 2 & 3")
        c2.metric("Encoding", "Gender & Age")
        c3.metric("Normalization", "StandardScaler")
        
        st.subheader("Cleaned & Engineered Features")
        st.write("This table shows the numerical transformations used for the AI models:")
        # Organized technical view
        tech_view = df[['User_ID', 'Gender_Num', 'Age_Num', 'Purchase_Scaled']].head(10)
        st.dataframe(tech_view, use_container_width=True)

    # --- STAGE 3: EDA ---
    elif page == "3. Market EDA":
        st.title("📊 Stage 3: Exploratory Data Analysis")
        
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.write("### Purchase by Gender")
            fig, ax = plt.subplots()
            sns.barplot(data=df, x='Gender', y='Purchase', ax=ax, palette='magma')
            st.pyplot(fig)
        with row1_col2:
            st.write("### Popular Categories")
            fig, ax = plt.subplots()
            df['Product_Category_1'].value_counts().head(5).plot(kind='pie', autopct='%1.1f%%', ax=ax)
            st.pyplot(fig)

    # --- STAGE 4: CLUSTERING ---
    elif page == "4. Customer Clustering":
        st.title("👥 Stage 4: Clustering Analysis")
        
        # K-Means logic
        X = df[['Age_Num', 'Purchase_Scaled']]
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X)
        
        # Label Clusters
        cluster_labels = {0: "Budget Shoppers", 1: "Premium Buyers", 2: "Average Spenders"}
        df['Segment'] = df['Cluster'].map(cluster_labels)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df.sample(2000), x='Age', y='Purchase', hue='Segment', ax=ax)
        st.pyplot(fig)

    # --- STAGE 5: ASSOCIATIONS ---
    elif page == "5. Product Associations":
        st.title("🛒 Stage 5: Association Rule Mining")
        # Logic for Apriori
        subset = df.head(5000)
        basket = subset.groupby(['User_ID', 'Product_Category_1'])['Product_Category_1'].count().unstack().fillna(0)
        basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
        
        freq_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)
        rules = association_rules(freq_itemsets, metric="lift", min_threshold=1)
        
        st.write("Top Product Combinations Found:")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10), use_container_width=True)

    # --- STAGE 6: ANOMALIES ---
    elif page == "6. Anomaly Detection":
        st.title("⚠️ Stage 6: Anomaly Detection")
        
        limit = df['Purchase'].mean() + (2 * df['Purchase'].std())
        anomalies = df[df['Purchase'] > limit]
        
        st.metric("Unusual High Spenders Found", len(anomalies))
        st.write(f"Transactions above **${limit:,.2f}** are flagged as anomalies.")
        st.dataframe(anomalies[['User_ID', 'Age', 'Gender', 'Purchase']].head(15), use_container_width=True)

