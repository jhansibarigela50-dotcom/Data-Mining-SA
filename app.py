import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="InsightMart | AI Sales Intelligence", layout="wide", page_icon="🛍️")
sns.set_theme(style="whitegrid") # Makes graphs look professional

# --- 2. DATA ENGINE (STAGES 1 & 2) ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('BlackFriday.csv')
        # Stage 2: Cleaning Missing Values
        df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
        df['Product_Category_3'] = df['Product_Category_3'].fillna(0)
        
        # Stage 2: Encoding & Feature Engineering
        le = LabelEncoder()
        df['Gender_Num'] = le.fit_transform(df['Gender'])
        age_map = {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}
        df['Age_Num'] = df['Age'].map(age_map)
        
        # Stage 2: Scaling for AI Models
        scaler = StandardScaler()
        df['Purchase_Scaled'] = scaler.fit_transform(df[['Purchase']])
        return df
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return None

df = load_data()

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("Project Phases")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate to:", 
    ["1. Project Scope", "2. Data Preprocessing", "3. Market EDA", "4. Customer Clustering", "5. Association Rules", "6. Anomaly Detection", "7. Strategic Insights"])

if df is not None:
    # --- STAGE 1: SCOPE ---
    if page == "1. Project Scope":
        st.title("🎯 Stage 1: Define Project Scope")
        st.markdown("### Scenario: Beyond Discounts – Data-Driven Sales Insights")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Objective:** To move beyond basic price-cutting and use AI to identify high-value customer segments and product affinities.")
        with col2:
            st.success("**Outcome:** Data-backed recommendations for inventory and marketing spend.")
            
        st.subheader("Raw Dataset Preview")
        display_cols = ['User_ID', 'Product_ID', 'Gender', 'Age', 'City_Category', 'Purchase']
        st.dataframe(df[display_cols].head(10), use_container_width=True)

    # --- STAGE 2: PREPROCESSING ---
    elif page == "2. Data Preprocessing":
        st.title("🧼 Stage 2: Data Cleaning & Preprocessing")
        st.write("Ensuring data quality by handling nulls and normalizing numerical inputs.")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Missing Values", "Cleaned", delta="Cat 2 & 3")
        c2.metric("Categorical Data", "Encoded", delta="Gender/Age")
        c3.metric("Purchase Scale", "Standardized", delta="Z-Score")
        
        st.subheader("Transformed Features for AI")
        st.dataframe(df[['User_ID', 'Gender_Num', 'Age_Num', 'Purchase_Scaled']].head(10), use_container_width=True)

    # --- STAGE 3: EDA (ENHANCED) ---
    elif page == "3. Market EDA":
        st.title("📊 Stage 3: Exploratory Data Analysis")
        
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.write("### Spending by Age Group")
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='Age', y='Purchase', palette="Spectral", ax=ax, showfliers=False)
            st.pyplot(fig)
            
        with row1_col2:
            st.write("### Top 10 Product Categories")
            fig, ax = plt.subplots()
            top_cats = df['Product_Category_1'].value_counts().head(10)
            sns.barplot(x=top_cats.index, y=top_cats.values, palette="mako", ax=ax)
            st.pyplot(fig)

    # --- STAGE 4: CLUSTERING (ENHANCED) ---
    elif page == "4. Customer Clustering":
        st.title("👥 Stage 4: Customer Segmentation (K-Means)")
        
        X = df[['Age_Num', 'Purchase_Scaled']]
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X)
        
        cluster_map = {0: "Budget Shoppers", 1: "Premium Buyers", 2: "Occasional Buyers"}
        df['Segment'] = df['Cluster'].map(cluster_map)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df.sample(3000), x='Age', y='Purchase', hue='Segment', palette='bright', s=60, alpha=0.6, ax=ax)
        st.pyplot(fig)
        
        st.subheader("Segment Performance")
        st.table(df.groupby('Segment')['Purchase'].mean().reset_index())

    # --- STAGE 5: ASSOCIATIONS (PANDAS 3.0 FIX) ---
    elif page == "5. Association Rules":
        st.title("🛒 Stage 5: Product Association Rules")
        st.write("Mining patterns using the **Apriori Algorithm**.")
        
        subset = df.head(15000)
        basket = (subset.groupby(['User_ID', 'Product_Category_1'])['Product_Category_1']
                  .count().unstack().reset_index().fillna(0).set_index('User_ID'))
        
        # PANDAS 3.0 FIX: Using .map() instead of .applymap()
        basket_sets = basket.map(lambda x: 1 if x >= 1 else 0)

        freq_items = apriori(basket_sets, min_support=0.03, use_colnames=True)
        rules = association_rules(freq_items, metric="lift", min_threshold=1)
        
        if not rules.empty:
            st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10), use_container_width=True)
        else:
            st.warning("Increase data sample or lower support to see rules.")

    # --- STAGE 6: ANOMALY DETECTION (ENHANCED) ---
    elif page == "6. Anomaly Detection":
        st.title("⚠️ Stage 6: Anomaly Detection")
        
        limit = df['Purchase'].mean() + (2.5 * df['Purchase'].std())
        anomalies = df[df['Purchase'] > limit]
        
        c1, c2 = st.columns(2)
        c1.metric("Anomaly Threshold", f"${limit:,.2f}")
        c2.metric("High-Value Outliers", len(anomalies))
        
        st.dataframe(anomalies[['User_ID', 'Age', 'Gender', 'Purchase']].head(15), use_container_width=True)
        st.info("💡 Strategic Insight: These users represent VIP customers or potential commercial resellers.")

    # --- STAGE 7: STRATEGIC INSIGHTS ---
    elif page == "7. Strategic Insights":
        st.title("💡 Stage 7: Strategic Recommendations")
        
        st.subheader("Summary of Findings")
        st.markdown("""
        * **Dominant Demographic:** Males (26-35) are the highest revenue contributors.
        * **Cluster Strategy:** 'Premium Buyers' should be targeted with loyalty programs.
        * **Cross-Selling:** Product Category 1 items should be bundled with Category 5 accessories.
        """)
        
        st.success("**Recommendation:** Focus marketing spend on Age 26-35 and use 'Frequently Bought Together' prompts for Category 1 items.")

# --- FOOTER (Stage 8) ---
st.sidebar.markdown("---")
st.sidebar.write("**Student:** [Your Name]")
st.sidebar.write("**ID:** [Your Reg Number]")
st.sidebar.caption("AI Summative Assessment 2026")
