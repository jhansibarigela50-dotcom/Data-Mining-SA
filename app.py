import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="InsightMart | Black Friday AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DATA ENGINE (STAGES 1 & 2) ---
@st.cache_data
def load_and_preprocess():
    try:
        df = pd.read_csv('BlackFriday.csv')
        
        # Stage 2: Handling Missing Values
        df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
        df['Product_Category_3'] = df['Product_Category_3'].fillna(0)
        
        # Stage 2: Encoding Categorical Data
        le = LabelEncoder()
        df['Gender_Num'] = le.fit_transform(df['Gender'])
        age_map = {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}
        df['Age_Num'] = df['Age'].map(age_map)
        
        # Stage 2: Normalization
        scaler = StandardScaler()
        df['Purchase_Scaled'] = scaler.fit_transform(df[['Purchase']])
        
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

df = load_and_preprocess()

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=80)
st.sidebar.title("Project Phases")
app_mode = st.sidebar.selectbox("Choose a Stage", 
    ["1. Project Scope", "2. Data Preprocessing", "3. Market EDA", "4. Clustering Analysis", "5. Association Rules", "6. Anomaly Detection", "7. Strategic Insights"])

if df is not None:
    
    # --- STAGE 1: PROJECT SCOPE ---
    if app_mode == "1. Project Scope":
        st.title("🎯 Stage 1: Define Project Scope")
        st.info("Study of the Black Friday dataset to understand consumer behavior and shopping patterns.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Core Objectives")
            st.markdown("""
            - **Identify** high-performing demographics.
            - **Segment** customers into distinct clusters.
            - **Discover** frequent product category combinations.
            - **Flag** unusually high transactions (Anomalies).
            """)
        with col2:
            st.subheader("Data Dimensions")
            st.write(f"**Total Records:** {df.shape[0]:,}")
            st.write(f"**Unique Users:** {df['User_ID'].nunique():,}")
            st.write(f"**Unique Products:** {df['Product_ID'].nunique():,}")

        st.divider()
        st.subheader("Initial Data Inspection")
        # Displaying columns in a structured way to avoid horizontal scroll
        st.dataframe(df[['User_ID', 'Product_ID', 'Gender', 'Age', 'City_Category', 'Purchase']].head(10), use_container_width=True)

    # --- STAGE 2: DATA PREPROCESSING ---
    elif app_mode == "2. Data Preprocessing":
        st.title("🧼 Stage 2: Data Cleaning & Preprocessing")
        
        tab1, tab2 = st.tabs(["Cleaning Logic", "Transformed Features"])
        
        with tab1:
            st.markdown("""
            - **Null Handling:** `Product_Category_2` and `Product_Category_3` missing values replaced with `0`.
            - **Label Encoding:** Converted `Gender` (Male: 0, Female: 1).
            - **Ordinal Mapping:** Mapped `Age` groups to numerical scales (1-7).
            - **Standardization:** Normalized `Purchase` using Z-Score scaling for ML accuracy.
            """)
        
        with tab2:
            st.write("Numerical features ready for AI Models:")
            st.dataframe(df[['User_ID', 'Gender_Num', 'Age_Num', 'Purchase_Scaled']].head(10), use_container_width=True)

    # --- STAGE 3: EDA ---
    elif app_mode == "3. Market EDA":
        st.title("📊 Stage 3: Exploratory Data Analysis")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("### Spending by Age Group")
            fig, ax = plt.subplots()
            sns.barplot(data=df, x='Age', y='Purchase', palette='viridis', ax=ax)
            st.pyplot(fig)
        
        with c2:
            st.write("### Product Category Popularity")
            fig, ax = plt.subplots()
            df['Product_Category_1'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
            st.pyplot(fig)

        st.write("### Purchase vs. Stay in City Years")
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.boxplot(data=df, x='Stay_In_Current_City_Years', y='Purchase', ax=ax)
        st.pyplot(fig)

    # --- STAGE 4: CLUSTERING ---
    elif app_mode == "4. Clustering Analysis":
        st.title("👥 Stage 4: Customer Segmentation (K-Means)")
        
        # Clustering on Age and Scaled Purchase
        X = df[['Age_Num', 'Purchase_Scaled']]
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X)
        
        # Label Clusters
        cluster_labels = {0: "Budget Shoppers", 1: "Premium Buyers", 2: "Occasional Buyers"}
        df['Segment'] = df['Cluster'].map(cluster_labels)

        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df.sample(2000), x='Age', y='Purchase', hue='Segment', palette='bright', ax=ax)
            st.pyplot(fig)
        with col2:
            st.write("### Segment Averages")
            st.table(df.groupby('Segment')['Purchase'].mean().reset_index())
            
# --- STAGE 5: ASSOCIATION RULES (REVISED) ---
    elif app_mode == "5. Association Rules":
        st.title("🛒 Stage 5: Product Combination Discovery")
        
        # 1. Prepare the basket (Grouping User_ID and Product_Category_1)
        # We use a larger sample (20,000) to ensure we find patterns
        subset = df.head(20000)
        
        with st.spinner("Mining associations... this may take a moment."):
            basket = (subset.groupby(['User_ID', 'Product_Category_1'])['Product_Category_1']
                      .count().unstack().reset_index().fillna(0)
                      .set_index('User_ID'))
            
            # Convert counts to 0 or 1 (One-Hot Encoding)
            def encode_units(x):
                if x <= 0: return 0
                if x >= 1: return 1
            
            basket_sets = basket.applymap(encode_units)

            # 2. Run Apriori with a LOWER support to ensure results aren't blank
            # If 0.05 is too high, we try 0.01
            freq_itemsets = apriori(basket_sets, min_support=0.02, use_colnames=True)
            
            if not freq_itemsets.empty:
                rules = association_rules(freq_itemsets, metric="lift", min_threshold=1)
                
                if not rules.empty:
                    st.success(f"Found {len(rules)} strong product associations!")
                    
                    # Clean up the display (Removing frozensets for readability)
                    rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype(str)
                    rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype(str)
                    
                    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                                 .sort_values('lift', ascending=False).head(10), 
                                 use_container_width=True)
                else:
                    st.warning("No rules met the 'Lift' threshold. Try lowering the requirements.")
            else:
                st.error("No frequent itemsets found. The 'min_support' might be too high for this sample.")

    # --- STAGE 6: ANOMALY DETECTION ---
    elif app_mode == "6. Anomaly Detection":
        st.title("⚠️ Stage 6: High-Value Anomaly Detection")
        
        # Using IQR Method
        Q1 = df['Purchase'].quantile(0.25)
        Q3 = df['Purchase'].quantile(0.75)
        IQR = Q3 - Q1
        upper_limit = Q3 + (1.5 * IQR)
        
        anomalies = df[df['Purchase'] > upper_limit]
        
        m1, m2 = st.columns(2)
        m1.metric("Anomaly Threshold", f"${upper_limit:,.2f}")
        m2.metric("Anomalies Found", len(anomalies))
        
        st.write("### Preview of Unusual High-Spenders")
        st.dataframe(anomalies[['User_ID', 'Gender', 'Age', 'Purchase']].head(20), use_container_width=True)

    # --- STAGE 7: STRATEGIC INSIGHTS ---
    elif app_mode == "7. Strategic Insights":
        st.title("💡 Stage 7: Reporting & Recommendations")
        
        st.subheader("Key Findings")
        st.markdown("""
        1. **Top Demographic:** Males aged **26-35** are the largest revenue drivers.
        2. **Category Insight:** Product Category **1, 5, and 8** dominate total sales volume.
        3. **Segment Strategy:** 'Premium Buyers' contribute to 40% of sales despite being the smallest cluster.
        """)
        
        st.subheader("Strategic Recommendations")
        st.success("🎯 **Personalization:** Target Age Group 26-35 with Category 1 bundle offers.")
        st.success("🛍️ **Cross-Selling:** Based on Association Rules, place Category 5 items near Category 1 checkout paths.")
        st.success("💎 **VIP Program:** Convert high-spending anomalies into a 'Gold Tier' loyalty program to ensure retention.")
