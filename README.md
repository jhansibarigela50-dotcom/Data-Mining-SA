# Mining the Future: Unlocking Business Intelligence with AI
# IDAI1021000480

* **Name:** Jhansi Barigela
* **Registration Number:** IDAI1021000480
* **Crs:** Artificial Intelligence 
* **Course:** Data Mining
* **School Name:** Birla Open Minds International School

# 🛍️ InsightMart: Beyond Discounts
**AI-Driven Sales Intelligence for Black Friday Operations**

## 📌 Project Overview
This project addresses the **"Beyond Discounts"** scenario for InsightMart. Rather than relying on blanket price cuts, this application uses **Machine Learning** and **Data Mining** to uncover high-value customer segments, product affinities, and purchase anomalies.

## 🚀 Live Application
(https://data-mining-sa-cre9bvlrre9253wewf9wvb.streamlit.app/)

---

## 🛠️ Data Pipeline & AI Methodology

### 1. Data Preprocessing (Stage 2)
To ensure model accuracy, the raw dataset underwent rigorous cleaning:
* **Null Handling:** Imputed missing values in `Product_Category_2` and `Product_Category_3`.
* **Feature Engineering:** Created `Gender_Num` (Label Encoding) and `Age_Num` (Ordinal Mapping).
* **Scaling:** Applied `StandardScaler` (Z-score normalization) to the `Purchase` feature to prevent high-magnitude outliers from biasing the clustering algorithm.

<img width="1988" height="890" alt="image" src="https://github.com/user-attachments/assets/08fc6a09-a74b-46ae-b16f-361a52c3e2eb" />

### 2. Market Exploratory Analysis (Stage 3)
We utilized **Seaborn** and **Matplotlib** to visualize spending distribution. Key findings include:
* The **26-35 age demographic** represents the highest total revenue.
* **Category 1** serves as the primary "Anchor Category" for the store.

<img width="1980" height="832" alt="image" src="https://github.com/user-attachments/assets/8d8e3a19-e792-4d3d-be8b-b51625db607a" />

### 3. Customer Segmentation (Stage 4)
We implemented the **K-Means Clustering** algorithm:
* **Algorithm:** K-Means with $k=3$ (determined via Elbow Method).
* **Segments:** Budget Shoppers, Occasional Buyers, and Premium Buyers.
* **Goal:** To move "Occasional" buyers into the "Premium" tier via personalized incentives.

<img width="2006" height="1022" alt="image" src="https://github.com/user-attachments/assets/51c00663-46b0-4d10-8508-479e59349f8a" />

### 4. Market Basket Analysis (Stage 5)
Using the **Apriori Algorithm**, we identified frequent itemsets and association rules:
* **Metric:** Focused on **Lift** and **Confidence** to find non-obvious product pairings.
* **Application:** Informs shelf-placement and digital "Frequently Bought Together" bundles.

<img width="1976" height="798" alt="image" src="https://github.com/user-attachments/assets/201645db-73f7-4945-b453-6f655971a622" />

### 5. Anomaly Detection (Stage 6)
Applied statistical **Outlier Detection** (IQR and Standard Deviation methods) to identify "Whale" spenders. These transactions are flagged for VIP loyalty conversion or fraud prevention.

<img width="1976" height="824" alt="image" src="https://github.com/user-attachments/assets/4fb1af20-ac41-4945-9fc3-6379f39f715d" />

---

## 💡 Strategic Insights & Analysis
The application provides an automated **Executive Summary** in Stage 7, offering:
* **Inventory Recommendations:** Based on association lift.
* **Marketing Focus:** Weighted toward the dominant 26-35 male demographic.
* **Margin Optimization:** Shifting from a discount-heavy model to a precision-marketing model.

## 📂 Repository Structure
* `app.py`: The main Streamlit application containing the UI and AI logic.
* `requirements.txt`: Environment dependencies (Pandas 3.0 compatible).
* `BlackFriday.csv`: The source dataset.
* `README.md`: Project documentation.


