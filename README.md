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

<img width="2008" height="890" alt="image" src="https://github.com/user-attachments/assets/5a89a9ec-9224-4ebd-8abe-59208cc902b3" />


### 2. Market Exploratory Analysis (Stage 3)
We utilized **Seaborn** and **Matplotlib** to visualize spending distribution. Key findings include:
* The **26-35 age demographic** represents the highest total revenue.
* **Category 1** serves as the primary "Anchor Category" for the store.

<img width="1998" height="906" alt="image" src="https://github.com/user-attachments/assets/792b6c03-130f-4567-8d3c-1b22624752b3" />

### 3. Customer Segmentation (Stage 4)
We implemented the **K-Means Clustering** algorithm:
* **Algorithm:** K-Means with $k=3$ (determined via Elbow Method).
* **Segments:** Budget Shoppers, Occasional Buyers, and Premium Buyers.
* **Goal:** To move "Occasional" buyers into the "Premium" tier via personalized incentives.

<img width="2078" height="836" alt="image" src="https://github.com/user-attachments/assets/350a1569-86fc-4ffa-b264-5c7d41effa07" />

### 4. Market Basket Analysis (Stage 5)
Using the **Apriori Algorithm**, we identified frequent itemsets and association rules:
* **Metric:** Focused on **Lift** and **Confidence** to find non-obvious product pairings.
* **Application:** Informs shelf-placement and digital "Frequently Bought Together" bundles.

<img width="2052" height="1036" alt="image" src="https://github.com/user-attachments/assets/ea98f62d-9558-44cc-b176-e806dfe7a828" />

### 5. Anomaly Detection (Stage 6)
Applied statistical **Outlier Detection** (IQR and Standard Deviation methods) to identify "Whale" spenders. These transactions are flagged for VIP loyalty conversion or fraud prevention.

<img width="1994" height="998" alt="image" src="https://github.com/user-attachments/assets/80463cc6-b4f5-4628-990c-bb1f80ff0590" />

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


