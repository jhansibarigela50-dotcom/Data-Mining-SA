# Mining the Future: Unlocking Business Intelligence with AI
**Scenario 1: Beyond Discounts – Data-Driven Black Friday Sales Insights**

## 📌 Project Overview
This project serves as a comprehensive Data Mining solution for **InsightMart Analytics**. By analyzing large-scale Black Friday retail data, we uncover hidden patterns in consumer behavior to drive strategic decision-making, optimize resource allocation, and enhance customer engagement.

## 🚀 Live Dashboard
**View the Deployed App:** [Insert Your Streamlit Link Here]

---

## 🛠️ Project Stages & Methodology

### 1. Project Scope (Stage 1)
The primary objective is to move "Beyond Discounts" by identifying:
* Key shopping behaviors across demographics.
* Distinct customer segments for tailored marketing.
* High-value cross-selling opportunities.
* Anomalous transactions for VIP or fraud detection.

### 2. Data Preprocessing (Stage 2)
Raw data was prepared for AI modeling through:
* **Cleaning:** Handled missing values in `Product_Category_2` and `Product_Category_3`.
* **Encoding:** Converted categorical `Gender` and `Age` into numerical formats for machine learning compatibility.
* **Normalization:** Applied `StandardScaler` to `Purchase` amounts to ensure feature parity during clustering.

### 3. Exploratory Data Analysis (Stage 3)
Visualized trends using:
* **Bar Charts:** To identify the most popular product categories.
* **Box Plots:** To compare spending distributions across Genders and Age groups.

### 4. Clustering Analysis (Stage 4)
Applied the **K-Means Algorithm** to segment customers based on Age and Spending habits. We used the **Elbow Method** logic to determine the optimal number of clusters, resulting in three distinct groups: *Budget Shoppers*, *Average Spenders*, and *Premium Buyers*.

### 5. Association Rule Mining (Stage 5)
Leveraged the **Apriori Algorithm** to discover frequent product combinations. By analyzing `support`, `confidence`, and `lift`, we identified categories often purchased together (e.g., Category 1 and Category 5), providing clear cross-selling insights.

### 6. Anomaly Detection (Stage 6)
Utilized statistical methods (**IQR/Z-Score**) to detect "Whales"—customers with exceptionally high purchase volumes. This helps retailers identify VIP customers or potential bulk-purchase anomalies.

---

## 🖥️ App Functionality
The Streamlit dashboard is organized into interactive sections:
* **Sidebar Navigation:** Easy access to each project stage.
* **Interactive Data Tables:** Cleaned, scannable views of the dataset without horizontal scrolling.
* **Live AI Models:** Real-time generation of clusters and association rules.
* **Strategic Reporting:** Actionable business recommendations based on findings.

## 📂 Repository Structure
* `app.py`: Main application script containing the UI and AI logic.
* `requirements.txt`: List of Python dependencies (Streamlit, Scikit-Learn, etc.).
* `BlackFriday.csv`: The source dataset.
* `README.md`: Project documentation.

---

## 👥 Project Details
* **Course:** Artificial Intelligence
* **Assessment:** Summative Assessment (60 Marks)
* **Student:** [Your Name]
* **Registration Number:** [Your Number]
