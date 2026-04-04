# Mining the Future: Unlocking Business Intelligence with AI
# IDAI1021000480

## 📌 Project Overview
This project serves as a comprehensive Data Mining solution for **InsightMart Analytics**. By analyzing large-scale Black Friday retail data, I uncovered hidden patterns in consumer behavior to drive strategic decision-making, optimize resource allocation, and enhance customer engagement.

## 🚀 Live Dashboard
**Link to access app:** (https://data-mining-sa-cre9bvlrre9253wewf9wvb.streamlit.app/)

---

## 🛠️ Project Stages & Methodology

### 1. Project Scope (Stage 1)
The primary objective is to move "Beyond Discounts" by identifying:
* Key shopping behaviors across demographics.
* Distinct customer segments for tailored marketing.
* High-value cross-selling opportunities.
* Anomalous transactions for VIP or fraud detection.

<img width="2014" height="916" alt="image" src="https://github.com/user-attachments/assets/38d1883a-e623-41ff-89ed-216c2008dad9" />


### 2. Data Preprocessing (Stage 2)
Raw data was prepared for AI modeling through:
* **Cleaning:** Handled missing values in `Product_Category_2` and `Product_Category_3`.
* **Encoding:** Converted categorical `Gender` and `Age` into numerical formats for machine learning compatibility.
* **Normalization:** Applied `StandardScaler` to `Purchase` amounts to ensure feature parity during clustering.

<img width="1990" height="884" alt="image" src="https://github.com/user-attachments/assets/4fa4f5fe-1891-4de0-818a-b455a1fb2bd9" />

### 3. Exploratory Data Analysis (Stage 3)
Visualized trends using:
* **Bar Charts:** To identify the most popular product categories.
* **Box Plots:** To compare spending distributions across Genders and Age groups.

<img width="1994" height="1072" alt="image" src="https://github.com/user-attachments/assets/6a660760-17ad-46a6-a588-831fbf8111ae" />

<img width="2050" height="854" alt="image" src="https://github.com/user-attachments/assets/1380532e-e9c7-4433-8995-bff742d4600e" />

### 4. Clustering Analysis (Stage 4)
Applied the **K-Means Algorithm** to segment customers based on Age and Spending habits. I used the **Elbow Method** logic to determine the optimal number of clusters, resulting in three distinct groups: *Budget Shoppers*, *Average Spenders*, and *Premium Buyers*.

<img width="2072" height="818" alt="image" src="https://github.com/user-attachments/assets/39f05f91-0089-48eb-adba-ae9ad3129501" />


### 5. Association Rule Mining (Stage 5)
Leveraged the **Apriori Algorithm** to discover frequent product combinations. By analyzing `support`, `confidence`, and `lift`, I identified categories often purchased together (e.g., Category 1 and Category 5), providing clear cross-selling insights.



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
