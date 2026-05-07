# 🛍️ Customer Segmentation using K-Means Clustering

## 📌 Overview
This project groups retail customers based on purchasing behavior using K-Means Clustering.

The model analyzes:
- Annual Income
- Spending Score
- Purchase Frequency
- Purchase Amount

and segments customers into meaningful groups.

---

## 🚀 Features
- 📊 Large synthetic customer dataset
- 🧠 K-Means clustering algorithm
- 📈 Customer segmentation visualization
- 🌐 Interactive Streamlit web app
- 🎨 Styled UI dashboard
- 📉 Cluster interpretation system

---

## 🛠 Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib

---

## 📸 Demo


## 📊 Customer Segments
- 🔵 Occasional Customers
- 🟡 Regular Customers
- 🟢 Premium Customers
- 🟣 Budget Customers

---

## ▶️ Run Project

```bash
pip install -r requirements.txt

python src/generate_data.py
python src/train_model.py

python -m streamlit run app/app.py