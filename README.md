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
<img width="983" height="800" alt="customer" src="https://github.com/user-attachments/assets/7e445b66-521a-4e97-8892-3e450e17b5b9" />
<img width="931" height="850" alt="cluster" src="https://github.com/user-attachments/assets/7fab88e0-61ec-4ca4-853e-7c23c14b2daa" />
<img width="650" height="151" alt="notations" src="https://github.com/user-attachments/assets/c8326266-cebe-4935-bd72-e5a10429bb6b" />

---

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
