import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load model and scaler
kmeans = joblib.load("model/kmeans_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Load dataset
customers = pd.read_csv("data/customers_clustered.csv")

# Page config
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

h1 {
    color: white;
    text-align: center;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

.result-box {
    background-color: #1e5631;
    padding: 20px;
    border-radius: 10px;
    color: white;
    font-size: 22px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🛍️ Customer Segmentation using K-Means")
st.write("Analyze customer groups based on purchase behavior")

# ---------------- INPUTS ----------------
income = st.slider("Annual Income", 15000, 150000, 50000)

score = st.slider("Spending Score", 1, 100, 50)

frequency = st.slider("Purchase Frequency", 1, 50, 10)

amount = st.slider("Purchase Amount", 5000, 100000, 20000)

# ---------------- SEGMENT NAMES ----------------
segment_names = {
    0: "Occasional Customers",
    1: "Regular Customers",
    2: "Premium Customers",
    3: "Budget Customers"
}

segment_descriptions = {
    0: "High income customers with moderate shopping behavior.",
    1: "Customers with balanced spending and regular purchases.",
    2: "High-value customers with high spending activity.",
    3: "Customers who purchase frequently but spend carefully."
}

# ---------------- PREDICTION ----------------
if st.button("Find Customer Segment"):

    input_data = [[income, score, frequency, amount]]

    scaled_data = scaler.transform(input_data)

    cluster = kmeans.predict(scaled_data)[0]

    st.markdown(f"""
    <div class='result-box'>
        🎯 Customer Segment: {segment_names[cluster]}
    </div>
    """, unsafe_allow_html=True)

    st.write(segment_descriptions[cluster])

    # Model insight
    st.subheader("📊 Model Insights")
    st.write(
        "K-Means clustering groups customers with similar purchasing behavior."
    )

# ---------------- CLUSTER GRAPH ----------------
st.subheader("📈 Customer Clusters")

fig, ax = plt.subplots(figsize=(10, 6))

colors = ['blue', 'yellow', 'green', 'purple']

for cluster in range(4):

    cluster_data = customers[customers["Cluster"] == cluster]

    ax.scatter(
        cluster_data["AnnualIncome"],
        cluster_data["SpendingScore"],
        c=colors[cluster],
        label=segment_names[cluster],
        s=80,
        alpha=0.7
    )

ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Segments")

# Move legend outside graph
ax.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

plt.tight_layout()

st.pyplot(fig)

# ---------------- SEGMENT EXPLANATION ----------------
st.write("""
🔵 **Occasional Customers** → High income but moderate shopping behavior  
🟡 **Regular Customers** → Balanced spending customers  
🟢 **Premium Customers** → High-value customers with high spending  
🟣 **Budget Customers** → Frequent buyers with controlled spending  
""")