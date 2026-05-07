import pandas as pd
import numpy as np
import os

np.random.seed(42)

n = 200

customer_id = np.arange(1, n + 1)
annual_income = np.random.randint(15000, 150000, n)
spending_score = np.random.randint(1, 100, n)
purchase_frequency = np.random.randint(1, 50, n)

# Generate total purchase amount
purchase_amount = (
    annual_income * 0.3 +
    spending_score * 100 +
    purchase_frequency * 500 +
    np.random.randint(-5000, 5000, n)
)
# Create DataFrame
customers = pd.DataFrame({
    "CustomerID": customer_id,
    "AnnualIncome": annual_income,
    "SpendingScore": spending_score,
    "PurchaseFrequency": purchase_frequency,
    "PurchaseAmount": purchase_amount.astype(int)
})

# Create folder if not exists
os.makedirs("data", exist_ok=True)

# Save dataset
customers.to_csv("data/customers.csv", index=False)

print("Customer dataset created successfully!")