import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("messy_sales_project1.csv")


# -----------------------------
# DATA CLEANING
# -----------------------------

# Product
df["Product"] = df["Product"].str.lower()
df["Product"] = df["Product"].str.strip()

# Region
df["Region"] = df["Region"].str.lower()
df["Region"] = df["Region"].str.strip()
df["Region"] = df["Region"].replace("bengaluru", "bangalore")

# Payment Method
df["Payment_Method"] = df["Payment_Method"].str.lower()
df["Payment_Method"] = df["Payment_Method"].str.strip()

# Quantity
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df.loc[df["Quantity"] <= 0, "Quantity"] = np.nan

# Unit Price
df["Unit_Price"] = pd.to_numeric(df["Unit_Price"], errors="coerce")
df.loc[df["Unit_Price"] <= 0, "Unit_Price"] = np.nan

# Discount
df["Discount"] = df["Discount"].astype(str)

percentage = df["Discount"].str.endswith("%")

df.loc[percentage, "Discount"] = (
    df.loc[percentage, "Discount"]
    .str.replace("%", "", regex=False)
    .astype(float) / 100
)

df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce")

df.loc[
    (df["Discount"] < 0) | (df["Discount"] > 1),
    "Discount"
] = np.nan

# Remove exact duplicate transactions
df = df.drop_duplicates()


# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

df["Gross_sales"] = df["Quantity"] * df["Unit_Price"]

df["Discount_amount"] = (
    df["Gross_sales"] * df["Discount"]
)

df["Net_sales"] = (
    df["Gross_sales"] - df["Discount_amount"]
)

# Date features
df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    errors="coerce"
)

df["ordered_Year"] = df["Order_Date"].dt.year
df["Ordered_month"] = df["Order_Date"].dt.month
df["Ordered_day"] = df["Order_Date"].dt.day

df["Quarter"] = df["Ordered_month"].apply(
    lambda x: (
        1 if x in [1, 2, 3]
        else 2 if x in [4, 5, 6]
        else 3 if x in [7, 8, 9]
        else 4
    )
)


# -----------------------------
# CUSTOMER-LEVEL DATA
# -----------------------------

grouped = df.groupby("Customer_ID").agg({
    "Product": ["count"],
    "Region": ["count"],
    "Payment_Method": ["count"],
    "Quantity": ["sum", "mean"],
    "Net_sales": ["sum", "mean"],
    "Order_ID": ["count"],
    "Quarter": ["nunique"]
})


# Create customer-level DataFrame
gdf = {
    "Customer_ID": grouped.index,
    "Net_sales_sum": grouped[("Net_sales", "sum")],
    "Orders": grouped[("Order_ID", "count")],
    "Quantity_sum": grouped[("Quantity", "sum")],
    "Net_sales_mean": grouped[("Net_sales", "mean")]
}

ndf = pd.DataFrame(gdf).reset_index(drop=True)


# Remove customers whose core sales information
# could not be reliably used for segmentation
ndf = ndf.drop([3, 36])
ndf = ndf.reset_index(drop=True)


# -----------------------------
# FEATURE SCALING
# -----------------------------

features = [
    "Net_sales_sum",
    "Orders",
    "Quantity_sum",
    "Net_sales_mean"
]

scaler = StandardScaler()

xt = scaler.fit_transform(ndf[features])


# -----------------------------
# K-MEANS CLUSTERING
# -----------------------------

kmean = KMeans(
    n_clusters=2,
    random_state=0
)

anss = kmean.fit_predict(xt)

ndf["Cluster"] = anss


# -----------------------------
# CLUSTER ANALYSIS
# -----------------------------

cluster_avg = ndf.groupby("Cluster")[features].mean()

print("Cluster averages:")
print(cluster_avg)


# -----------------------------
# VISUALIZATION
# -----------------------------

plt.scatter(
    ndf["Net_sales_sum"],
    ndf["Orders"],
    c=ndf["Cluster"]
)

plt.xlabel("Total Net Sales")
plt.ylabel("Number of Orders")
plt.title("Customer Segmentation")

plt.show()