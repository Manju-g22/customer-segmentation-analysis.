import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="👥",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("👥 Customer Segmentation Dashboard")
st.markdown(
    "Analyze customer demographics, purchasing behavior and spending patterns "
    "using **K-Means Clustering**."
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("customer_data.csv")

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔎 Filters")

gender_options = df["Gender"].unique().tolist()

selected_gender = st.sidebar.multiselect(
    "Gender",
    gender_options,
    default=gender_options
)

age_range = st.sidebar.slider(
    "Age Range",
    int(df["Age"].min()),
    int(df["Age"].max()),
    (int(df["Age"].min()), int(df["Age"].max()))
)

income_range = st.sidebar.slider(
    "Annual Income",
    int(df["AnnualIncome"].min()),
    int(df["AnnualIncome"].max()),
    (int(df["AnnualIncome"].min()), int(df["AnnualIncome"].max()))
)

filtered_df = df[
    (df["Gender"].isin(selected_gender)) &
    (df["Age"].between(age_range[0], age_range[1])) &
    (df["AnnualIncome"].between(income_range[0], income_range[1]))
].copy()

# ---------------- KPI CARDS ----------------
st.subheader("📊 Key Performance Indicators")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "👥 Customers",
    f"{len(filtered_df):,}"
)

c2.metric(
    "🎂 Avg Age",
    f"{filtered_df['Age'].mean():.1f}"
)

c3.metric(
    "💰 Avg Income",
    f"₹{filtered_df['AnnualIncome'].mean():,.0f}"
)

c4.metric(
    "⭐ Avg Spending",
    f"{filtered_df['SpendingScore'].mean():.1f}"
)

st.divider()

# ---------------- CLUSTERING ----------------
features = [
    "Age",
    "AnnualIncome",
    "SpendingScore",
    "PurchaseFrequency"
]

if len(filtered_df) < 10:
    st.warning("Please select more customers using the filters.")
    st.stop()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(filtered_df[features])

# ---------------- ELBOW METHOD ----------------
st.subheader("📉 Determine the Optimal Number of Segments")

inertia = []

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(X_scaled)
    inertia.append(model.inertia_)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(2, 9), inertia, marker="o")
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Method")
ax.grid(alpha=0.3)

st.pyplot(fig)

# ---------------- CLUSTER SELECTION ----------------
st.subheader("🎯 Customer Segmentation")

num_clusters = st.slider(
    "Choose Number of Customer Segments",
    min_value=2,
    max_value=8,
    value=4
)

kmeans = KMeans(
    n_clusters=num_clusters,
    random_state=42,
    n_init=10
)

filtered_df["Cluster"] = kmeans.fit_predict(X_scaled)

# ---------------- SEGMENT PROFILE ----------------
profile = filtered_df.groupby("Cluster")[features].mean()

# Create meaningful segment names
overall_income = filtered_df["AnnualIncome"].mean()
overall_spending = filtered_df["SpendingScore"].mean()
overall_frequency = filtered_df["PurchaseFrequency"].mean()

segment_names = {}

for cluster in profile.index:
    income = profile.loc[cluster, "AnnualIncome"]
    spending = profile.loc[cluster, "SpendingScore"]
    frequency = profile.loc[cluster, "PurchaseFrequency"]

    if income >= overall_income and spending >= overall_spending and frequency >= overall_frequency:
        name = "High-Value Customers"
    elif spending >= overall_spending and frequency >= overall_frequency:
        name = "Loyal & Active Customers"
    elif income >= overall_income and spending < overall_spending:
        name = "Premium Low-Spenders"
    elif income < overall_income and spending >= overall_spending:
        name = "Budget High-Spenders"
    else:
        name = "Low-Engagement Customers"

    segment_names[cluster] = name

filtered_df["Segment"] = filtered_df["Cluster"].map(segment_names)

# ---------------- SEGMENT COUNT ----------------
st.subheader("👥 Customer Distribution")

segment_counts = (
    filtered_df["Segment"]
    .value_counts()
    .reset_index()
)

segment_counts.columns = ["Segment", "Customers"]

fig, ax = plt.subplots(figsize=(9, 4))

sns.barplot(
    data=segment_counts,
    x="Segment",
    y="Customers",
    ax=ax
)

ax.set_xlabel("Customer Segment")
ax.set_ylabel("Number of Customers")
ax.set_title("Customers by Segment")
plt.xticks(rotation=20)

st.pyplot(fig)

# ---------------- SCATTER PLOT ----------------
st.subheader("💰 Income vs Spending Behavior")

fig, ax = plt.subplots(figsize=(9, 5))

sns.scatterplot(
    data=filtered_df,
    x="AnnualIncome",
    y="SpendingScore",
    hue="Segment",
    s=80,
    ax=ax
)

ax.set_title("Customer Segments by Income and Spending")
ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")

st.pyplot(fig)

# ---------------- AGE VS SPENDING ----------------
st.subheader("🎂 Age vs Spending Score")

fig, ax = plt.subplots(figsize=(9, 5))

sns.scatterplot(
    data=filtered_df,
    x="Age",
    y="SpendingScore",
    hue="Segment",
    s=70,
    ax=ax
)

ax.set_title("Age and Spending Behavior")
ax.set_xlabel("Age")
ax.set_ylabel("Spending Score")

st.pyplot(fig)

# ---------------- SEGMENT PROFILE ----------------
st.subheader("📋 Segment Characteristics")

display_profile = (
    filtered_df
    .groupby("Segment")[features]
    .mean()
    .round(2)
)

st.dataframe(
    display_profile,
    use_container_width=True
)

# ---------------- PURCHASE FREQUENCY ----------------
st.subheader("🛒 Purchase Frequency by Segment")

frequency = (
    filtered_df
    .groupby("Segment")["PurchaseFrequency"]
    .mean()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(9, 4))

frequency.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Customer Segment")
ax.set_ylabel("Average Purchase Frequency")
ax.set_title("Average Purchase Frequency")
plt.xticks(rotation=20)

st.pyplot(fig)

# ---------------- BUSINESS INSIGHTS ----------------
st.subheader("💡 Business Insights")

for segment in display_profile.index:

    row = display_profile.loc[segment]

    st.markdown(f"### 🎯 {segment}")

    st.write(
        f"- Average age: **{row['Age']:.1f} years**"
    )

    st.write(
        f"- Average annual income: **₹{row['AnnualIncome']:,.0f}**"
    )

    st.write(
        f"- Average spending score: **{row['SpendingScore']:.1f}**"
    )

    st.write(
        f"- Average purchase frequency: **{row['PurchaseFrequency']:.1f} purchases**"
    )

    if row["SpendingScore"] >= overall_spending:
        st.success(
            "Recommendation: Target this segment with personalized offers, "
            "loyalty rewards and cross-selling campaigns."
        )
    else:
        st.info(
            "Recommendation: Use discounts, personalized recommendations "
            "and engagement campaigns to increase spending."
        )

# ---------------- DOWNLOAD ----------------
st.subheader("📥 Download Segmented Customer Data")

download_df = filtered_df.drop(columns=["Cluster"])

csv_data = download_df.to_csv(index=False)

st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name="customer_segments.csv",
    mime="text/csv"
)

# ---------------- RAW DATA ----------------
with st.expander("👤 View Customer Data"):

    st.dataframe(
        filtered_df.drop(columns=["Cluster"]),
        use_container_width=True
    )

st.divider()

st.caption(
    "Customer Segmentation Analysis | Python • Pandas • Scikit-learn • Streamlit"
)

