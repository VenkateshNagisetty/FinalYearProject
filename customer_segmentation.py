import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Generate synthetic customer dataset
data = {
    'CustomerID': range(1, 501),
    'Annual Income (k$)': np.random.randint(15, 150, 500),
    'Spending Score (1-100)': np.random.randint(1, 100, 500),
    'Age': np.random.randint(18, 70, 500)
}
df = pd.DataFrame(data)

# Standardizing data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']])

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['KMeans Cluster'] = kmeans.fit_predict(df_scaled)

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
df['Hierarchical Cluster'] = hierarchical.fit_predict(df_scaled)

# Streamlit UI
st.title("📊 Customer Segmentation Dashboard")
st.sidebar.header("Choose Visualization")
option = st.sidebar.selectbox("Select Graph", ["K-Means Clustering", "Hierarchical Dendrogram", "Income vs Spending Analysis", "Cluster Distribution", "Feature Correlation Heatmap"])

if option == "K-Means Clustering":
    st.subheader("K-Means Customer Segmentation")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['KMeans Cluster'], palette='viridis', s=100)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score")
    plt.title("K-Means Clustering")
    st.pyplot(fig)

elif option == "Hierarchical Dendrogram":
    st.subheader("Hierarchical Clustering Dendrogram")
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram = sch.dendrogram(sch.linkage(df_scaled, method='ward'))
    plt.xlabel("Customers")
    plt.ylabel("Distance")
    st.pyplot(fig)

elif option == "Income vs Spending Analysis":
    st.subheader("Income vs Spending Score Analysis")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='KMeans Cluster', y='Annual Income (k$)', data=df, palette='coolwarm')
    plt.title('Income Distribution by Cluster')
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='KMeans Cluster', y='Spending Score (1-100)', data=df, palette='coolwarm')
    plt.title('Spending Score by Cluster')
    st.pyplot(fig)

elif option == "Cluster Distribution":
    st.subheader("Customer Cluster Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='KMeans Cluster', data=df, palette='viridis')
    plt.title('Customer Distribution in Clusters')
    st.pyplot(fig)

elif option == "Feature Correlation Heatmap":
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']].corr(), annot=True, cmap='coolwarm', linewidths=2)
    plt.title('Feature Correlation')
    st.pyplot(fig)

# Business Insights
st.sidebar.subheader("📌 Business Insights")
st.sidebar.write("- High-income & high-spending customers are VIPs.")
st.sidebar.write("- Low-income, low-spending customers need promotions.")
st.sidebar.write("- Clustering helps businesses target the right audience.")

st.sidebar.write("📢 Use these insights for better marketing strategies!")
