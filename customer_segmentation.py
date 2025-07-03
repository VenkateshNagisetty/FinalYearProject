# 👇 All necessary imports remain the same
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy import stats
import io
from fpdf import FPDF
import re

# Page Config
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("📊 Customer Segmentation Dashboard")

# Sample Data Section
with st.expander("📂 Click to View/Download Sample Dataset"):
    st.markdown("This sample helps you understand the expected input format.")
    sample_data = pd.DataFrame({
        'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8],
        'Gender': ['Male', 'Male', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female'],
        'Age': [19, 21, 20, 23, 31, 22, 35, 23],
        'Annual Income (k$)': [15, 15, 16, 16, 17, 17, 18, 18],
        'Spending Score (1-100)': [39, 81, 6, 77, 40, 76, 6, 94]
    })
    st.dataframe(sample_data)
    sample_csv = sample_data.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Sample CSV", data=sample_csv, file_name="sample_customer_data.csv", mime="text/csv")

    if st.button("▶️ Try Clustering with Sample Data"):
        uploaded_file = io.BytesIO(sample_csv)

# Sidebar Upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"]) or locals().get("uploaded_file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Raw Data Preview")
    st.write(df.head())

    # 🔧 Data Cleaning
    st.subheader("🔍 Checking for Missing Values")
    st.write(df.isnull().sum())
    df.dropna(inplace=True)

    # Encode categorical columns
    df = pd.get_dummies(df, drop_first=True)

    # Z-score Outlier Removal
    df = df[(np.abs(stats.zscore(df.select_dtypes(include=[np.number]))) < 3).all(axis=1)]

    # Standardization
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))

    # 🎯 Sidebar Params
    st.sidebar.subheader("🔢 Clustering Parameters")
    n_clusters = st.sidebar.slider("Select number of clusters", 2, 10, 4)

    # 🚀 Clustering Algorithms
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['KMeans Cluster'] = kmeans.fit_predict(df_scaled)

    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    df['Hierarchical Cluster'] = hierarchical.fit_predict(df_scaled)

    dbscan = DBSCAN(eps=1.5, min_samples=5)
    df['DBSCAN Cluster'] = dbscan.fit_predict(df_scaled)

    # 📈 Visualizations
    st.sidebar.header("Choose Visualization")
    option = st.sidebar.selectbox("Select Graph", [
        "K-Means Clustering", "Hierarchical Dendrogram", "DBSCAN Clustering",
        "Income vs Spending Analysis", "Cluster Distribution", "Feature Correlation Heatmap",
        "Cluster Summary Table", "Recommendations"])

    # Intelligent column name fallback
    cols = df.columns.str.lower()
    income_column = next((col for col in df.columns if 'income' in col.lower()), None)
    spending_column = next((col for col in df.columns if 'spending' in col.lower()), None)

    if option == "K-Means Clustering":
        st.subheader("K-Means Customer Segmentation")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=df[income_column], y=df[spending_column], hue=df['KMeans Cluster'], palette='viridis', s=100)
        plt.title("K-Means Clustering")
        st.pyplot(fig)

        img_bytes = io.BytesIO()
        fig.savefig(img_bytes, format='png')
        img_bytes.seek(0)
        st.download_button("📥 Download K-Means Clustering Plot", data=img_bytes, file_name="kmeans_plot.png", mime="image/png")

    elif option == "Hierarchical Dendrogram":
        st.subheader("Hierarchical Clustering Dendrogram")
        fig, ax = plt.subplots(figsize=(10, 6))
        dendrogram = sch.dendrogram(sch.linkage(df_scaled, method='ward'))
        plt.xlabel("Customers")
        plt.ylabel("Distance")
        plt.title("Hierarchical Dendrogram")
        st.pyplot(fig)

    elif option == "DBSCAN Clustering":
        st.subheader("DBSCAN Clustering")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=df[income_column], y=df[spending_column], hue=df['DBSCAN Cluster'], palette='Set2', s=100)
        plt.title("DBSCAN Clustering")
        st.pyplot(fig)

    elif option == "Income vs Spending Analysis":
        st.subheader("Income vs Spending Score Analysis")
        if income_column and spending_column:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x='KMeans Cluster', y=income_column, data=df, palette='coolwarm')
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x='KMeans Cluster', y=spending_column, data=df, palette='coolwarm')
            st.pyplot(fig)
        else:
            st.warning("Income/Spending Score columns not found!")

    elif option == "Cluster Distribution":
        st.subheader("Customer Cluster Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='KMeans Cluster', data=df, palette='viridis')
        st.pyplot(fig)

    elif option == "Feature Correlation Heatmap":
        st.subheader("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        st.pyplot(fig)

    elif option == "Cluster Summary Table":
        st.subheader("📋 Cluster Summary Table")
        summary = df.groupby('KMeans Cluster').mean(numeric_only=True)
        st.dataframe(summary.style.highlight_max(axis=0))

    elif option == "Recommendations":
        st.subheader("📌 Cluster-Based Business Recommendations")

        def generate_pdf(cluster_num, recommendation):
            cleaned_text = re.sub(r'[^\x00-\x7F]+', '', recommendation)
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Cluster {cluster_num} Recommendations", ln=True, align='C')
            pdf.multi_cell(0, 10, txt=cleaned_text)
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            pdf_output = io.BytesIO(pdf_bytes)
            pdf_output.seek(0)
            return pdf_output

        for cluster in sorted(df['KMeans Cluster'].unique()):
            st.info(f"Cluster {cluster} Recommendations")
            cluster_df = df[df['KMeans Cluster'] == cluster]
            avg_income = cluster_df[income_column].mean()
            avg_spending = cluster_df[spending_column].mean()

            recommendation = ""
            if avg_income > df[income_column].mean() and avg_spending > df[spending_column].mean():
                recommendation = "💎 High Income - High Spending: Offer premium services and loyalty programs."
            elif avg_income < df[income_column].mean() and avg_spending > df[spending_column].mean():
                recommendation = "📈 Low Income - High Spending: Provide discount vouchers and cashback offers."
            elif avg_income < df[income_column].mean() and avg_spending < df[spending_column].mean():
                recommendation = "🔍 Low Income - Low Spending: Introduce budget-friendly deals and referral incentives."
            else:
                recommendation = "💼 High Income - Low Spending: Market exclusive products and personalized experiences."

            st.write(recommendation)

            st.download_button(
                label=f"Download Cluster {cluster} Data (CSV)",
                data=cluster_df.to_csv(index=False).encode('utf-8'),
                file_name=f'cluster_{cluster}_data.csv',
                mime='text/csv')

            pdf_data = generate_pdf(cluster, recommendation)
            st.download_button(
                label=f"Download Recommendation PDF for Cluster {cluster}",
                data=pdf_data,
                file_name=f'cluster_{cluster}_recommendation.pdf',
                mime='application/pdf')

    # 📄 Export Data
    st.sidebar.subheader("📄 Export Processed Dataset")
    processed_csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("📥 Download Processed Dataset", data=processed_csv, file_name="processed_customer_data.csv", mime="text/csv")

    st.sidebar.markdown("### 📌 Business Insights")
    st.sidebar.write("- High-income & high-spending customers are VIPs.")
    st.sidebar.write("- Low-income, low-spending customers need promotions.")
    st.sidebar.write("- Clustering helps businesses target the right audience.")
    st.sidebar.write("📢 Use these insights for better marketing strategies!")
