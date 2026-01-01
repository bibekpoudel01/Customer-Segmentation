import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("Customer Segmentation Dashboard (RFM + KMeans + PCA)")
# ------------------ Helper Functions ------------------ #
def load_pickle(file_name):
    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning(f"{file_name} not found!")
        return None

def plot_cluster_distribution(rfm_df):
    cluster_counts = rfm_df['Cluster'].value_counts().sort_index()
    fig, ax = plt.subplots()
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax, palette="viridis")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Cluster Distribution")
    st.pyplot(fig)

def plot_cluster_profiles(rfm_df):
    cluster_stats = rfm_df.groupby('Cluster')[['Recency','Frequency','Monetary']].mean()
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(cluster_stats, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
    ax.set_title("Cluster Profiles (Average RFM Values)")
    st.pyplot(fig)

def plot_pca_3d(df_pca):
    fig = px.scatter_3d(df_pca, x='PCA1', y='PCA2', z='PCA3', color='Cluster',
                        color_discrete_sequence=px.colors.qualitative.Dark24)
    st.plotly_chart(fig, use_container_width=True)

# ------------------ Main App ------------------ #
uploaded_file = st.file_uploader("Upload your RFM CSV file", type=["csv"])
if uploaded_file:
    try:
        rfm = pd.read_csv(uploaded_file)
        st.subheader("Raw RFM Data")
        st.dataframe(rfm.head())

        # Load saved models
        scaler = load_pickle('scaler.pkl')
        kmeans = load_pickle('kmeans.pkl')
        pca = load_pickle('pca.pkl')
        use_pca = pca is not None

        if scaler and kmeans:
            # Scale and predict clusters
            X_scaled = scaler.transform(rfm[['Recency','Frequency','Monetary']])
            rfm['Cluster'] = kmeans.predict(X_scaled).astype(str)

            # Show clustered data
            st.subheader("Clustered Data")
            st.dataframe(rfm)

            # Visualizations
            st.subheader("Cluster Analysis")
            col1, col2 = st.columns(2)
            with col1:
                plot_cluster_distribution(rfm)
            with col2:
                plot_cluster_profiles(rfm)

            # PCA plot if available
            if use_pca:
                X_pca = pca.transform(X_scaled)
                df_pca = pd.DataFrame(X_pca, columns=['PCA1','PCA2','PCA3'])
                df_pca['Cluster'] = rfm['Cluster']
                st.subheader("3D PCA Scatter Plot of Clusters")
                plot_pca_3d(df_pca)
                # Download PCA data
                st.download_button(
                    label="Download PCA Data as CSV",
                    data=df_pca.to_csv(index=False),
                    file_name='df_pca.csv',
                    mime='text/csv'
                )

            # Filter by cluster
            st.subheader("Filter Customers by Cluster")
            selected_cluster = st.selectbox("Select Cluster", sorted(rfm['Cluster'].unique()))
            filtered_df = rfm[rfm['Cluster'] == selected_cluster]
            st.dataframe(filtered_df)
            st.download_button(
                label=f"Download Cluster {selected_cluster} Data",
                data=filtered_df.to_csv(index=False),
                file_name=f'cluster_{selected_cluster}_data.csv',
                mime='text/csv'
            )

        else:
            st.error("Scaler or KMeans model not found. Please check the files.")
    except Exception as e:
        st.error(f"Error processing file: {e}")


