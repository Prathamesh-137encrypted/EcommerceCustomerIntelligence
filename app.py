import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="E-Commerce Analytics Dashboard", layout="wide")

st.title("E-Commerce Customer Intelligence & Sales Analytics")

# Define paths
OUTPUTS_DIR = 'outputs/'
FIGURES_DIR = 'outputs/figures/'

@st.cache_data
def load_data():
    yearly_kpis = pd.read_csv(OUTPUTS_DIR + 'yearly_kpis.csv') if os.path.exists(OUTPUTS_DIR + 'yearly_kpis.csv') else None
    customer_summary = pd.read_csv(OUTPUTS_DIR + 'customer_summary.csv') if os.path.exists(OUTPUTS_DIR + 'customer_summary.csv') else None
    segments = pd.read_csv(OUTPUTS_DIR + 'customer_segments.csv') if os.path.exists(OUTPUTS_DIR + 'customer_segments.csv') else None
    return yearly_kpis, customer_summary, segments

yearly_kpis, customer_summary, segments = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a view:", ["Executive Summary", "Customer Segmentation (RFM)", "Geographic & Product Insights"])

if page == "Executive Summary":
    st.header("Executive Summary")
    if yearly_kpis is not None:
        st.dataframe(yearly_kpis, use_container_width=True)
    
    st.subheader("Key Trends")
    # Show pre-rendered figure if exists
    trend_img = os.path.join(FIGURES_DIR, "18_monthly_gross_net_revenue.png")
    if os.path.exists(trend_img):
        st.image(trend_img, caption="Monthly Revenue Trends", use_column_width=True)
    
elif page == "Customer Segmentation (RFM)":
    st.header("Customer Segmentation (RFM + K-Means)")
    
    if segments is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers Segmented", len(segments))
        col2.metric("Total Segments Found", segments['Cluster'].nunique())
        
        st.subheader("Cluster Distribution")
        st.bar_chart(segments['Cluster'].value_counts())
        
        st.subheader("RFM Segments Sample")
        st.dataframe(segments.head(20), use_container_width=True)
        
        cluster_img = os.path.join(FIGURES_DIR, "52_kmeans_clusters.png")
        if os.path.exists(cluster_img):
            st.image(cluster_img, caption="K-Means Clusters", use_column_width=True)
    else:
        st.warning("Customer segments data not found. Please ensure Notebook 06 is executed.")

elif page == "Geographic & Product Insights":
    st.header("Geographic & Product Insights")
    
    geo_img = os.path.join(FIGURES_DIR, "25_top10_countries.png")
    if os.path.exists(geo_img):
        st.image(geo_img, caption="Top 10 Countries by Revenue", use_column_width=True)
        
    prod_img = os.path.join(FIGURES_DIR, "31_top15_products_net_revenue.png")
    if os.path.exists(prod_img):
        st.image(prod_img, caption="Top 15 Products by Net Revenue", use_column_width=True)
