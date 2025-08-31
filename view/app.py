import langgraph
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import TypedDict, List, Dict, Any


st.set_page_config(page_title="Invoice AI Agent", layout="wide")

st.title("Tabular Salesforce AI Agent")

data = pd.read_csv("../data/data.csv" ,  encoding="latin1" )

def data_preprocess(df):
    df.dropna(subset=["Description"], inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Month'] = df['InvoiceDate'].dt.to_period('M')
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    df['CustomerID'] = df['CustomerID'].astype(str)
    df['CustomerID'] = df['CustomerID'].fillna('Unknown')
    invalid_rows = df[(df['Quantity'] <= 0) | (df['UnitPrice'] <= 0)].index
    df.drop(index=invalid_rows, inplace=True)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

data_preprocess(data)


if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if query := st.chat_input("Ask me anything about invoices..."):
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

def Metrics(df=data):
    total_revenue = df["Revenue"].sum()
    total_invoices = df["InvoiceNo"].nunique()
    total_customers = df["CustomerID"].nunique()
    return {'total_revenue' : total_revenue , 'total_invoices' : total_invoices , 'total_customers' : total_customers}

def top_selled_product(num = 10 , df = data):
    top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(num)
    return top_products

def top_revenue_product(num=10,df=data):
    top_revenue_products = df.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(num)
    return top_revenue_products

def top_selled_product_by_country(country , num=10 ,df = data):
    top_selled_product = df[df["Country"] == country].groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(num)
    return top_selled_product

def top_country(num = 10 , df=data):
    country_sales = df.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(num)
    return country_sales

def monthly_sales_trend(df=data):
     monthly_sales = df.groupby('Month')["Revenue"].sum()
     return monthly_sales

def customer_segmentation(df, n_clusters=4):
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df.reset_index(drop=True)
    NOW = df["InvoiceDate"].max() + dt.timedelta(days=1)
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (NOW - x.max()).days,  
        "InvoiceNo": "count",                          
        "TotalAmount": "sum"                           
    }).rename(columns={
        "InvoiceDate": "Recency",
        "InvoiceNo": "Frequency",
        "TotalAmount": "Monetary"
    })
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
    cluster_summaries = []
    for cid, group in rfm.groupby("Cluster"):
        cluster_summaries.append({
            "cluster_id": int(cid),
            "num_customers": int(len(group)),
            "avg_recency_days": float(group["Recency"].mean()),
            "avg_frequency": float(group["Frequency"].mean()),
            "avg_monetary": float(group["Monetary"].mean()),
            "min_monetary": float(group["Monetary"].min()),
            "max_monetary": float(group["Monetary"].max())
        })

    summary_text = f"I identified {n_clusters} customer segments using RFM analysis.\n"
    for c in cluster_summaries:
        summary_text += (
            f"- Cluster {c['cluster_id']} has {c['num_customers']} customers, "
            f"with an average recency of {c['avg_recency_days']:.1f} days, "
            f"buying {c['avg_frequency']:.1f} times, and spending on average ${c['avg_monetary']:.2f}.\n"
        )

    return {
        "summary": summary_text.strip(),
        "clusters": cluster_summaries,
    }


def monthly_revenue_stats(df):
    monthly = df.groupby("Month")["Revenue"].sum()
    return {
        "max_month": str(monthly.idxmax()),
        "min_month": str(monthly.idxmin()),
        "avg_monthly_revenue": float(monthly.mean()),
        "monthly_values": monthly.to_dict()
    }


def correlation_matrix(df):
    return df[["Quantity", "UnitPrice", "Revenue"]].corr().to_dict()

def top_customers_by_revenue(df, num=10):
    return df.groupby("CustomerID")["Revenue"].sum().sort_values(ascending=False).head(num).to_dict()

def revenue_distribution(df):
    return {
        "min": float(df["Revenue"].min()),
        "max": float(df["Revenue"].max()),
        "mean": float(df["Revenue"].mean()),
        "median": float(df["Revenue"].median()),
        "std": float(df["Revenue"].std()),
        "25%": float(df["Revenue"].quantile(0.25)),
        "50%": float(df["Revenue"].quantile(0.50)),
        "75%": float(df["Revenue"].quantile(0.75)),
    }


class SalesforceState(TypedDict, total=False):
    query: str                
    response: str      

    metrics: Dict[str, float] 
    top_selled_products: Dict[str, int]       
    top_revenue_products: Dict[str, float]   
    top_country_sales: Dict[str, float]      
    monthly_sales_trend: Dict[str, float]
    monthly_stats: Dict[str, Any]         
    top_customers: Dict[str, float]           
    rfm_summary: str                          
    rfm_clusters: List[Dict[str, Any]]        
    revenue_distribution: Dict[str, float]         
    correlation_matrix: Dict[str, Dict[str, float]]   

