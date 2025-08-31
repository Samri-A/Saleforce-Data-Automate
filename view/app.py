import langgraph
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import TypedDict, List, Dict, Any
from langgraph.graph.state import StateGraph
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()

# st.set_page_config(page_title="Invoice AI Agent", layout="wide")

# st.title("Tabular Salesforce AI Agent")

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
        "Revenue": "sum"                           
    }).rename(columns={
        "InvoiceDate": "Recency",
        "InvoiceNo": "Frequency",
        "Revenue": "Monetary"
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


# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# for msg in st.session_state["messages"]:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

# if query := st.chat_input("Ask me anything about invoices..."):
#     st.session_state["messages"].append({"role": "user", "content": query})
#     with st.chat_message("user"):
#         st.markdown(query)

def ask_handler(state):
    query = input("ask about the data: " )
    state["query"] = query
    return state

def metrics_handler(state: SalesforceState) -> SalesforceState:
    state["metrics"] = Metrics(data)
    return state

def products_handler(state: SalesforceState) -> SalesforceState:
    state["top_selled_products"] = top_selled_product().to_dict()
    state["top_revenue_products"] = top_revenue_product().to_dict()
    return state

def customers_handler(state: SalesforceState) -> SalesforceState:
    rfm = customer_segmentation(data)
    state["rfm_summary"] = rfm["summary"]
    state["rfm_clusters"] = rfm["clusters"]
    state["top_customers"] = top_customers_by_revenue(data)
    return state

def stats_handler(state: SalesforceState) -> SalesforceState:
    state["monthly_sales_trend"] = monthly_sales_trend(data).to_dict()
    state["monthly_stats"] = monthly_revenue_stats(data)
    state["revenue_distribution"] = revenue_distribution(data)
    state["correlation_matrix"] = correlation_matrix(data)
    return state

graph = StateGraph(state_schema=SalesforceState)


graph.add_node("ask", ask_handler)
graph.add_node("metrics", metrics_handler)
graph.add_node("products", products_handler)
graph.add_node("customers", customers_handler)
graph.add_node("stats", stats_handler)

graph.set_entry_point("ask")
graph.add_edge("ask", "metrics")
graph.add_edge("metrics", "products")
graph.add_edge("products", "customers")
graph.add_edge("customers", "stats")



llm = ChatOpenAI(
            model="openai/gpt-oss-20b:free",
            temperature=0,
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url="https://openrouter.ai/api/v1"
        )


def llm_handler(state: SalesforceState) -> SalesforceState:
    prompt = f"""
        You are a **Salesforce Invoice Analytics AI Agent**. 
        Your job is to summarize invoice and sales data in a way that is **clear, professional, and actionable**.
        
        The user’s request:
        - {state['query']}
        
        Here is the structured data extracted for you:
        - **Metrics (KPIs):** {state.get('metrics')}
        - **Top Products by Sales Volume:** {state.get('top_selled_products')}
        - **Top Products by Revenue:** {state.get('top_revenue_products')}
        - **Customer Segmentation (RFM):** {state.get('rfm_summary')}
        - **Monthly Revenue Stats:** {state.get('monthly_stats')}
        - **Revenue Distribution:** {state.get('revenue_distribution')}
        - **Top Customers:** {state.get('top_customers')}
        - **Correlation Matrix:** {state.get('correlation_matrix')}
        
        Instructions:
        1. Begin with a high-level overview (overall sales health, revenue, customers).
        2. Highlight key insights (best-selling products, top revenue drivers, high-value customers).
        3. Mention patterns or trends (monthly stats, distribution, anomalies if any).
        4. Explain the RFM segmentation briefly (clusters and what they mean).
        5. Conclude with actionable recommendations for business strategy.
        
        Keep your answer **concise, insightful, and business-oriented**.
        Avoid raw tables unless necessary—focus on interpretation.
        
        Now, write the summary.
    """
    state["response"] = llm.predict(prompt)
    return state

graph.add_node("llm", llm_handler)
graph.add_edge("stats", "llm")

saleforce_agent = graph.compile()

state = {}
final_state = saleforce_agent.invoke(state)
print(final_state.get("response"))
