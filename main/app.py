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
from supabase import create_client, Client
import supabase
import os

load_dotenv()

st.set_page_config(page_title="Invoice AI Agent", layout="wide")

st.title("Tabular Salesforce AI Agent")


url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

response = (
    supabase.table("invoices_table")
    .select("*")
    .execute()
)

data = pd.DataFrame(response.data)

data.rename(columns={
    "stockcode": "StockCode",
    "description": "Description",
    "quantity": "Quantity",
    "invoicedate": "InvoiceDate",
    "unitprice": "UnitPrice",
    "customerid": "CustomerID",
    "country": "Country",
    "invoiceno": "InvoiceNo"
}, inplace=True)

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
    return df

data = data_preprocess(data)
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

def least_country_sales(df, num=10):
    return df.groupby("Country")["Revenue"].sum().sort_values(ascending=True).head(num).to_dict()

def least_product_sales(df, num=10):
    return df.groupby("ProductID")["Revenue"].sum().sort_values(ascending=True).head(num).to_dict()
def least_product_revenue(df, num=10):
    return df.groupby("ProductID")["Revenue"].sum().sort_values(ascending=True).head(num).to_dict()

def cohort_analysis(df):
    df['FirstPurchaseMonth'] = df.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')
    cohort_revenue = df.groupby(['FirstPurchaseMonth', 'Month'])['Revenue'].sum().unstack(fill_value=0)
    return cohort_revenue

def customer_retention(df):
    df = df.sort_values('InvoiceDate')
    customer_first_last = df.groupby('CustomerID')['InvoiceDate'].agg(['min','max'])
    customer_first_last['retained'] = (customer_first_last['max'] - customer_first_last['min']).dt.days > 30
    retention_rate = customer_first_last['retained'].mean()
    return retention_rate
def monthly_growth(df):
    monthly = df.groupby('Month')['Revenue'].sum().sort_index()
    growth = monthly.pct_change().fillna(0) * 100
    return growth.to_dict()

def check_kpi_alerts(df, revenue_threshold=50000, invoices_threshold=100):
    metrics = Metrics(df)
    alerts = []
    if metrics['total_revenue'] < revenue_threshold:
        alerts.append(f"Revenue below threshold: ${metrics['total_revenue']:.2f}")
    if metrics['total_invoices'] < invoices_threshold:
        alerts.append(f"Invoices below threshold: {metrics['total_invoices']}")
    return alerts

class SalesforceState(TypedDict, total=False):
    query: str
    response: str

    metrics: Dict[str, float]
    top_selled_products: Dict[str, int]
    top_revenue_products: Dict[str, float]
    top_country_sales: Dict[str, float]
    least_country_sales: Dict[str, float]
    least_product_sales: Dict[str, float]
    least_product_revenue: Dict[str, float]

    monthly_sales_trend: Dict[str, float]
    monthly_stats: Dict[str, Any]
    monthly_growth: Dict[str, float]

    top_customers: Dict[str, float]
    rfm_summary: str
    rfm_clusters: List[Dict[str, Any]]
    customer_retention: float
    cohort_analysis: pd.DataFrame

    revenue_distribution: Dict[str, float]
    correlation_matrix: Dict[str, Dict[str, float]]

    alerts: List[str]


def ask_handler(state: SalesforceState) -> SalesforceState:
    state["query"] = st.session_state.user_query
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

def least_handler(state: SalesforceState) -> SalesforceState:
    state["least_country_sales"] = least_country_sales(data)
    state["least_product_sales"] = least_product_sales(data)
    state["least_product_revenue"] = least_product_revenue(data)
    return state


def retention_handler(state: SalesforceState) -> SalesforceState:
    state["customer_retention"] = customer_retention(data)
    state["cohort_analysis"] = cohort_analysis(data)
    return state

def growth_handler(state: SalesforceState) -> SalesforceState:
    state["monthly_growth"] = monthly_growth(data)
    return state

def alerts_handler(state: SalesforceState) -> SalesforceState:
    state["alerts"] = check_kpi_alerts(data)
    return state

def full_analysis_handler(state: SalesforceState) -> SalesforceState:
    state = ask_handler(state)
    state = metrics_handler(state)
    state = products_handler(state)
    state = customers_handler(state)
    state = stats_handler(state)
    state = least_handler(state)
    state = retention_handler(state)
    state = growth_handler(state)
    state = alerts_handler(state)
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
            model="cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
            temperature=0.5,
            api_key=os.getenv('api_key'),
            base_url="https://openrouter.ai/api/v1"
        )



saleforce_agent = graph.compile()


prompt = """You are a **Salesforce Invoice Analytics AI Agent** that assists business managers with invoice, sales, and customer insights.  
        
        Instructions:
        1. **Interpret the manager’s request** carefully.  
           - If the question requires data, call the appropriate tool(s).  
           - If multiple tools are needed, call them step by step.  
           - Do not make up numbers — only use tool outputs.  
        
        2. **Response Style**  
           - Always respond in a **clear, professional, insight-driven** tone.  
           - Use **plain English** — avoid technical jargon unless explicitly asked.  
           - Use **bullets or short paragraphs** for readability.  
           - When numbers are important, highlight them with **bold formatting** (e.g., “**$25,430** in revenue”).  
        
        3. **Flexibility**  
           - If the manager asks for raw data (e.g., “Show me the top 5 products”), return it in a **clean table or list**.  
           - If the manager asks for insights (e.g., “Give me an overview of sales health”), provide an **executive-style summary**:
             - **Executive Summary**: 2–3 sentences on sales health.  
             - **Key Insights**: bullet points on top products, customers, revenue.  
             - **Trends & Patterns**: monthly sales changes, distribution anomalies.  
             - **Customer Segmentation**: describe RFM clusters in simple terms.  
             - **Actionable Recommendations**: 3–5 business-oriented suggestions.  
        
        4. **Important Rules**  
           - Only use information from tools — never fabricate results.  
           - If the query is ambiguous, ask a clarifying question.  
           - Be concise and actionable — your audience is business managers, not data scientists.  
        
        You are not just summarizing — you are a **business copilot** that answers queries, runs the right tools, and communicates insights effectively.
        """

