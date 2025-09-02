import langgraph
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
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
from datetime import datetime
from langchain.tools import Tool
from flask_cors import CORS
now = datetime.now()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

load_dotenv()
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

resultponse = (
    supabase.table("invoices_table")
    .select("*")
    .execute()
)

data = pd.DataFrame(resultponse.data)

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
    return df.groupby("Description")["Revenue"].sum().sort_values(ascending=True).head(num).to_dict()

def least_product_revenue(df, num=10):
    return df.groupby("Description")["Revenue"].sum().sort_values(ascending=True).head(num).to_dict()

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

def check_kpi_alerts(df, revenue_thresulthold=50000, invoices_thresulthold=100):
    metrics = Metrics(df)
    alerts = []
    if metrics['total_revenue'] < revenue_thresulthold:
        alerts.append(f"Revenue below thresulthold: ${metrics['total_revenue']:.2f}")
    if metrics['total_invoices'] < invoices_thresulthold:
        alerts.append(f"Invoices below thresulthold: {metrics['total_invoices']}")
    return alerts

class SalesforceState(TypedDict, total=False):
    query: str
    resultponse: str
    is_it_chat: bool
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


def ask_handler(state: SalesforceState , user_query: str) -> SalesforceState:
    state["query"] = user_query
    return state

def metrics_handler(state: SalesforceState) -> SalesforceState:
    state["metrics"] = Metrics(data)
    return state

def products_handler(state: SalesforceState) -> SalesforceState:
    state["top_selled_products"] = top_selled_product().to_dict()
    state["top_revenue_products"] = top_revenue_product().to_dict()
    state["top_country_sales"] = top_country().to_dict()
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

def country_products_handler(state: SalesforceState, country: str, num: int = 10) -> SalesforceState:
    state[f"top_products_{country}"] = top_selled_product_by_country(country, num).to_dict()
    return state

def top_countries_handler(state: SalesforceState, num: int = 10) -> SalesforceState:
    state["top_country_sales"] = top_country(num).to_dict()
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


from langchain.tools import Tool

# Metrics Tool
def get_metrics_summary(state=None) -> str: 
    metrics = Metrics(data)
    summary = (
        f"Total Revenue: ${metrics['total_revenue']:.2f}\n"
        f"Total Invoices: {metrics['total_invoices']}\n"
        f"Total Customers: {metrics['total_customers']}"
    )
    return summary

metrics_tool = Tool(
    name="MetricsTool",
    func=get_metrics_summary,
    description="Returns total revenue, invoices, and customer count."
)

# Products Tool
def get_products_summary(state=None) -> str:
    top_sold = top_selled_product().head(5).to_dict()
    top_revenue = top_revenue_product().head(5).to_dict()
    top_sold_str = "\n".join([f"{k}: {v}" for k, v in top_sold.items()])
    top_rev_str = "\n".join([f"{k}: ${v:.2f}" for k, v in top_revenue.items()])
    return f"Top Selling Products:\n{top_sold_str}\n\nTop Revenue Products:\n{top_rev_str}"

products_tool = Tool(
    name="ProductsTool",
    func=get_products_summary,
    description="Returns top-selling and top-revenue products in a readable format."
)


def get_top_countries_summary(state=None) -> str:
    top_countries = top_country().head(5).to_dict()
    top_countries_str = "\n".join([f"{k}: ${v:.2f}" for k, v in top_countries.items()])
    return f"Top Countries by Revenue:\n{top_countries_str}"

top_countries_tool = Tool(
    name="TopCountriesTool",
    func=get_top_countries_summary,
    description="Returns top countries by revenue."
)

def get_customers_summary(state=None) -> str:
    rfm = customer_segmentation(data)
    top_customers = top_customers_by_revenue(data)
    top_customers_str = "\n".join([f"{k}: ${v:.2f}" for k, v in dict(list(top_customers.items())[:5]).items()])
    return f"RFM Summary:\n{rfm['summary']}\n\nTop Customers by Revenue:\n{top_customers_str}"

customers_tool = Tool(
    name="CustomersTool",
    func=get_customers_summary,
    description="Returns RFM summary and top customers by revenue."
)

def get_stats_summary(state=None) -> str:
    monthly_trend = monthly_sales_trend(data).head(5).to_dict()
    monthly_trend_str = "\n".join([f"{k}: ${v:.2f}" for k, v in monthly_trend.items()])
    return f"Monthly Sales Trend (first 5 months):\n{monthly_trend_str}"

stats_tool = Tool(
    name="StatsTool",
    func=get_stats_summary,
    description="Provides monthly sales trends in a readable format."
)

def get_least_summary(state=None) -> str:
    least_country = least_country_sales(data)
    least_product = least_product_sales(data)
    least_country_str = "\n".join([f"{k}: ${v:.2f}" for k, v in dict(list(least_country.items())[:5]).items()])
    least_product_str = "\n".join([f"{k}: ${v:.2f}" for k, v in dict(list(least_product.items())[:5]).items()])
    return f"Least Performing Countries:\n{least_country_str}\n\nLeast Performing Products:\n{least_product_str}"

least_tool = Tool(
    name="LeastTool",
    func=get_least_summary,
    description="Returns least performing countries and products."
)

def get_retention_summary(state=None) -> str:
    retention = customer_retention(data)
    return f"Customer Retention Rate: {retention*100:.2f}%"

retention_tool = Tool(
    name="RetentionTool",
    func=get_retention_summary,
    description="Returns the customer retention rate."
)

def get_growth_summary(state=None) -> str:
    growth = monthly_growth(data)
    growth_str = "\n".join([f"{k}: {v:.2f}%" for k, v in dict(list(growth.items())[:5]).items()])
    return f"Monthly Growth (first 5 months):\n{growth_str}"

growth_tool = Tool(
    name="GrowthTool",
    func=get_growth_summary,
    description="Provides monthly revenue growth percentages."
)

def get_alerts_summary(state=None) -> str:
    alerts = check_kpi_alerts(data)
    if not alerts:
        return "No KPI alerts detected."
    return "KPI Alerts:\n" + "\n".join(alerts)

alerts_tool = Tool(
    name="AlertsTool",
    func=get_alerts_summary,
    description="Checks for KPI alerts and reports them."
)

from pydantic import BaseModel, Field

class CountryProductsInput(BaseModel):
    country: str = Field(..., description="The country to fetch top products for")
    num: int = Field(10, description="Number of products to return (default 10)")


def get_country_products_summary(input_str: str) -> str:
    """Input should be 'country,num' e.g. 'USA,5'"""
    parts = input_str.split(",")
    country = parts[0].strip()
    num = int(parts[1]) if len(parts) > 1 else 10

    products = top_selled_product_by_country(country, num).to_dict()
    products_str = "\n".join([f"{k}: {v}" for k, v in products.items()])
    return f"Top {num} Products in {country}:\n{products_str}"

country_products_tool = Tool(
    name="CountryProductsTool",
    func=get_country_products_summary,
    description="Input format: 'Country,Num'. Example: 'USA,5' → returns top 5 products in USA."
)


tools = [
    metrics_tool,
    products_tool,
    country_products_tool,
    top_countries_tool,
    customers_tool,
    stats_tool,
    least_tool,
    retention_tool,
    growth_tool,
    alerts_tool
]


llm = ChatOpenAI(
            model="mistralai/mistral-small-3.2-24b-instruct:free",
            temperature=0.5,
            api_key=os.getenv('api_key'),
            base_url="https://openrouter.ai/api/v1"
        )

def is_chat(state: SalesforceState):
    pass


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

graph = StateGraph(state_schema=SalesforceState)


graph.add_node("ask", ask_handler)
graph.add_node("full_analysis", full_analysis_handler)

graph.set_entry_point("ask")
graph.add_edge("ask", "full_analysis")


saleforce_agent = graph.compile()


agent = initialize_agent(
   
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
     memory= memory,
    handle_parsing_errors = True, 
    verbose=True
)
# response = agent.run("Show me top purchasing countries")
# print(response)


from flask import Flask
from flask import request

app = Flask(__name__)
CORS(app)


@app.route('/chat', methods=['POST'])
def index():
        user_input = request.json['user_input']
        response = agent.run(user_input)
        return {"response" : response}

if __name__ == '__main__':
    app.run(debug=True , port=5000)