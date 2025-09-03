from typing import TypedDict, List, Dict, Any
import pandas as pd
from tools import ( 
        bar_chart,
        line_chart,
        area_chart,
        histogram,
        box_plot,
        scatter_plot,
        heatmap,
        pie_chart,
        network_graph)
from analysis import (
    Metrics,
    top_selled_product,
    top_selled_product_by_country,
    top_revenue_product,
    top_country,
    customer_segmentation,
    monthly_sales_trend,
    monthly_revenue_stats,
    revenue_distribution,
    correlation_matrix,
    top_customers_by_revenue,
    least_country_sales,
    least_product_sales,
    least_product_sales as least_product_revenue,
    cohort_analysis,
    customer_retention,
    monthly_growth,
    check_kpi_alerts
)
from langchain.agents import initialize_agent , AgentType
from langchain.memory import ConversationBufferMemory 

graph_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

SALESFORCE_REPORT_PROMPT = """
You are a Salesforce data analysis assistant. 
You are provided with the results of a full sales analysis pipeline. 
Your task is to generate a detailed business intelligence report 
with clear insights, KPIs, and visualizations using the available graph tools.

### Analysis Context:
You are given a state object with:
- KPIs: {metrics}
- Alerts: {alerts}
- Top Products (by sales and revenue): {top_selled_products}, {top_revenue_products}
- Top Countries: {top_country_sales}
- Least performing Products/Countries: {least_product_sales}, {least_country_sales}
- Monthly Trends: {monthly_sales_trend}, growth rates: {monthly_growth}
- Monthly Stats: {monthly_stats}
- Customer Segmentation: {rfm_summary}, clusters: {rfm_clusters}
- Top Customers: {top_customers}
- Customer Retention: {customer_retention}
- Cohort Analysis: {cohort_analysis}
- Revenue Distribution: {revenue_distribution}
- Correlation Matrix: {correlation_matrix}


### Output Requirements:
- Use correct graphs if available : {graphs}
- Return a structured business report along with references 
  to the graph images.


"""

# ### Available Graphs:
# You already have generated graphs stored at these locations:
# {graphs}

# ### Instructions for Graphs:
# - When you create a new chart, always give it a clear, descriptive title 
#   (e.g., "Monthly Revenue Trend", "Top 10 Customers by Revenue").
# - If a graph with that title already exists in {graphs}, just reference it 
#   instead of redrawing.



class SalesforceState(TypedDict, total=False):
    query: str
    result: str
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
    graphs: str


def ask_handler(state: SalesforceState, user_query: str) -> SalesforceState:
    state["query"] = user_query
    return state


def metrics_handler(state: SalesforceState, df: pd.DataFrame) -> SalesforceState:
    state["metrics"] = Metrics(df)
    return state


def products_handler(state: SalesforceState, df: pd.DataFrame) -> SalesforceState:
    state["top_selled_products"] = top_selled_product(df).to_dict()
    state["top_revenue_products"] = top_revenue_product(df).to_dict()
    state["top_country_sales"] = top_country(df).to_dict()
    return state


def customers_handler(state: SalesforceState, df: pd.DataFrame) -> SalesforceState:
    rfm = customer_segmentation(df)
    state["rfm_summary"] = rfm["summary"]
    state["rfm_clusters"] = rfm["clusters"]
    state["top_customers"] = top_customers_by_revenue(df)
    return state


def stats_handler(state: SalesforceState, df: pd.DataFrame) -> SalesforceState:
    state["monthly_sales_trend"] = monthly_sales_trend(df).to_dict()
    state["monthly_stats"] = monthly_revenue_stats(df)
    state["revenue_distribution"] = revenue_distribution(df)
    state["correlation_matrix"] = correlation_matrix(df)
    return state


def least_handler(state: SalesforceState, df: pd.DataFrame) -> SalesforceState:
    state["least_country_sales"] = least_country_sales(df)
    state["least_product_sales"] = least_product_sales(df)
    state["least_product_revenue"] = least_product_revenue(df)
    return state


def retention_handler(state: SalesforceState, df: pd.DataFrame) -> SalesforceState:
    state["customer_retention"] = customer_retention(df)
    state["cohort_analysis"] = cohort_analysis(df)
    return state


def growth_handler(state: SalesforceState, df: pd.DataFrame) -> SalesforceState:
    state["monthly_growth"] = monthly_growth(df)
    return state


def alerts_handler(state: SalesforceState, df: pd.DataFrame) -> SalesforceState:
    state["alerts"] = check_kpi_alerts(df)
    return state


def country_products_handler(state: SalesforceState, df: pd.DataFrame, country: str, num: int = 10) -> SalesforceState:
    state[f"top_products_{country}"] = top_selled_product_by_country(df, country, num).to_dict()
    return state


def top_countries_handler(state: SalesforceState, df: pd.DataFrame, num: int = 10) -> SalesforceState:
    state["top_country_sales"] = top_country(df, num).to_dict()
    return state


def full_analysis_handler(state: SalesforceState, df: pd.DataFrame) -> SalesforceState:
    state = metrics_handler(state, df)
    state = products_handler(state, df)
    state = customers_handler(state, df)
    state = stats_handler(state, df)
    state = least_handler(state, df)
    state = retention_handler(state, df)
    state = growth_handler(state, df)
    state = alerts_handler(state, df)
    return state



def insight_handler(state: SalesforceState, llm) -> SalesforceState:
    prompt = SALESFORCE_REPORT_PROMPT.format(**state)
    state["result"] = llm.invoke(prompt)
    return state



def draw_all_graphs(state: SalesforceState, llm) -> Dict[str, str]:
    prompt = f"""
    Generate all graphs from the SalesforceState data and return
    a dictionary mapping chart titles to their saved file paths.
    
    The following graphs will be generated:
    
        - "Top Products by Sales", "bar_chart",{ state.get("top_selled_products")}
        - "Top Products by Revenue", "bar_chart", {state.get("top_revenue_products")}
        - "Monthly Sales Trend", "line_chart", {state.get("monthly_sales_trend")}
        - "Monthly Revenue Stats", "area_chart", {state.get("monthly_stats")}
        - "Revenue Distribution", "pie_chart", {state.get("revenue_distribution")}
        - "Correlation Matrix", "heatmap", {state.get("correlation_matrix")}
        - "Customer Retention Cohort", "scatter_plot", {state.get("cohort_analysis")}
        - "Top Customers by Revenue", "bar_chart", {state.get("top_customers")}
        - "Least Performing Products", "bar_chart", {state.get("least_product_sales")}
        - "Least Performing Countries", "bar_chart", {state.get("least_country_sales")}
    
    Returns:
        Dict[str, str]: Mapping of all chart titles to their corresponding file paths.
    """

    tools = [
        bar_chart,
        line_chart,
        area_chart,
        histogram,
        box_plot,
        scatter_plot,
        heatmap,
        pie_chart,
        network_graph
    ]
    
    graph_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=graph_memory,
        verbose=True
    )

    graph_results = graph_agent.invoke(prompt)
    state["graphs"] = graph_results
    return state
