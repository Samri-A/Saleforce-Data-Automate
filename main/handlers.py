from typing import TypedDict, List, Dict, Any
import pandas as pd
from tools import (bar_chart, line_chart, area_chart, histogram, 
         box_plot, scatter_plot, heatmap, pie_chart, network_graph)
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

### Instructions:
1. Begin the report with **overall KPIs** (Revenue, Invoices, Customers).
   - If alerts exist, highlight them clearly.
   - Use `bar_chart` or `pie_chart` to show KPI breakdowns.

2. Show **top performing products and countries** with `bar_chart`.
   - Also show **least performing ones**.

3. Show **monthly sales trend** and **growth rates** with `line_chart` or `area_chart`.

4. Use `box_plot` and `histogram` to visualize **revenue distribution**.

5. Visualize **customer segmentation**:
   - Use `scatter_plot` (Recency vs Frequency vs Monetary, colored by cluster).
   - Use `bar_chart` to show number of customers per cluster.

6. Show **top customers by revenue** with `bar_chart`.

7. Show **correlation matrix** as a `heatmap`.

8. If cohort analysis is provided, visualize it with a `heatmap`.

9. End with **insights & recommendations**, based on trends and alerts.

### Output Requirements:
- Call the correct visualization tools (`bar_chart`, `line_chart`, `area_chart`, 
  `histogram`, `box_plot`, `scatter_plot`, `heatmap`, `pie_chart`, `network_graph`) 
  for each relevant dataset.
- Return a structured business report in natural language along with references 
  to the generated graph images.
"""

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
    tools = [bar_chart, line_chart, area_chart, histogram, 
             box_plot, scatter_plot, heatmap, pie_chart, network_graph]

    if "steps_completed" not in state:
        state["steps_completed"] = []

    prompt = SALESFORCE_REPORT_PROMPT.format(**state)
    memory = ConversationBufferMemory(memory_key="chat_history" , return_messages=True) 
    agent = initialize_agent(
        tools, llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )
    state["result"] = agent.invoke(prompt + f"\nSteps completed: {state['steps_completed']}")
    
    state["steps_completed"].append("overall_kpis")
    
    return state
