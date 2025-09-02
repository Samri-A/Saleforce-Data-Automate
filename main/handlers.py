from typing import TypedDict, List, Dict, Any
import pandas as pd
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
