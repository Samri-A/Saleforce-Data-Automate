import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
from typing import List, Dict, Any


def Metrics(df: pd.DataFrame) -> Dict[str, Any]:
    total_revenue = df["Revenue"].sum()
    total_invoices = df["InvoiceNo"].nunique()
    total_customers = df["CustomerID"].nunique()
    return {'total_revenue': total_revenue, 'total_invoices': total_invoices, 'total_customers': total_customers}


def top_selled_product(df: pd.DataFrame, num: int = 10) -> pd.Series:
    return df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(num)


def top_revenue_product(df: pd.DataFrame, num: int = 10) -> pd.Series:
    return df.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(num)


def top_selled_product_by_country(df: pd.DataFrame, country: str, num: int = 10) -> pd.Series:
    return df[df["Country"] == country].groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(num)


def top_country(df: pd.DataFrame, num: int = 10) -> pd.Series:
    return df.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(num)


def monthly_sales_trend(df: pd.DataFrame) -> pd.Series:
    trend = df.groupby('Month')["Revenue"].sum()
    trend.index = trend.index.astype(str)
    return trend

def sales_country(df: pd.DataFrame, num: int = 10) -> pd.Series:
    return df.groupby('Country')['Revenue'].sum().sort_values(ascending=False)

def customer_segmentation(df: pd.DataFrame, n_clusters: int = 4) -> Dict[str, Any]:
    df = df.copy()
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


def monthly_revenue_stats(df: pd.DataFrame) -> Dict[str, Any]:
    monthly = df.groupby("Month")["Revenue"].sum()
    return {
        "max_month": str(monthly.idxmax()),
        "min_month": str(monthly.idxmin()),
        "avg_monthly_revenue": float(monthly.mean()),
        "monthly_values": monthly.to_dict()
    }


def correlation_matrix(df: pd.DataFrame) -> Dict[str, Any]:
    return df[["Quantity", "UnitPrice", "Revenue"]].corr().to_dict()


def top_customers_by_revenue(df: pd.DataFrame, num: int = 10) -> Dict[str, float]:
    return df.groupby("CustomerID")["Revenue"].sum().sort_values(ascending=False).head(num).to_dict()


def revenue_distribution(df: pd.DataFrame) -> Dict[str, float]:
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


def least_country_sales(df: pd.DataFrame, num: int = 10) -> Dict[str, float]:
    return df.groupby("Country")["Revenue"].sum().sort_values(ascending=True).head(num).to_dict()


def least_product_sales(df: pd.DataFrame, num: int = 10) -> Dict[str, float]:
    return df.groupby("Description")["Revenue"].sum().sort_values(ascending=True).head(num).to_dict()


def cohort_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['FirstPurchaseMonth'] = df.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')
    cohort_revenue = df.groupby(['FirstPurchaseMonth', 'Month'])['Revenue'].sum().unstack(fill_value=0)
    return cohort_revenue


def customer_retention(df: pd.DataFrame) -> float:
    df = df.sort_values('InvoiceDate')
    customer_first_last = df.groupby('CustomerID')['InvoiceDate'].agg(['min', 'max'])
    customer_first_last['retained'] = (customer_first_last['max'] - customer_first_last['min']).dt.days > 30
    retention_rate = customer_first_last['retained'].mean()
    return retention_rate


def monthly_growth(df: pd.DataFrame) -> Dict[str, float]:
    monthly = df.groupby('Month')['Revenue'].sum().sort_index()
    growth = monthly.pct_change().fillna(0) * 100
    return growth.to_dict()


def check_kpi_alerts(df: pd.DataFrame, revenue_thresulthold: float = 50000, invoices_thresulthold: int = 100) -> List[str]:
    metrics = Metrics(df)
    alerts = []
    if metrics['total_revenue'] < revenue_thresulthold:
        alerts.append(f"Revenue below thresulthold: ${metrics['total_revenue']:.2f}")
    if metrics['total_invoices'] < invoices_thresulthold:
        alerts.append(f"Invoices below thresulthold: {metrics['total_invoices']}")
    return alerts
