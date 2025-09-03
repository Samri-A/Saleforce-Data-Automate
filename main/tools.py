from langchain.tools import Tool , tool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import tempfile
import pandas as pd
from analysis import (
    Metrics,
    top_selled_product,
    top_revenue_product,
    top_country,
    customer_segmentation,
    monthly_sales_trend,
    least_country_sales,
    least_product_sales,
    top_customers_by_revenue,
    customer_retention,
    monthly_growth,
    check_kpi_alerts
)

def metrics_tool(df):
    def get_metrics_summary(_):
        metrics = Metrics(df)
        return (
            f"Total Revenue: ${metrics['total_revenue']:.2f}\n"
            f"Total Invoices: {metrics['total_invoices']}\n"
            f"Total Customers: {metrics['total_customers']}"
        )
    return Tool(
        name="MetricsTool",
        func=get_metrics_summary,
        description="Returns total revenue, invoices, and customer count."
    )

def products_tool(df):
    def get_products_summary(_):
        top_sold = top_selled_product(df).to_dict()
        top_revenue = top_revenue_product(df).to_dict()
        top_sold_str = "\n".join([f"{k}: {v}" for k, v in top_sold.items()])
        top_rev_str = "\n".join([f"{k}: ${v:.2f}" for k, v in top_revenue.items()])
        return f"Top Selling Products:\n{top_sold_str}\n\nTop Revenue Products:\n{top_rev_str}"
    return Tool(
        name="ProductsTool",
        func=get_products_summary,
        description="Returns top-selling and top-revenue products."
    )


def top_countries_tool(df):
    def get_top_countries_summary(_):
        top_countries = top_country(df).to_dict()
        top_countries_str = "\n".join([f"{k}: ${v:.2f}" for k, v in top_countries.items()])
        return f"Top Countries by Revenue:\n{top_countries_str}"
    return Tool(
        name="TopCountriesTool",
        func=get_top_countries_summary,
        description="Returns top countries by revenue."
    )

def customers_tool(df):
    def get_customers_summary(_):
        rfm = customer_segmentation(df)
        top_customers = top_customers_by_revenue(df)
        top_customers_str = "\n".join([f"{k}: ${v:.2f}" for k, v in dict(list(top_customers.items())[:5]).items()])
        return f"RFM Summary:\n{rfm['summary']}\n\nTop Customers by Revenue:\n{top_customers_str}"
    return Tool(
        name="CustomersTool",
        func=get_customers_summary,
        description="Returns RFM summary and top customers by revenue."
    )

def stats_tool(df):
    def get_stats_summary(_):
        monthly_trend = monthly_sales_trend(df).to_dict()
        monthly_trend_str = "\n".join([f"{k}: ${v:.2f}" for k, v in monthly_trend.items()])
        return f"Monthly Sales Trend (first 5 months):\n{monthly_trend_str}"
    return Tool(
        name="StatsTool",
        func=get_stats_summary,
        description="Provides monthly sales trends."
    )

def least_tool(df):
    def get_least_summary(_):
        least_country_dict = least_country_sales(df)
        least_product_dict = least_product_sales(df)
        least_country_str = "\n".join([f"{k}: ${v:.2f}" for k, v in dict(list(least_country_dict.items())[:5]).items()])
        least_product_str = "\n".join([f"{k}: ${v:.2f}" for k, v in dict(list(least_product_dict.items())[:5]).items()])
        return f"Least Performing Countries:\n{least_country_str}\n\nLeast Performing Products:\n{least_product_str}"
    return Tool(
        name="LeastTool",
        func=get_least_summary,
        description="Returns least performing countries and products."
    )

def retention_tool(df):
    def get_retention_summary(_):
        retention = customer_retention(df)
        return f"Customer Retention Rate: {retention*100:.2f}%"
    return Tool(
        name="RetentionTool",
        func=get_retention_summary,
        description="Returns the customer retention rate."
    )

def growth_tool(df):
    def get_growth_summary(_):
        growth = monthly_growth(df)
        growth_str = "\n".join([f"{k}: {v:.2f}%" for k, v in dict(list(growth.items())).items()])
        return f"Monthly Growth (first 5 months):\n{growth_str}"
    return Tool(
        name="GrowthTool",
        func=get_growth_summary,
        description="Provides monthly revenue growth percentages."
    )

def alerts_tool(df):
    def get_alerts_summary(_):
        alerts = check_kpi_alerts(df)
        if not alerts:
            return "No KPI alerts detected."
        return "KPI Alerts:\n" + "\n".join(alerts)
    return Tool(
        name="AlertsTool",
        func=get_alerts_summary,
        description="Returns KPI alerts summary."
    )



def _save_plot(title: str = "plot"):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.title(title)
    plt.savefig(tmp_file.name, format="png", bbox_inches="tight")
    plt.close()
    return f"Graph saved at {tmp_file.name}"


@tool("bar_chart", return_direct=True)
def bar_chart(data: dict, title: str = "Bar Chart"):
    """
    Generate a bar chart.
    Args:
        data (dict): key=label, value=numeric value
        title (str): chart title
    Returns: path to PNG image
    """
    plt.figure(figsize=(8,6))
    pd.Series(data).plot(kind="bar")
    return _save_plot(title)

@tool("line_chart", return_direct=True)
def line_chart(data: dict, title: str = "Line Chart"):
    """
    Generate a line chart.
    Args:
        data (dict): key=x-axis, value=numeric
        title (str): chart title
    """
    plt.figure(figsize=(8,6))
    pd.Series(data).plot(kind="line", marker="o")
    return _save_plot(title)


@tool("area_chart", return_direct=True)
def area_chart(data: dict, title: str = "Area Chart"):
    """
    Generate an area chart.
    Args:
        data (dict): key=x-axis, value=numeric
    """
    plt.figure(figsize=(8,6))
    pd.Series(data).plot(kind="area", alpha=0.4)
    return _save_plot(title)


@tool("histogram", return_direct=True)
def histogram(values: list, bins: int = 10, title: str = "Histogram"):
    """
    Generate a histogram from numeric values.
    Args:
        values (list): numeric data
        bins (int): number of bins
    """
    plt.figure(figsize=(8,6))
    plt.hist(values, bins=bins, alpha=0.7)
    return _save_plot(title)


@tool("box_plot", return_direct=True)
def box_plot(values: list, title: str = "Box Plot"):
    """
    Generate a box plot.
    Args:
        values (list): numeric data
    """
    plt.figure(figsize=(6,6))
    plt.boxplot(values, vert=True, patch_artist=True)
    return _save_plot(title)

@tool("scatter_plot", return_direct=True)
def scatter_plot(x: list, y: list, labels: list = None, title: str = "Scatter Plot"):
    """
    Generate a scatter plot.
    Args:
        x (list): x-axis values
        y (list): y-axis values
        labels (list): optional categories for coloring
    """
    plt.figure(figsize=(8,6))
    if labels:
        sns.scatterplot(x=x, y=y, hue=labels, palette="Set2")
    else:
        plt.scatter(x, y, alpha=0.7)
    return _save_plot(title)


@tool("heatmap", return_direct=True)
def heatmap(matrix: dict, title: str = "Heatmap"):
    """
    Generate a heatmap from a correlation matrix or pivot table.
    Args:
        matrix (dict): nested dict or 2D-like structure
    """
    df = pd.DataFrame(matrix)
    plt.figure(figsize=(8,6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm")
    return _save_plot(title)


@tool("pie_chart", return_direct=True)
def pie_chart(data: dict, title: str = "Pie Chart"):
    """
    Generate a pie chart.
    Args:
        data (dict): key=label, value=numeric value
    """
    plt.figure(figsize=(7,7))
    plt.pie(data.values(), labels=data.keys(), autopct="%1.1f%%", startangle=90)
    return _save_plot(title)



@tool("network_graph", return_direct=True)
def network_graph(edges: list, title: str = "Network Graph"):
    """
    Generate a network/transaction graph.
    Args:
        edges (list): list of tuples (source, target, weight)
    """
    G = nx.DiGraph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w, label=str(w))
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8,6))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", arrows=True)
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    return _save_plot(title)
