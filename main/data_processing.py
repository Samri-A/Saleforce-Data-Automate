import pandas as pd
import datetime as dt
from typing import Optional
from config import supabase
import asyncio

def load_data_from_supabase(table_name: str = "invoices_table") -> pd.DataFrame:
    if supabase is None:
        raise RuntimeError("Supabase client not configured. Set SUPABASE_URL and SUPABASE_KEY in env.")
    response = supabase.table(table_name).select("*").execute()
    df = pd.DataFrame(response.data)
    df.rename(columns={
        "stockcode": "StockCode",
        "description": "Description",
        "quantity": "Quantity",
        "invoicedate": "InvoiceDate",
        "unitprice": "UnitPrice",
        "customerid": "CustomerID",
        "country": "Country",
        "invoiceno": "InvoiceNo"
    }, inplace=True)
    return df


def data_preprocess(df: pd.DataFrame) -> pd.DataFrame:
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


def load_and_prepare(table_name: str = "invoices_table") -> pd.DataFrame:
    df = load_data_from_supabase(table_name)
    return data_preprocess(df)


