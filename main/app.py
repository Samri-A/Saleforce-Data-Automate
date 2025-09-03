from flask import Flask , request
from langchain.memory import ConversationSummaryMemory
from langchain.agents import AgentType , initialize_agent 
from flask_cors import CORS
from data_processing import load_and_prepare
from analysis import Metrics , sales_country , monthly_sales_trend , monthly_growth
from tools import (
    metrics_tool,
    products_tool,
    top_countries_tool,
    customers_tool,
    stats_tool,
    least_tool,
    retention_tool,
    growth_tool,
    alerts_tool
)
from handlers import full_analysis_handler , SalesforceState , insight_handler
from config import llm
import langgraph
from langgraph.graph.state import StateGraph

app = Flask(__name__)
CORS(app)



memory = ConversationSummaryMemory( llm=llm , memory_key="chat_history" , return_messages=True )


def initialize(table_name: str = "invoices_table"):
    df = load_and_prepare(table_name)
    return df

df = initialize()
tools  = [
    metrics_tool(df),
    products_tool(df),
    top_countries_tool(df),
    customers_tool(df),
    stats_tool(df),
    least_tool(df),
    retention_tool(df),
    growth_tool(df),
    alerts_tool(df)
    ]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory= memory,
    handle_parsing_errors = True, 
    verbose=True
)

@app.route('/chat', methods=['POST'])
def index():
        user_input = request.json['user_input']
        response = agent.invoke(user_input)
        return {"response": response}

@app.route('/dashboard', methods=['GET'])
def dashboard():
    metrics = Metrics(df)
    monthly_trend = monthly_sales_trend(df).to_dict()
    region_sales = sales_country(df).to_dict()
    # growth = monthly_growth(df)
    # print(growth)
    report()
    return (
            {"Total Revenue": f"${metrics['total_revenue']:.2f}",
            "Total Invoices": metrics['total_invoices'],
            "Total Customers": metrics['total_customers'] , 
            "Country Sales" : region_sales ,
            "Monthly Trend" : monthly_trend,
            # "Monthly Growth" : growth
            }
        )


# @app.route('/report' , methods=['GET'])
def report():
    graph =StateGraph(state_schema= SalesforceState)
    graph.add_node("full_analysis", lambda state: full_analysis_handler(state= state, df=df))
    graph.add_node("insight_handler" , lambda state: insight_handler( state=state, llm=llm))
    
    graph.set_entry_point("full_analysis")
    graph.add_edge("full_analysis", "insight_handler")
    saleforceGraph = graph.compile()
    state = {}
    result = saleforceGraph.invoke(state)
    print(result["result"])
    return {"message": "Report generated successfully"}

if __name__ == '__main__':
    app.run(debug=True , port=5000)
    

