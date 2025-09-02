import os
from dotenv import load_dotenv
from datetime import datetime
from supabase import create_client, Client
from langchain_openai import ChatOpenAI

load_dotenv()

now = datetime.now()

URL: str = os.environ.get("SUPABASE_URL")
KEY: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(URL, KEY) if URL and KEY else None

llm: ChatOpenAI = ChatOpenAI(
            model="mistralai/mistral-small-3.2-24b-instruct:free",
            temperature=0.5,
            api_key=os.getenv('api_key'),
            base_url="https://openrouter.ai/api/v1"
        )
