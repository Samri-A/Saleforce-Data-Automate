# Saleforce Data Automate

Saleforce Data Automate is a small full-stack project that provides
Salesforce-style sales analysis and report automation using a Langchain , Langgraph and Flask for
backend  and a React + TypeScrip
frontend (dashboard + chat UI).

This README explains the project structure, environment variables,
local setup (PowerShell), API endpoints, and developer notes to get
you started.

## Quick summary

- Backend: Flask app in `main/` that loads sales data (Supabase), runs
	analysis (RFM, monthly trends, growth, cohort, KPIs) and exposes
	endpoints used by the UI. The backend also composes a LangGraph
	state graph to produce a full LLM-generated report.
# Saleforce Data Automate

Saleforce Data Automate is a small full-stack project that provides
Salesforce-style sales analysis and report automation using a Flask
backend (analysis + LLM-driven reporting) and a React + TypeScript
frontend (dashboard + chat UI).

This README explains the project structure, environment variables,
local setup (PowerShell), API endpoints, and developer notes to get
you started.

## Quick summary

- Backend: Flask app in `main/` that loads sales data (Supabase), runs
	analysis (RFM, monthly trends, growth, cohort, KPIs) and exposes
	endpoints used by the UI. The backend also composes a LangGraph
	state graph to produce a full LLM-generated report.

- Frontend: React + TypeScript app in `frontend/` (Vite) with a
	dashboard and chat page that call the backend endpoints.

## Repo layout (important files)

- `main/` - Python backend code
	- `app.py` - Flask app and agent initialization (endpoints: `/chat`, `/dashboard`, `/report`)
	- `config.py` - environment and client setup (Supabase, LLM client)
	- `data_processing.py` - load & clean data
	- `analysis.py` - metric & analytics functions (RFM, trends, growth)
	- `tools.py` - plotting tools and LangChain tool wrappers
	- `handlers.py` - orchestrates full analysis and graph generation

- `frontend/` - React + TypeScript UI (Vite)
	- `src/` - source files (`App.tsx`, pages, components)
	- `package.json` - frontend scripts & dependencies

- `data/` - sample data files (`data.csv`)
- `test/` - notebooks and exploratory analysis artifacts

## Requirements / Dependencies

The Python `requirement.txt` currently lists a minimal entry (`streamlit`).
You'll need additional packages used by the backend. Suggested Python
packages to install:

```text
flask
flask-cors
pandas
scikit-learn
supabase
python-dotenv
langchain
langgraph
langchain-openai
matplotlib
seaborn
networkx
```

Frontend dependencies live in `frontend/package.json` and use Vite + React.

## Environment variables

Create a `.env` file (or set env vars in your shell) with at least:

```env
SUPABASE_URL=
SUPABASE_KEY=
api_key=
```

- `SUPABASE_URL` and `SUPABASE_KEY`: used by `main/data_processing.py` to
	load data from your Supabase table (default table name: `invoices_table`).

- `api_key`: API key for the LLM/OpenRouter or provider used in `main/config.py`.

If you don't want to use Supabase during local development, consider
adding a small fallback in `main/data_processing.py` to load `data/data.csv`.

## Run locally (PowerShell)

1. Backend (Python)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\requirement.txt
# install additional packages used by the app:
pip install flask flask-cors pandas scikit-learn supabase python-dotenv langchain langgraph langchain-openai matplotlib seaborn networkx

# set env vars (example)
# $env:SUPABASE_URL = "https://<project>.supabase.co"
# $env:SUPABASE_KEY = "<supabase-key>"
# $env:api_key = "<your-llm-key>"

python .\main\app.py
```

2. Frontend (separate terminal)

```powershell
cd .\frontend
npm install
npm run dev
```

Notes:

- The frontend `package.json` contains a `proxy` pointing to `http://localhost:5000`,
	so relative API calls will be forwarded to the Flask backend in dev.

## API endpoints

1. POST /chat

- Request JSON: `{ "user_input": "<text>" }`
- Response JSON: `{ "response": "<text>" }`


2. GET /dashboard

- Returns KPI JSON object with `Total Revenue`, `Total Invoices`, `Total Customers`,
	`Country Sales`, and `Monthly Trend`.


3. GET /report

- Triggers the LangGraph state graph that runs the full analysis pipeline and
	returns a generated report (LLM output).



## Development notes & suggestions

- Add a `.env.example` to the repo documenting required env vars.
- Add a local fallback data loader (load `data/data.csv`) when Supabase is
	not configured for easier local development.
- Persist generated graphs to a `./tmp/graphs` folder instead of ephemeral
	temporary files and implement a cleanup policy.
- Add a small unit test suite for `main/analysis.py` (pytest) covering
	`Metrics`, `monthly_growth`, and `customer_retention`.
- Improve error handling in `main/app.py` to return JSON errors instead of
	HTML tracebacks.

## Troubleshooting

- If you see `RuntimeError: Supabase client not configured.`, set
	`SUPABASE_URL` and `SUPABASE_KEY` or modify `main/data_processing.py` to
	load CSV for local testing.
- If frontend requests fail, confirm the backend is running on port 5000.

