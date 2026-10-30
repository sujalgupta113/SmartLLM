# 🤖 SmartLLM — Semantic Cache and Cost Optimizer
🌐 **Live Demo:** [Open SmartLLM](https://smartllm-zkjfjjjvhg7zsvufxryzdm.streamlit.app/)

SmartLLM is a Python-based application that uses semantic caching to reuse answers to similar questions, reduce unnecessary LLM calls, and monitor estimated API costs.

## 🚀 Features

- **Semantic Caching:** Uses Sentence Transformers to identify semantically similar questions.
- **Persistent Storage:** Stores cached questions, answers, and embeddings using SQLite.
- **Safety Bypass:** Prevents selected sensitive requests from being cached.
- **Cost Tracking:** Tracks token usage and estimates API costs.
- **Performance Dashboard:** Displays cache hits, misses, avoided calls, and request statistics.
- **Provider Support:** Includes MOCK mode and optional OpenAI integration.

## 🛠️ Tech Stack

- Python
- Streamlit
- Sentence Transformers
- NumPy
- Pandas
- SQLite
- OpenAI API (optional)

## ⚙️ Run Locally

1. Clone the repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   cd SmartLLM
   ```

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Activate it on Windows:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Configure your `.env` file using the required environment variables.

6. Start the application:

   ```bash
   python -m streamlit run app.py
   ```

## 🔐 Provider Configuration

The project supports MOCK mode for free simulation and optional OpenAI integration.

MOCK mode does not make real LLM API calls. OpenAI mode requires a separately configured API key and applicable API billing.

## 📁 Project Structure

```text
SmartLLM/
├── app.py
├── cache.py
├── database.py
├── cost_tracker.py
├── evaluate_safety.py
├── evaluate_thresholds.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 📊 Project Objective

The objective of SmartLLM is to demonstrate how semantic similarity-based caching, persistent storage, safety checks, and usage analytics can help optimize LLM application workflows.

## 👨‍💻 Author

Add your name and GitHub profile here.