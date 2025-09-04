# Recipe Buddy (MVP)

A local-first, privacy-friendly AI recipe recommender built with Streamlit. Suggests recipes using your pantry and constraints, powered by a local LLM (Ollama) or a simple fallback generator.

## Features
- Login/profile management
- Pantry management
- AI-generated recipe recommendations (via Ollama)
- Card-style recipe visualization
- Dietary preferences, allergies, constraints
- Local fallback recipe generator

## Setup Instructions

### 1. Clone the repository
```
git clone <your-repo-url>
cd "6. Smart Recipe Generator"
```

### 2. Create and activate a Python virtual environment (recommended)
```
python -m venv env_recipe
# On Windows PowerShell:
.\env_recipe\Scripts\Activate.ps1
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. (Optional) Install Ollama and pull a small model
- Download Ollama: https://ollama.com/download
- Start Ollama and pull a model (e.g. gemma3:1b):
```
ollama pull gemma3:1b
```

### 5. Run the app
```
streamlit run recipe_buddy_app.py
```

## Requirements
See `requirements.txt` for Python dependencies.

## Notes
- Ollama is optional. If not available, the app uses a simple local recipe generator.
- All data is stored locally in `.recipe_buddy_data/users.json`.
- For best results, use a modern browser and Python 3.9+.

## License
MIT
