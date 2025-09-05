# Eat-Smart

A Streamlit app that uses a local open-source LLM via Ollama (optional) to suggest recipes. Now supports Docker deployment with Ollama integration.

## Features
- Login/profile management
- Pantry management
- AI-generated recipe recommendations (via Ollama)
- Card-style recipe visualization
- Dietary preferences, allergies, constraints
- Local fallback recipe generator

## Setup Instructions (uses Docker) 

### 1. Clone the repository
```sh
git clone <repo-url>
cd eat-smart
```

### 2. Build and start with Docker Compose
```sh
docker-compose up --build
```
This will start two containers:
- `ollama`: Runs the Ollama LLM server
- `app`: Runs the Streamlit app

### 3. Access the app
Open your browser and go to:
```
http://localhost:8501
```

### 4. Model Download
The app will automatically download the required Ollama model (e.g., `gemma3:1b`) if not already present. No manual steps required.

### 5. Stopping the app
```sh
docker-compose down
```

## Manual Model Management (Optional)
If you want to manually pull a model inside the Ollama container:
```sh
docker exec -it <ollama_container_name> ollama pull gemma3:1b
```

## Development (Local, without Docker)
1. Install Python 3.12 and dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Start Ollama locally (see [Ollama docs](https://ollama.com/)).
3. Run the app:
   ```sh
   streamlit run recipe_buddy_app.py
   ```

## Environment Variables
- `OLLAMA_URL`: (optional) Set the Ollama API base URL. Defaults to `http://ollama:11434`.

## Troubleshooting
- If Ollama is not responding, ensure the container is running and listening on port 11434.
- If the model is missing, the app will attempt to pull it automatically.
- For permission issues, ensure `.recipe_buddy_data` is writable by the container.

---
For more details, see the code and comments in `recipe_buddy_app.py`.
