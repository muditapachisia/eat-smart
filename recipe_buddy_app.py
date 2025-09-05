# Recipe Buddy - MVP (Local, CPU-Friendly)
# Streamlit app that uses a local open-source LLM via Ollama (optional) to suggest recipes

import json
import os
from typing import List, Dict, Any, Optional
import requests
import streamlit as st

APP_TITLE = "Recipe Buddy (MVP)"
DATA_DIR = ".recipe_buddy_data"
USERS_FILE = os.path.join(DATA_DIR, "users.json")

# ---------- Utilities ----------
def ensure_data_dir():
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.isfile(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)

def load_users() -> Dict[str, Any]:
    ensure_data_dir()
    with open(USERS_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_users(users: Dict[str, Any]):
    ensure_data_dir()
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def get_user(users: Dict[str, Any], username: str) -> Dict[str, Any]:
    if username not in users:
        users[username] = {
            "pantry": [],
            "profile": {"diet": [], "allergies": []},
            "history": []
        }
    return users[username]

# ---------- LLM Integration (Ollama) ----------
def ensure_ollama_model(model: str, ollama_url_base: str = None, timeout: int = 180) -> bool:
    """
    Checks if the model is available, and pulls it if not.
    Returns True if model is available or successfully pulled, False otherwise.
    """
    import time
    if ollama_url_base is None:
        ollama_url_base = os.getenv("OLLAMA_URL", "http://ollama:11434")
    # Check model availability
    try:
        resp = requests.get(f"{ollama_url_base}/api/tags", timeout=timeout)
        if resp.status_code == 200:
            tags = resp.json().get("models", [])
            if any(m.get("name", "") == model for m in tags):
                return True
    except Exception as e:
        print("Error checking Ollama models:", e)
        return False
    # Pull model if not available
    try:
        print(f"Pulling Ollama model: {model}")
        pull_resp = requests.post(f"{ollama_url_base}/api/pull", json={"name": model}, timeout=timeout)
        if pull_resp.status_code == 200:
            # Wait for model to finish pulling
            for _ in range(30):
                resp = requests.get(f"{ollama_url_base}/api/tags", timeout=timeout)
                if resp.status_code == 200:
                    tags = resp.json().get("models", [])
                    if any(m.get("name", "") == model for m in tags):
                        return True
                time.sleep(10)
        print(f"Failed to pull model {model}: {pull_resp.text}")
    except Exception as e:
        print("Error pulling Ollama model:", e)
    return False
def ollama_generate(prompt: str, model: str = "gemma3:1b", temperature: float = 0.6, timeout: int = 180) -> Optional[str]:
    ollama_url_base = os.getenv("OLLAMA_URL", "http://ollama:11434")
    # Ensure model is available
    ensure_ollama_model(model, ollama_url_base, timeout)
    url = f"{ollama_url_base}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }
    try:
        print("Calling Ollama...")
        resp = requests.post(url, json=payload, timeout=timeout)
        print("Ollama response status:", resp.status_code)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        return None
    except Exception as e:
        print(e)
        return None

SYSTEM_INSTRUCTIONS = """You are Recipe Buddy, a helpful cooking assistant.
You must produce strictly VALID JSON when asked for recipe lists. No commentary.
Each recipe must include: title, summary, total_time_minutes, ingredients (list of strings),
steps (list of short imperative steps), and tags (list of strings).
Prefer using the provided pantry ingredients. Respect constraints and time limits.
"""

def build_recipe_prompt(pantry: List[str], meal_type: str, time_limit: int, mood: List[str], constraints: List[str], must_use: List[str]) -> str:
    pantry_text = ", ".join(sorted(set([p.strip() for p in pantry if p.strip()])))
    mood_text = ", ".join(mood) if mood else "none"
    cons_text = ", ".join(constraints) if constraints else "none"
    must_text = ", ".join(must_use) if must_use else "none"
    return f"""{SYSTEM_INSTRUCTIONS}
Given the following context, generate EXACTLY 5 distinct recipes as a JSON array.
Context:
- Meal type: {meal_type}
- Time limit (minutes): {time_limit}
- Mood keywords: {mood_text}
- Constraints: {cons_text}
- Must-use ingredients: {must_text}
- Pantry ingredients available: {pantry_text}

Rules:
- ONLY return valid JSON: an array of 5 recipe objects.
- Each recipe object must have keys:
  "title" (string),
  "summary" (string),
  "total_time_minutes" (integer <= {time_limit}),
  "ingredients" (list of strings, relying on pantry where possible),
  "steps" (list of 5-10 concise steps),
  "tags" (list of strings).
- Favor simple, quick recipes within the time limit.
- Avoid exotic ingredients not in the pantry unless absolutely necessary.
- Keep titles unique and succinct.
JSON:
"""

# ---------- Local fallback ----------
DEFAULT_STEPS = [
    "Prep all ingredients as needed.",
    "Heat a pan or pot and add oil if required.",
    "Cook main components until done.",
    "Season to taste and combine all elements.",
    "Plate and serve."
]

def naive_generate_recipes(pantry: List[str], meal_type: str, time_limit: int, mood: List[str], constraints: List[str], must_use: List[str]) -> List[Dict[str, Any]]:
    base_names = {
        "breakfast": ["Quick Skillet Hash", "Speedy Scramble Bowl", "Pantry Oat Parfait", "Toasty Sandwich Melt", "5-Min Omelet Wrap"],
        "lunch": ["15-Min Pantry Pasta", "Zippy Grain Bowl", "Crisp Veggie Wrap", "One-Pan Fried Rice", "Hearty Bean Salad"],
        "dinner": ["Weeknight Stir-Fry", "Simple Sheet-Pan Bake", "Creamy Pantry Pasta", "Speedy Chili", "Golden Veg Curry"],
        "snacks": ["Savory Trail Mix", "Nutty Energy Bites", "Crisp Chickpea Snack", "Cheesy Toast Bites", "Yogurt Fruit Cup"]
    }
    key = meal_type.lower()
    titles = base_names.get(key, base_names["dinner"])
    recipes = []
    for i, title in enumerate(titles):
        recipes.append({
            "title": f"{title} #{i+1}",
            "summary": f"A quick {meal_type} using pantry staples.",
            "total_time_minutes": min(time_limit, 20),
            "ingredients": pantry[:5] if pantry else ["salt", "pepper", "oil"],
            "steps": DEFAULT_STEPS,
            "tags": constraints[:3]
        })
    return recipes

# ---------- UI ----------
st.set_page_config(page_title=APP_TITLE, page_icon="🍳", layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Login / Profile")
    username = st.text_input("Username", placeholder="e.g., alex", key="username")
    login_btn = st.button("Log In / Create Account", use_container_width=True)

if "session_user" not in st.session_state:
    st.session_state.session_user = None

users = load_users()
user_obj = None

if login_btn:
    if username.strip():
        user_obj = get_user(users, username.strip())
        st.session_state.session_user = username.strip()
        save_users(users)
    else:
        st.warning("Please provide a username.")

if st.session_state.session_user:
    st.success(f"Logged in as: {st.session_state.session_user}")
    users = load_users()
    user_obj = get_user(users, st.session_state.session_user)

    with st.expander("Profile Settings", expanded=False):
        diets = st.multiselect("Dietary preferences (optional)",
                               ["vegetarian", "vegan", "gluten-free", "dairy-free", "halal", "kosher", "low-carb", "keto", "paleo"],
                               default=user_obj["profile"].get("diet", []))
        allergies = st.text_input("Allergies (comma-separated)", value=", ".join(user_obj["profile"].get("allergies", [])))
        if st.button("Save Profile"):
            user_obj["profile"]["diet"] = diets
            user_obj["profile"]["allergies"] = [a.strip() for a in allergies.split(",") if a.strip()]
            users[st.session_state.session_user] = user_obj
            save_users(users)
            st.success("Profile saved.")

    st.subheader("Your Pantry")
    pantry_text = st.text_area("Pantry list", value="\n".join(user_obj.get("pantry", [])), height=200)
    if st.button("Save Pantry"):
        user_obj["pantry"] = [line.strip() for line in pantry_text.splitlines() if line.strip()]
        users[st.session_state.session_user] = user_obj
        save_users(users)
        st.success("Pantry saved.")

    st.subheader("What do you want to cook?")
    meal_type = st.radio("Meal", ["breakfast", "lunch", "dinner", "snacks"], horizontal=True)
    time_limit = st.slider("How much time do you have? (minutes)", 5, 90, 25, 5)
    mood = st.multiselect("In the mood for (optional)", ["comforting", "spicy", "fresh", "creamy", "crispy", "hearty", "light", "tangy"])
    constraints = st.multiselect("Constraints (optional)", ["healthy", "high-protein", "vegetarian", "vegan", "gluten-free", "dairy-free", "low-carb", "low-calorie"])
    must_use_raw = st.text_input("Must-use ingredient(s) (optional, comma-separated)")
    must_use = [m.strip() for m in must_use_raw.split(",") if m.strip()]

    model_name = st.text_input("Ollama model (optional)", value="gemma3:1b")
    generate_btn = st.button("Generate 5 Recipes", type="primary")

    recipes = None
    if generate_btn:
        with st.spinner("Generating recipes..."):
            prompt = build_recipe_prompt(user_obj.get("pantry", []), meal_type, time_limit, mood, constraints, must_use)
            response_text = ollama_generate(prompt, model=model_name) if model_name else None
            
            if response_text:
                # Try to parse the JSON
                response_text = response_text[8:-4]
                try:
                    recipes = json.loads(response_text)
                    if not isinstance(recipes, list) or len(recipes) != 5:
                        raise ValueError("Expected a list of 5 recipes.")
                except json.JSONDecodeError:
                    # Handle non-JSON output
                    st.warning("Model returned non-JSON output. Displaying raw response.")
                    recipes = [{"recipe": response_text}]  # Wrap raw text in a list for display
                except Exception as e:
                    print("Error parsing JSON:", e)  # Debug: Log the error
                    st.warning("Unexpected error occurred. Falling back to local generator.")
                    recipes = naive_generate_recipes(user_obj.get("pantry", []), meal_type, time_limit, mood, constraints, must_use)
            else:
                # Ollama not available -> fallback
                print("Ollama not available or failed. Using naive generator.")
                recipes = naive_generate_recipes(user_obj.get("pantry", []), meal_type, time_limit, mood, constraints, must_use)

            st.session_state.generated_recipes = recipes

    if "generated_recipes" in st.session_state:
        recipes = st.session_state.generated_recipes
        st.subheader("Recommendations")
        # Card-style display for each recipe
        if recipes and isinstance(recipes, list):
            cols = st.columns(len(recipes))
            for i, recipe in enumerate(recipes):
                with cols[i]:
                    if isinstance(recipe, dict) and "recipe" in recipe:
                        st.text(recipe["recipe"])
                    else:
                        st.markdown(f"**{recipe.get('title', f'Recipe {i+1}')}**")
                        st.caption(recipe.get("summary", ""))
                        st.markdown(f"⏱ **{recipe.get('total_time_minutes', '?')} min**")
                        if recipe.get("tags"):
                            st.caption(" · ".join(recipe["tags"]))
                        st.markdown("**Ingredients**")
                        st.markdown("\n".join([f"- {ing}" for ing in recipe.get("ingredients", [])]))
                        st.markdown("**Steps**")
                        st.markdown("\n".join([f"{idx+1}. {step}" for idx, step in enumerate(recipe.get("steps", []))]))
    
    
    
    
    


else:
    st.info("Use the sidebar to log in or create an account.")
