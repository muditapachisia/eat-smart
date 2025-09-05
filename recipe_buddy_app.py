# Recipe Buddy - MVP (Local, CPU-Friendly)
# Streamlit app that uses a local open-source LLM via Ollama (optional) to suggest recipes

import json
import os
from typing import List, Dict, Any, Optional
import requests
import streamlit as st

st.set_page_config(page_title="EatSmart", page_icon="🍳", layout="centered")

custom_css = '''
<style>
body, .stApp, .stMarkdown, .stText, .stHeader, .stSubheader, .stTitle, .stCaption, .stDataFrame, .stAlert, .stTextInput, .stTextArea, .stSelectbox, .stMultiSelect, .stRadio, .stButton, .stCheckbox, .stSlider, .stNumberInput, .stDateInput, .stTimeInput, .stFileUploader, .stColorPicker, .stForm, .stFormSubmitButton, .stExpander, .stTabs, .stTab, .stMetric, .stJson, .stCode, .stException, .stError, .stWarning, .stSuccess, .stInfo, .stHelp, .stTooltip, .stProgress, .stSpinner, .stSidebar, .stSidebarContent, .stSidebarHeader, .stSidebarSubheader, .stSidebarTitle, .stSidebarCaption, .stSidebarDataFrame, .stSidebarAlert, .stSidebarTextInput, .stSidebarTextArea, .stSidebarSelectbox, .stSidebarMultiSelect, .stSidebarRadio, .stSidebarButton, .stSidebarCheckbox, .stSidebarSlider, .stSidebarNumberInput, .stSidebarDateInput, .stSidebarTimeInput, .stSidebarFileUploader, .stSidebarColorPicker, .stSidebarForm, .stSidebarFormSubmitButton, .stSidebarExpander, .stSidebarTabs, .stSidebarTab, .stSidebarMetric, .stSidebarJson, .stSidebarCode, .stSidebarException, .stSidebarError, .stSidebarWarning, .stSidebarSuccess, .stSidebarInfo, .stSidebarHelp, .stSidebarTooltip, .stSidebarProgress, .stSidebarSpinner {
        color: #D35400 !important;
    }
    /* Input fields text and border color */
    input, textarea, select, .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"], .stMultiSelect div[data-baseweb="select"], .stNumberInput input, .stDateInput input, .stTimeInput input {
        color: #3CB371 !important;
        border-color: #3CB371 !important;
    }
    /* Input placeholder color */
    ::placeholder {
        color: #3CB371 !important;
        opacity: 1;
    }
    /* Streamlit widget label color */
    label, .css-1cpxqw2, .stTextInput label, .stTextArea label, .stSelectbox label, .stMultiSelect label, .stNumberInput label, .stDateInput label, .stTimeInput label {
        color: #D35400 !important;
    }
    /* Remove Streamlit default blue focus ring */
    input:focus, textarea:focus, select:focus {
        outline: 2px solid #3CB371 !important;
        box-shadow: 0 0 0 2px #3CB37133 !important;
    }
</style>
'''

st.markdown(custom_css, unsafe_allow_html=True)

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

SYSTEM_INSTRUCTIONS = """You are AI assistant specialized in crafting meal recipes for the user that the user constraints. \
You must produce strictly VALID JSON when asked for recipe lists. No commentary. \
Each recipe should also be a valid JSON and must include: title, summary, total_time_minutes, ingredients (list of strings), \
steps (list of short imperative steps), and tags (list of strings). \
Assume, you only have the provided pantry ingredients which are available for your use. Respect constraints and time limits. \
Given the following context, generate EXACTLY 4 distinct recipes as a JSON array. Do not add new lines and tab spacing when creating the JSON response.
"""

def build_recipe_prompt(pantry: List[str], meal_type: str, time_limit: int, mood: List[str], constraints: List[str], must_use: List[str]) -> str:
    pantry_text = ", ".join(sorted(set([p.strip() for p in pantry if p.strip()])))
    mood_text = ", ".join(mood) if mood else "none"
    cons_text = ", ".join(constraints) if constraints else "none"
    must_text = ", ".join(must_use) if must_use else "none"
    return f"""{SYSTEM_INSTRUCTIONS}
Given the following context, generate EXACTLY 4 distinct recipes as a JSON array.
Context:
- Meal type: {meal_type}
- Time limit (minutes): {time_limit}
- Mood keywords: {mood_text}
- Constraints: {cons_text}
- Must-use ingredients: {must_text}
- Pantry ingredients available: {pantry_text}

Rules:
- ONLY return valid JSON: an array of 4 recipe objects.
- Each recipe object must have keys:
  "title" (string),
  "summary" (string),
  "total_time_minutes" (integer <= {time_limit}),
  "ingredients" (list of strings, relying on pantry where possible),
  "steps" (list of 5-10 concise steps),
  "tags" (list of strings).
- Favor simple, quick recipes that respect the time limit.
- Avoid exotic ingredients not in the pantry.
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

# ---------- Onboarding Flow ----------
if "onboarding_step" not in st.session_state:
    st.session_state.onboarding_step = 0
if "onboarding_user" not in st.session_state:
    st.session_state.onboarding_user = ""
if "onboarding_diet" not in st.session_state:
    st.session_state.onboarding_diet = []
if "onboarding_allergies" not in st.session_state:
    st.session_state.onboarding_allergies = []
if "onboarding_pantry" not in st.session_state:
    st.session_state.onboarding_pantry = []
if "session_user" not in st.session_state:
    st.session_state.session_user = None

logo_path = "eatsmart_logo.png"  # Place your logo in the same directory

def onboarding():
    step = st.session_state.onboarding_step
    if step == 0:
        # Landing page
        st.markdown("""
            <div style='display: flex; flex-direction: column; align-items: center; justify-content: center;'>
                <h1 style='color: #d35400; font-family: Georgia, Arial, serif; margin-bottom: 0.2rem; text-align: center;'>Welcome to EatSmart!</h1>
                <h3 style='text-align:center;'>Your AI-powered kitchen companion</h3>
                <h5 style='text-align:center; font-style: italic;'>Get personalized recipes based on your pantry and preferences.</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""<span style='color:#d35400; font-size:1.1rem;'>Enter your name or ID to get started:</span>""", unsafe_allow_html=True)
        name = st.text_input("", key="onboarding_name", label_visibility="collapsed")
        def next1_callback():
            if name.strip():
                st.session_state.onboarding_user = name.strip()
                st.session_state.onboarding_step = 1
            else:
                st.session_state.show_warning = True
        st.button("Next", key="onboarding_next1", on_click=next1_callback)
        if st.session_state.get("show_warning"):
            st.warning("Please enter your name or ID.")
            st.session_state.show_warning = False
    elif step == 1:
        # Dietary preferences
        st.header(f"Hi {st.session_state.onboarding_user}! Let's set up your profile.")
        diet_val = st.session_state.get("onboarding_diet", [])
        allergies_list = st.session_state.get("onboarding_allergies", [])
        allergies_val = ", ".join(allergies_list) if allergies_list else ""
        diet = st.multiselect(
            "Dietary preferences:",
            ["Vegetarian", "Non-Vegetarian", "Vegan"],
            default=diet_val,
            key="onboarding_diet_widget"
        )
        allergies = st.text_input(
            "Allergies (comma-separated):",
            value=allergies_val,
            key="onboarding_allergies_widget"
        )
        def next2_callback():
            st.session_state.onboarding_diet = diet
            st.session_state.onboarding_allergies = [a.strip() for a in allergies.split(",") if a.strip()]
            st.session_state.onboarding_step = 2
        def back1_callback():
            st.session_state.onboarding_step = 0
        col_back, col_next, _ = st.columns([0.1, 0.1, 0.8])
        with col_back:
            st.button("Back", key="onboarding_back1", on_click=back1_callback)
        with col_next:
            st.button("Next", key="onboarding_next2", on_click=next2_callback)
    elif step == 2:
        st.header("Enter your pantry items")
        pantry_val = st.session_state.get("onboarding_pantry", [])
        pantry_str = ", ".join(pantry_val) if isinstance(pantry_val, list) else (pantry_val or "")
        pantry = st.text_area("List your pantry items (comma-separated):", value=pantry_str, key="onboarding_pantry_widget")
        def finish_callback():
            st.session_state.onboarding_pantry = [item.strip() for item in pantry.split(",") if item.strip()]
            st.session_state.onboarding_complete = True
            st.session_state.onboarding_step = 99  # Mark as done so main app loads
        def back2_callback():
            st.session_state.onboarding_step = 1
        col_back, col_finish, _ = st.columns([0.1, 0.1, 0.8])
        with col_back:
            st.button("Back", key="onboarding_back2", on_click=back2_callback)
        with col_finish:
            st.button("Finish", key="onboarding_finish", on_click=finish_callback)
    else:
        # Onboarding complete, save user and proceed to main app
        users = load_users()
        user_obj = get_user(users, st.session_state.onboarding_user)
        user_obj["pantry"] = st.session_state.onboarding_pantry
        user_obj["profile"]["diet"] = st.session_state.onboarding_diet
        user_obj["profile"]["allergies"] = st.session_state.onboarding_allergies
        users[st.session_state.onboarding_user] = user_obj
        save_users(users)
        st.session_state.session_user = st.session_state.onboarding_user
        st.session_state.onboarding_step = 99  # Mark as done
        st.success("Profile saved! Proceeding to your recipe dashboard...")

if st.session_state.get("onboarding_step", 0) < 3:
    onboarding()
    st.stop()

# ---------- Main App (after onboarding) ----------
if st.session_state.get("onboarding_step", 0) >= 3:
    # Load user object for the current session
    users = load_users()
    user_obj = get_user(users, st.session_state.session_user)
    # Set default model name for Ollama
    model_name = "gemma3:1b"
    import base64

    with open(logo_path, "rb") as f:
        logo_bytes = f.read()
    logo_base64 = base64.b64encode(logo_bytes).decode("utf-8")

    st.markdown(f"""
        <div style='display: flex; align-items: center;'>
            <img src='data:image/png;base64,{logo_base64}' alt='EatSmart Logo' style='height:48px; margin-right:16px;'>
            <h2 style='color:#d35400; font-family:Georgia, Arial, serif; margin:0;'>What are you hungry for today?</h2>
        </div>
    """, unsafe_allow_html=True)

    with st.form("main_app_form"):
        meal_type = st.radio(
            "Select a meal:",
            ["Breakfast", "Lunch", "Dinner", "Snack"],
            key="main_meal_type",
            horizontal=True
        )
        time_limit = st.slider(
            "How much time do you have? (minutes)",
            min_value=5, max_value=120, value=30, step=1, key="main_time_limit"
        )
        mood_options = ["Comforting", "Spicy", "Creamy", "Light", "Tangy", "Savory", "Sweet", "Fresh", "Hearty", "Zesty"]
        mood = st.multiselect(
            "What are you in the mood for? (optional)",
            mood_options,
            key="main_mood"
        )
        constraint_options = ["High-protein", "Low-calorie", "Vegan", "Vegetarian", "Gluten-free", "Dairy-free", "Nut-free", "Low-carb"]
        constraints = st.multiselect(
            "Any constraints? (optional)",
            constraint_options,
            key="main_constraints"
        )
        include_ingredients = st.text_input(
            "Any ingredients you want included? (optional)",
            key="main_include_ingredients"
        )
        col1, col2 = st.columns([1,1])
        with col1:
            back_clicked = st.form_submit_button("Back to Edit Profile", use_container_width=True)
        with col2:
            submitted = st.form_submit_button("Get Recipes", use_container_width=True)
        if back_clicked:
            st.session_state.onboarding_step = 2
            st.stop()

    if submitted:
        # Process the form data and generate recipes
        meal_type = st.session_state.main_meal_type.lower()
        time_limit = st.session_state.main_time_limit
        mood = st.session_state.main_mood
        constraints = st.session_state.main_constraints
        must_use = [m.strip() for m in st.session_state.main_include_ingredients.split(",") if m.strip()]

        with st.spinner("Generating recipes..."):
            prompt = build_recipe_prompt(user_obj.get("pantry", []), meal_type, time_limit, mood, constraints, must_use)
            response_text = ollama_generate(prompt, model=model_name) if model_name else None
            
            if response_text:
                # Try to parse the JSON
                response_text = response_text[8:-4]
                print("Ollama response text:", response_text)  # Debug: Log the raw response
                try:
                    recipes = json.loads(response_text)
                    if not isinstance(recipes, list) or len(recipes) != 4:
                        raise ValueError("Expected a list of 4 recipes.")
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
        recipes = st.session_state.generated_recipes[:4] if len(st.session_state.generated_recipes) >= 4 else st.session_state.generated_recipes
        st.subheader("Recommendations")
        if recipes and isinstance(recipes, list):
            num_recipes = len(recipes)
            if "selected_recipe_idx" not in st.session_state:
                st.session_state.selected_recipe_idx = None
            selected = st.session_state.selected_recipe_idx
            if selected is not None and selected < num_recipes:
                # Enlarged, flipped main card (simulate flip with conditional rendering)
                st.markdown(f"""
                <style>
                .main-flip-card {{
                    background-color: transparent;
                    width: 520px;
                    height: 520px;
                    perspective: 1000px;
                    margin: 0 auto 2rem auto;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .main-flip-card-inner {{
                    width: 100%;
                    height: 100%;
                    border-radius: 16px;
                    box-shadow: 0 2px 16px #d3540033;
                    background: #fff7ed;
                    padding: 2rem;
                    display: flex;
                    flex-direction: column;
                    justify-content: flex-start;
                    align-items: flex-start;
                    overflow-y: auto;
                    max-height: 520px;
                }}
                .mini-flip-card {{
                    width: 180px;
                    height: 180px;
                    background: #fff7ed;
                    border-radius: 16px;
                    box-shadow: 0 2px 8px #d3540033;
                    display: flex;
                    flex-direction: column;
                    justify-content: flex-start;
                    align-items: flex-start;
                    cursor: pointer;
                    transition: box-shadow 0.2s;
                    overflow-y: auto;
                    padding: 1rem;
                }}
                .mini-flip-card:hover {{
                    box-shadow: 0 4px 16px #d3540066;
                }}
                .main-flip-card-inner::-webkit-scrollbar, .mini-flip-card::-webkit-scrollbar {{
                    width: 8px;
                }}
                .main-flip-card-inner::-webkit-scrollbar-thumb, .mini-flip-card::-webkit-scrollbar-thumb {{
                    background: #d3540033;
                    border-radius: 8px;
                }}
                </style>
                <div class="main-flip-card">
                  <div class="main-flip-card-inner" onclick="window.parent.postMessage({{isStreamlitMessage: true, type: 'streamlit:setComponentValue', value: null}}, '*');">
                    <h3 style='color:#d35400;'>{recipes[selected].get('title','')}</h3>
                    <b>Description:</b> {recipes[selected].get('summary','')}<br>
                    <b>Time Required:</b> {recipes[selected].get('total_time_minutes','?')} min<br>
                    <b>Tags:</b> {', '.join(recipes[selected].get('tags',[]))}<br>
                    <b>Ingredients:</b>
                    <ul style='text-align:left;'>
                      {''.join([f'<li>{ing}</li>' for ing in recipes[selected].get('ingredients',[])])}
                    </ul>
                    <b>Steps:</b>
                    <ol style='text-align:left;'>
                      {''.join([f'<li>{step}</li>' for step in recipes[selected].get('steps',[])])}
                    </ol>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                # Mini cards row (other cards) using columns for horizontal layout
                mini_cards = [i for i in range(num_recipes) if i != selected]
                if mini_cards:
                    mini_cols = st.columns(len(mini_cards), gap="large")
                    for col, idx in zip(mini_cols, mini_cards):
                        recipe = recipes[idx]
                        with col:
                            if st.button(" ", key=f"mini_card_{idx}"):
                                st.session_state.selected_recipe_idx = idx
                            st.markdown(f"""
                            <div class="mini-flip-card">
                                <h5 style='color:#d35400; margin-bottom:0.5rem;'>{recipe.get('title','')}</h5>
                                <p style='font-size:0.95rem; margin-bottom:0.3rem;'>{recipe.get('summary','')}</p>
                                <span style='font-size:0.9rem; color:#d35400;'>⏱ {recipe.get('total_time_minutes','?')} min</span><br>
                                <span style='font-size:0.9rem; color:#d35400;'>Tags: {', '.join(recipe.get('tags',[])[:4])}</span>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                # Show grid of cards (default view)
                grid_rows = (num_recipes + 1) // 2
                grid = [st.columns(2) for _ in range(grid_rows)]
                for idx in range(num_recipes):
                    row, col = divmod(idx, 2)
                    recipe = recipes[idx]
                    with grid[row][col]:
                        if st.button(" ", key=f"card_{idx}"):
                            st.session_state.selected_recipe_idx = idx
                        st.markdown(f"""
                        <style>
                        .flip-card-{idx} {{
                            background-color: transparent;
                            width: 260px;
                            height: 260px;
                            perspective: 1000px;
                            margin: 0 auto 1.2rem auto;
                            cursor: pointer;
                        }}
                        .flip-card-inner-{idx} {{
                            position: relative;
                            width: 100%;
                            height: 100%;
                            text-align: center;
                            transition: transform 0.6s;
                            transform-style: preserve-3d;
                        }}
                        .flip-card-front-{idx} {{
                            position: absolute;
                            width: 100%;
                            height: 100%;
                            backface-visibility: hidden;
                            border-radius: 16px;
                            box-shadow: 0 2px 8px #d3540033;
                            background: #fff7ed;
                            display: flex;
                            flex-direction: column;
                            justify-content: flex-start;
                            align-items: flex-start;
                            z-index: 2;
                            overflow-y: auto;
                            padding: 1rem;
                        }}
                        .flip-card-front-{idx}::-webkit-scrollbar {{
                            width: 8px;
                        }}
                        .flip-card-front-{idx}::-webkit-scrollbar-thumb {{
                            background: #d3540033;
                            border-radius: 8px;
                        }}
                        </style>
                        <div class="flip-card-{idx}">
                          <div class="flip-card-inner-{idx}">
                            <div class="flip-card-front-{idx}">
                              <h4 style='color:#d35400; margin-bottom:0.5rem;'>{recipe.get('title','')}</h4>
                              <p style='font-size:1.05rem; margin-bottom:0.3rem;'>{recipe.get('summary','')}</p>
                              <span style='font-size:0.95rem; color:#d35400;'>⏱ {recipe.get('total_time_minutes','?')} min</span><br>
                              <span style='font-size:0.95rem; color:#d35400;'>Tags: {', '.join(recipe.get('tags',[])[:4])}</span>
                            </div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
