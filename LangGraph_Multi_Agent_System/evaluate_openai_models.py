import os
import requests
import time
from dotenv import load_dotenv

# Enable ANSI escape sequences on Windows cmd/powershell
os.system("")

# ANSI Color Codes
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
GRAY = "\033[90m"

def test_model(api_key, model_name, method):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    if method == "chat/completions":
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5
        }
    elif method == "embeddings":
        url = "https://api.openai.com/v1/embeddings"
        payload = {
            "model": model_name,
            "input": "Hi"
        }
    else:
        return False, "Unknown method"
        
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return True, "OK"
        else:
            try:
                err = response.json().get("error", {}).get("message", response.text)
                return False, f"HTTP {response.status_code}: {err}"
            except:
                return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

def format_row(model_name, works, details, name_width, details_width):
    status_icon = "✔ OK" if works else "✘ ERR"
    status_color = GREEN if works else RED
    
    # Trim details string
    det_str = details if not works else "-"
    det_str = det_str.replace('\n', ' ')
    if len(det_str) > details_width:
        det_str = det_str[:details_width-3] + "..."
        
    return f"  {model_name.ljust(name_width)} | {status_color}{status_icon.ljust(6)}{RESET} | {GRAY}{det_str.ljust(details_width)}{RESET}"

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(f"{RED}Error: OPENAI_API_KEY not found in environment.{RESET}")
        return

    # Fetch ALL available models dynamically
    url = "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        print(f"{CYAN}Fetching available models from OpenAI API...{RESET}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        models_data = response.json().get("data", [])
        # Extract just the IDs
        models = [m["id"] for m in models_data]
    except Exception as e:
        print(f"{RED}Failed to fetch models list: {e}{RESET}")
        return

    results = []

    print(f"\n{BOLD}Initializing Model Evaluation... Please wait as we ping each API.{RESET}")
    print(f"{GRAY}Note: OpenAI does not explicitly broadcast supported endpoints per model in its /v1/models response.{RESET}")
    print(f"{GRAY}We will test each model against Chat Completion and Embedding endpoints to discover functionality.{RESET}\n")
    
    # We test every model on both endpoints (if it fails one, it might work on the other, or fail both)
    for model_name in models:
        
        # Test Chat Completion
        chat_works, chat_reason = test_model(api_key, model_name, "chat/completions")
        if chat_works or "does not support" not in chat_reason.lower() and "model not found" not in chat_reason.lower() and "model is not an endpoint" not in chat_reason.lower() and "model is a" not in chat_reason.lower() and "is a model" not in chat_reason.lower():
            # If it works, or if it fails for Quota / other transient reasons instead of "doesn't support chat", record it
             if "is not a chat model" not in chat_reason.lower() and "completions" not in chat_reason.lower():
                 results.append({
                    "Model": model_name,
                    "Type": "Chat Completion",
                    "Works": chat_works,
                    "Details": chat_reason
                 })
        time.sleep(0.1) 
            
        # Test Embedding
        emb_works, emb_reason = test_model(api_key, model_name, "embeddings")
        if emb_works:
            results.append({
                "Model": model_name,
                "Type": "Embedding",
                "Works": emb_works,
                "Details": emb_reason
            })
        elif "is not an embedding model" not in emb_reason.lower() and "embeddings" not in emb_reason.lower() and "model is a" not in emb_reason.lower() and "is a model" not in emb_reason.lower():
             # Record if it failed for quota and NOT because it's the wrong model type entirely
             pass # embeddings usually explicitly error out nicely, let's just log working ones or quota errors
             
             if "quota" in emb_reason.lower():
                 results.append({
                    "Model": model_name,
                    "Type": "Embedding",
                    "Works": emb_works,
                    "Details": emb_reason
                 })

        time.sleep(0.1)

    if not results:
        print(f"{RED}No tested models responded positively or with standard quota errors.{RESET}")
        return

    # Group by Type
    grouped_results = {}
    for r in results:
        t = r["Type"]
        if t not in grouped_results:
            grouped_results[t] = []
        grouped_results[t].append(r)

    # Output formatting like Linux professional table
    name_width = 46
    details_width = 50
    header_str = f"  {'Model Name'.ljust(name_width)} | {'Status'.ljust(6)} | {'Details'.ljust(details_width)}"
    separator = "-" * (name_width + sum([6, details_width]) + 10)

    for group_type, items in grouped_results.items():
        # Sort so working models appear first
        items = sorted(items, key=lambda x: (not x["Works"], x["Model"]))
        
        icon = "[MSG]" if group_type == "Chat Completion" else "[EMB]"
        print(f"{BOLD}{YELLOW}▶ {icon} {group_type.upper()} MODELS{RESET}")
        print(separator)
        print(f"{BOLD}{header_str}{RESET}")
        print(separator)
        
        # Deduplicate models in group just in case
        seen = set()
        for r in items:
            if r["Model"] not in seen:
                print(format_row(r["Model"], r["Works"], r["Details"], name_width, details_width))
                seen.add(r["Model"])
        print(separator + "\n")

    # Save to Markdown
    with open("working_openai_models.md", "w", encoding="utf-8") as f:
        f.write("# OpenAI Models Test Results\n\n")
        
        for group_type, items in grouped_results.items():
            f.write(f"## {group_type}\n\n")
            f.write("| Model Name | Status | Details |\n")
            f.write("|---|---|---|\n")
            seen = set()
            for r in items:
                if r["Model"] not in seen:
                    status_icon = "✅ Yes" if r["Works"] else "❌ No"
                    det_str = r['Details'].replace('\n', ' ') if not r['Works'] else "-"
                    f.write(f"| {r['Model']} | {status_icon} | {det_str} |\n")
                    seen.add(r["Model"])
            f.write("\n")
            
    print(f"{GREEN}{BOLD}Evaluation Complete! Markdown copy saved to 'working_openai_models.md'{RESET}")

if __name__ == "__main__":
    main()
