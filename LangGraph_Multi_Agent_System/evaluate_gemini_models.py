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
    base_url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:{method}?key={api_key}"
    
    if method == "generateContent":
        payload = {"contents": [{"parts": [{"text": "Hi"}]}]}
    elif method == "embedContent":
        payload = {"model": model_name, "content": {"parts": [{"text": "Hi"}]}}
    else:
        return False, "Unknown method"
        
    try:
        response = requests.post(base_url, json=payload)
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
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(f"{RED}Error: GEMINI_API_KEY not found in environment.{RESET}")
        return

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        print(f"{CYAN}Fetching available models from Gemini API...{RESET}")
        response = requests.get(url)
        response.raise_for_status()
        models = response.json().get("models", [])
    except Exception as e:
        print(f"{RED}Failed to fetch models list: {e}{RESET}")
        return

    results = []

    print(f"\n{BOLD}Initializing Model Evaluation... Please wait as we ping each API.{RESET}\n")
    for m in models:
        name = m['name']
        methods = m.get("supportedGenerationMethods", [])
        
        if "generateContent" in methods:
            works, reason = test_model(api_key, name, "generateContent")
            results.append({
                "Model": name,
                "Type": "Chat Completion",
                "Works": works,
                "Details": reason
            })
            time.sleep(0.5) 
            
        if "embedContent" in methods:
            works, reason = test_model(api_key, name, "embedContent")
            results.append({
                "Model": name,
                "Type": "Embedding",
                "Works": works,
                "Details": reason
            })
            time.sleep(0.5)

    if not results:
        print(f"{RED}No models found.{RESET}")
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
        items = sorted(items, key=lambda x: not x["Works"])
        
        icon = "[MSG]" if group_type == "Chat Completion" else "[EMB]"
        print(f"{BOLD}{YELLOW}▶ {icon} {group_type.upper()} MODELS{RESET}")
        print(separator)
        print(f"{BOLD}{header_str}{RESET}")
        print(separator)
        for r in items:
            print(format_row(r["Model"], r["Works"], r["Details"], name_width, details_width))
        print(separator + "\n")

    # Save to Markdown
    with open("working_gemini_models.md", "w", encoding="utf-8") as f:
        f.write("# Gemini Models Test Results\n\n")
        
        for group_type, items in grouped_results.items():
            f.write(f"## {group_type}\n\n")
            f.write("| Model Name | Status | Details |\n")
            f.write("|---|---|---|\n")
            for r in items:
                status_icon = "✅ Yes" if r["Works"] else "❌ No"
                det_str = r['Details'].replace('\n', ' ') if not r['Works'] else "-"
                f.write(f"| {r['Model']} | {status_icon} | {det_str} |\n")
            f.write("\n")
            
    print(f"{GREEN}{BOLD}Evaluation Complete! Markdown copy saved to 'working_gemini_models.md'{RESET}")

if __name__ == "__main__":
    main()
