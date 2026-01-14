python
 📦 Import necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from smolagents import InferenceClientModel, CodeAgent 
from dotenv import load_dotenv

 🔑 Load HuggingFace token and log in
load_dotenv()

 🧠 Initialize LLM model with Inference API
model = InferenceClientModel("meta-llama/Llama-3.1-70B-Instruct")

 🤖 Define agent with allowed libraries
agent = CodeAgent(
    tools=[],
    model=model,
    additional_authorized_imports=[
        "numpy",
        "pandas",
        "matplotlib.pyplot",
        "seaborn"
    ]
)

 📁 Ensure output directory exists
os.makedirs("figures", exist_ok=True)

 📓 Additional notes (e.g., column descriptions)
additional_notes = """
 Variable Description:
- 'company': Company name
- 'concept': Financial metric (e.g., revenue, expenses, equity)
- Columns in '2024-03-31' format: represent quarterly data
- This file contains financial results for various companies across multiple quarters.
"""

def generate_apple_profit_plot():
    """Creates a plot of Apple's profit over the last 3 years from all_company_financials.csv and saves it to figures/apple_profit_last3years.png. Returns the image path."""
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
     Load CSV
    df = pd.read_csv("all_company_financials.csv")
     Filter for Apple, concept=Profit
    df_apple = df[(df['company'].str.lower() == 'apple') & (df['concept'].str.lower().str.contains('profit'))]
     Extract only columns with year data
    year_cols = [col for col in df_apple.columns if re.match(r"^20\d{2}", col)]
     Find last 3 years
    years = sorted(year_cols)[-3:]
    values = [float(df_apple[y].values[0]) if not df_apple[y].isnull().all() else 0 for y in years]
     Plot
    plt.figure(figsize=(6,4))
    plt.bar(years, values, color="0071c5")
    plt.title("Apple Profit (Last 3 Years)")
    plt.ylabel("Profit (Billion USD)")
    plt.xlabel("Year")
    plt.tight_layout()
     Filename: Only letters/numbers, all lowercase
    safe_name = re.sub(r'[^a-z0-9]', '', 'apple')
    fname = f"{safe_name}_profit_last3years.png"
    out_path = os.path.join("figures", fname)
    plt.savefig(out_path)
    plt.close()
    return out_path

if __name__ == "__main__":
     📣 User interaction - input prompt
    print("🔍 Please enter your analysis request (e.g., 'Compare the liabilities of Apple and Microsoft in 2024.'):\n")
    user_prompt = input("> ")

     🏃 Run agent with analysis request
    response = agent.run(
        user_prompt,
        additional_args={
            "source_file": "all_company_financials.csv",
            "additional_notes": additional_notes
        }
    )

     🖨 Display result
    print("\n📊 Analysis Result:\n")
    print(response)
