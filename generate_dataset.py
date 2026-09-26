from google import genai
import json
import os

# ==============================
# CONFIGURATION
# ==============================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set. "
        "Run: export GEMINI_API_KEY='your_api_key'"
    )

MODEL = "gemini-3.6-flash"

PROMPTS_FILE = "prompts.txt"
OUTPUT_FILE = "dataset.json"

# ==============================
# INITIALIZE GEMINI
# ==============================

client = genai.Client(api_key=API_KEY)

# ==============================
# LOAD PROMPTS
# ==============================

with open(PROMPTS_FILE, "r", encoding="utf-8") as file:
    prompts = [
        line.strip()
        for line in file
        if line.strip()
    ]

print(f"Found {len(prompts)} prompts.")

# ==============================
# LOAD EXISTING DATA
# ==============================

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    print(f"Loaded {len(dataset)} previously generated records.")
else:
    dataset = []

# Avoid generating the same prompt again
completed_prompts = {
    item["prompt"]
    for item in dataset
    if "prompt" in item
}

# ==============================
# GENERATE DATASET
# ==============================

for index, prompt in enumerate(prompts, start=1):

    # Skip already generated prompts
    if prompt in completed_prompts:
        print(f"[{index}/{len(prompts)}] Skipping: {prompt}")
        continue

    print(f"\n[{index}/{len(prompts)}] Generating...")
    print(f"Prompt: {prompt}")

    try:
        # ==============================
        # GENERATE RESPONSE
        # ==============================

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        generated_text = response.text

        # ==============================
        # CREATE RECORD
        # ==============================

        record = {
            "prompt": prompt,
            "response": generated_text
        }

        dataset.append(record)
        completed_prompts.add(prompt)

        # ==============================
        # SAVE AFTER EVERY PROMPT
        # ==============================

        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                dataset,
                file,
                indent=4,
                ensure_ascii=False
            )

        print("✓ Generated successfully")

    except Exception as error:
        print("✗ Error occurred!")
        print(f"Error: {error}")

        # Continue with the next prompt
        continue

# ==============================
# FINAL SUMMARY
# ==============================

print("\n" + "=" * 50)
print("DATASET GENERATION COMPLETE")
print("=" * 50)

print(f"Total prompts : {len(prompts)}")
print(f"Generated     : {len(dataset)}")

print(f"\nDataset saved to: {OUTPUT_FILE}")
