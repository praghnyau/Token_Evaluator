from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Explain recursion in simple terms.",
    config=types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        )
    )
)

print("\n===== OUTPUT =====")
print(response.text)

print("\n===== TOKEN USAGE =====")
usage = response.usage_metadata

print("Input tokens :", usage.prompt_token_count)
print("Output tokens:", usage.candidates_token_count)
print("Total tokens :", usage.total_token_count)
