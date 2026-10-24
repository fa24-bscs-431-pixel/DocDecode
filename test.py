from google import genai
from google.genai import types

client = genai.Client(api_key="AQ.Ab8RN6LO6tBr3rsDdRYcUzaBceyXNRwU14UM4BVZ57pS9bjOeA")

with open("letter.jpg", "rb") as f:
    img = f.read()

resp = client.models.generate_content(
    model="gemma-4-26b-a4b-it",
    contents=[
        types.Part.from_bytes(data=img, mime_type="image/jpeg"),
        "Explain this letter in simple words: what is it, what must I do, and by when?",
    ],
)
print(resp.text)