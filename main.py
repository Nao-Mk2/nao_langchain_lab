import os
from google import genai

def main():
  GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
  client = genai.Client(api_key=GEMINI_API_KEY)

  response = client.models.generate_content(
      model="gemini-2.0-flash",
      contents="Explain how AI works in a few words",
  )

  print(response.text)


if __name__ == "__main__":
  main()