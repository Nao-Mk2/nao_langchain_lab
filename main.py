import llm.google

def main():
  model = llm.google.Gemini()
  ai_msg = model.invoke("Explain how AI works in a few words")
  print(ai_msg.content)


if __name__ == "__main__":
  main()