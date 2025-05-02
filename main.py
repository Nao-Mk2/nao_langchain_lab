import llm.google

def main():
  model = llm.google.Gemini()
  user_input = input("Enter your message for the AI: ")
  ai_msg = model.invoke(user_input)
  print(ai_msg.content)


if __name__ == "__main__":
  main()