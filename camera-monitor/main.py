from openai import OpenAI
import os
client = OpenAI(
  base_url="https://routellm.abacus.ai/v1",
  api_key=os.environ.get("ROUTE_LLM_API_KEY"),
)
stream = True # or False
chat_completion = client.chat.completions.create(
  model="gpt-5",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ],
  stream=stream
)
if stream:
  for event in chat_completion:
    if event.choices[0].finish_reason:
      print(event.choices[0].finish_reason)
    else:
      if event.choices[0].delta:
        print(event.choices[0].delta.content)
else:
  print(chat_completion.choices[0].message.content)