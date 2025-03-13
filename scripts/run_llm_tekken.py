import ollama


response = ollama.chat(
    model="llava",
    messages=[{"role": "user", "content": "Why is grass green"}],
)


print(response["message"]["content"])