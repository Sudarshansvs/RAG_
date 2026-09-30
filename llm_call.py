import ollama
from retriever import retrieve


def generate_answer(question, context):

    prompt = f"""
You are HR Buddy.

Answer the user's question using only
the information provided in the context.

If the answer is not available in the context,
say that you do not have enough information.

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


question = "How many annual leaves do I get?"

results = retrieve(question)

context = "\n\n".join(
    result["document"]
    for result in results
)

answer = generate_answer(
    question,
    context
)

print("llm Output:", answer)