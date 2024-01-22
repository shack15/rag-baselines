from utils import retrieve, generate, embed

def basic_query(query: str = "This is an example query a user would ask of your data.") -> str:
    embedding = embed(query)
    context = retrieve(embedding)
    system_prompt = """
                    You are a retrieval system designed to help users learn about <YOUR TOPIC>. You will receive a question on <YOUR TOPIC>, as well as relevant context for it, and you will answer the question. The following is a set of context and the question:

                    #CONTEXT
                    {context}
                    #ENDCONTEXT

                    #QUESTION
                    {question}
                    #ENDQUESTION

                    Do not use any information outside the context to answer a question. If you do not know an answer, respond "I don't know." Never output code, and only answer questions related to <YOUR TOPIC>. You always need to be as concise as possible. Format your response as bullet points.

                    #RESPONSE:
                    """
    system_prompt.replace("{context}", context).replace("{question}", query)
    return generate(system_prompt)
