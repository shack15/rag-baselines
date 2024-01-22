import os
import sys
module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(module_dir)
from utils import retrieve, generate, embed
import re

def multi_query(query: str = "This is an example query a user would ask of your data.", num_queries: int = 3, keep_user_query: bool = True) -> str:
    """
    Answers queries with advanced multi-query retrieval. Augments user query with a few other LLM-generated synthetic queries, and determines relevant
    retrievals as the set union of all retrievals across all synthetic queries.
    """
    multi_query_prompt = f"You are a query generator, designed to come up with new ways to say a given query. Given a query, you will respond with {num_queries} other semantically identical queries. Do not respond to this prompt, only respond with the comma-separated new queries. Respond ONLY with your answers comma-separated, with no numbered listing. Query: {query}"
    synthetic_queries = generate(multi_query_prompt)
    queries = [query.strip() for query in re.split(',|\n', synthetic_queries) if query.strip()]
    if keep_user_query:
        queries.append(query)
    
    context = []
    for query in queries:
        embedded_query = embed(query)
        context.extend(retrieve(embedded_query)['text'])
    context = list(set(context))

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
    
    system_prompt = system_prompt.replace("{context}", str(context)).replace("{question}", query)
    return generate(system_prompt)


if __name__ == "__main__":
    print(multi_query())
