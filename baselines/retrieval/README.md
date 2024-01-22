# Retrieval Baselines
Set of base implementations of different retrieval methods.

# Best Practices
* An LLM is able to understand and process *markdown*, *html*, or *xml* the best as a query due to its structure with minimal tagging
* Always follow this order when crafting your prompts:
    * Text Type (user, assistant, system)
    * Task Context
    * Background Data and Documents
    * Task Description / Request
    * Step by Step One or Few Shot Examples
    * Output Formatting Instructions
    * Text Type (user, assistant, system)
    
# basic_query
Most basic implementation of RAG. Simply pulls from a vector database using the direct user query, loads it into context, and answers via LLM.