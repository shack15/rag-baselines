# Ingestion Baselines
Set of implementations of different ingestion methods.

# Best Practices
* Store all relevant information you want in metadata, NOT in the actual database entry
    * Later on, you can retrieve the metadata by using the vector database as a semantic key-value matching system
    * If you want, you can store the full text content in the metadata of each chunk, such that you have a pointer chunk -> full text
* When ingesting structured data, always use YAML instead of JSON
    * JSON takes up many more tokens, and adds confusion with the bracket structure
    * YAML uses many less tokens, retains the same structure, and reduces unnecessary comfusion 