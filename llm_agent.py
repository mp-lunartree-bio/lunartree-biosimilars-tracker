from pinecone import Pinecone

# embedding_function  = SentenceTransformerEmbeddings(model_name="neuml/pubmedbert-base-embeddings")
# embedding_function  = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def to_ascii(value):
    if isinstance(value, str):
        return value.encode('ascii', errors='ignore').decode('ascii')
    return value

def query_embeddings(vector, n_results=5, embedding_name='trials', pc_api_key=None):
    if not pc_api_key:
        print("Missing Pinecone API key")
        raise

    pc = Pinecone(pc_api_key)
    index = pc.Index(embedding_name)
    try:
        result = index.query(
            vector=vector.tolist(),
            top_k=n_results,
            include_values=True,
            include_metadata=True,
            filter={}
        )
        matches = result['matches']
        response = [{
            'id':v['id'],
            'score': v['score'],
            'metadata': v['metadata']
        } for v in matches]
    except Exception as e:
        print(f"Error querying collection: {e}")
        raise

    return response

def get_prompt_template(message, context_trials):
    template = f"""
    Answer the following question based on the provided context:
    
    **Relevant Trials:**
    {context_trials}

    The user is interacting with a dashboard containing data on biosimilar trials and sponsors. Provide a coherent and concise answer.

    **Question:**
    {message}
 
    If the relevant trials do not address the question, respond with general knowledge, clarify that the response is not from the dashboard, and note that the chatbot is designed to answer questions about biosimilars trials.
    Even if the relevant trials address the question, add information from your general knowledge to answer the question more richly. 
    """
    return template

def get_memory_prompt_template(messages):
    if (len(messages)) == 0:
        return ""

    template = f"""
    Create a concise 1-2 sentence query that captures the essence of the given question. 
    Use the old messages to only fill any missing information. But the core goal is answering the question.
    
    **Question:**
    {messages[-1]['content']}

    **Old messages:**
    {messages[-5:-1]}

    ONLY CREATE A MORE CONCISE AND REFINED QUESTION. Do not add anything made up to the question. 
    Do not add any unrelated context from the old messages that is not required to answer the question.
    """
    return template