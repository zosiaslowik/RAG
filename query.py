from langchain_openai import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def get_result(query, articles_database, k=3):
    prompt_template = """
    Answer the question:
    {question}
    ---
    Based on the context: {context}
    """
    # Search the database
    results = articles_database.similarity_search_with_relevance_scores(query, k)
    if not results or results[0][1] < 0.7:
        return "Unable to find any information in provided database."
    context = "\n\n---\n\n".join(article.page_content for article, _ in results)
    prompt = prompt_template.format(context=context, question=query)
    # Create a chat model instance and get the response
    chat_model = ChatOpenAI()
    response = chat_model.invoke(prompt)
    return response.content, sources


while True:
    # get the query
    user_query = input("Enter the query text: ")
    if user_query == "stop":
        break
    # load the database
    articles_database = Chroma(persist_directory='chroma', embedding_function=OpenAIEmbeddings())
    #get a response for provided query
    response, sources = get_result(user_query, articles_database)
    # show response
    print(f"Response: {response}, \n Sources: {sources}")
