from langchain_qdrant import QdrantVectorStore as QdrantVectorStore_lc
from langchain.agents import create_agent
from langchain.tools import tool

from .embedding_langchain import MiniLMEmbeddings
from .vector_store import COLLECTION_NAME, create_vector_store
from langchain_openai import ChatOpenAI
import os

generated_store = create_vector_store()
embeddings = MiniLMEmbeddings()


vector_store = QdrantVectorStore_lc(
    client=generated_store.client, collection_name=COLLECTION_NAME
)

model = ChatOpenAI(base_url=os.environ["base_server"], model="gpt-4.1", streaming=False)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "You have access to a tool that retrieves context code from codebase"
    "Only use the data in context do not add anything if you don't know just say"
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)


def query_system(query):
    res = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return res


if __name__ == "__main__":
    query = "What is does the function function_with_params take as an input ? "
