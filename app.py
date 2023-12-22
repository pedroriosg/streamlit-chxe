import streamlit as st
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def get_response(query, vector_store):

    relevant_docs = vector_store.similarity_search(query, k=3)
    llm = ChatOpenAI(
        model_name='gpt-3.5-turbo',
        temperature=0.0,
        openai_api_key=str(st.secrets["openai_api_key"].value),
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(),
    )

    model_output = qa.run(query)
    st.write(model_output)

def main():
    st.title("Chilenos por Europa 🇨🇱")

    # Descripción e instrucciones
    st.markdown("""
        Ingresa tu pregunta y obtén información relevante.
    """)

    # Pinecone connection
    pinecone_api_key = str(st.secrets["pinecone_api_key"].value)
    pinecone_env = str(st.secrets["pinecone_env"].value)
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    index = pinecone.Index("chpe")

    # Embedding
    embed = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=str(st.secrets["openai_api_key"].value),
    )

    # Vector store
    vector_store = Pinecone(
        index,
        embed,
        'text'
    )

    # Input para ingresar la query
    query = st.text_input("Ingrese su pregunta:")

    if query:
        # Muestra "Cargando" mientras se procesa la respuesta
        with st.spinner("Cargando..."):
            get_response(query, vector_store)

if __name__ == "__main__":
    main()
