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
    st.markdown(
        """
        ## 🇨🇱 Chilenos por Europa
        Haz consultas sobre recomendaciones en Europa y obtén información relevante.


        Ejemplo: `Hostales en Roma` - `Qué visitar en Amsterdam` - `Restaurantes en París`

        """
    )

    # Pinecone connection
    pinecone_api_key = str(st.secrets["pinecone_api_key"].value)
    pinecone_env = str(st.secrets["pinecone_env"].value)
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    index = pinecone.Index(str(st.secrets["pinecone_index"].value))

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
    query = st.text_input("", max_chars=50)

    if query:
        # Muestra "Cargando" mientras se procesa la respuesta
        with st.spinner("Cargando..."):
            get_response(query, vector_store)

    st.markdown("""
        ---

        Hecho por [Pedro Ríos](https://github.com/pedroriosg)
    """)

if __name__ == "__main__":
    main()
