"""
Prototipo de RAG (item 8 do plano de acao pos-banca) -- interface Streamlit.

Fluxo: pergunta em linguagem natural -> reconhecimento de entidades
mencionadas -> recuperacao de contexto no grafo (relacoes diretas +
vizinhos semanticos via embeddings RotatE) -> geracao da resposta por um
LLM local via Ollama, usando o contexto recuperado.

Pre-requisitos:
1. Neo4j rodando com o grafo ja populado (ligar-tcc / docker compose up -d).
2. 'python treinar_embeddings_rag.py' ja executado ao menos uma vez.
3. Ollama instalado e rodando localmente, com um modelo baixado
   (ex.: 'ollama pull llama3.2').

Rodar com: streamlit run app_rag.py
"""
import streamlit as st

from rag_retriever import GraphRAGRetriever
from rag_llm import gerar_resposta

st.set_page_config(page_title="RAG sobre o Grafo Clinico SemClinBr", layout="wide")


@st.cache_resource
def carregar_retriever():
    return GraphRAGRetriever()


st.title("Prototipo de RAG -- Grafo de Conhecimento Clinico (SemClinBr)")
st.caption(
    "Recupera contexto do grafo unificado (relacoes diretas + vizinhos "
    "semanticos via embeddings RotatE) e gera a resposta com um LLM local via Ollama."
)

with st.sidebar:
    st.header("Configuracao")
    modelo_ollama = st.text_input("Modelo Ollama", value="llama3.2")
    host_ollama = st.text_input("Host do Ollama", value="http://localhost:11434")
    max_entidades = st.slider("Max. de entidades reconhecidas na pergunta", 1, 5, 3)
    vizinhos_diretos_n = st.slider("Vizinhos diretos por entidade", 1, 10, 5)
    vizinhos_semanticos_n = st.slider("Vizinhos semanticos (embeddings) por entidade", 1, 10, 5)

try:
    retriever = carregar_retriever()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

pergunta = st.text_input(
    "Pergunta clinica",
    placeholder="Ex.: Quais sintomas estao relacionados a osteomielite?",
)

if st.button("Perguntar", type="primary") and pergunta.strip():
    with st.spinner("Recuperando contexto do grafo..."):
        contexto, detalhes = retriever.montar_contexto(
            pergunta,
            max_entidades=max_entidades,
            vizinhos_diretos_por_entidade=vizinhos_diretos_n,
            vizinhos_semanticos_por_entidade=vizinhos_semanticos_n,
        )

    if not detalhes["entidades_reconhecidas"]:
        st.warning(
            "Nenhum conceito do grafo foi reconhecido na pergunta (a busca e por "
            "correspondencia de texto com os termos do corpus SemClinBr). Tente "
            "usar o termo clinico como aparece no corpus (ex.: em maiusculas, como no prontuario)."
        )

    with st.expander("Entidades reconhecidas na pergunta", expanded=True):
        for ent in detalhes["entidades_reconhecidas"]:
            st.markdown(
                f"**{ent.get('text')}** _(categoria: {ent.get('tag_type')}, "
                f"ocorrencias no corpus: {ent.get('ocorrencias')})_"
            )

    with st.expander("Contexto recuperado do grafo"):
        st.text(contexto if contexto else "(vazio)")

    with st.spinner(f"Gerando resposta com {modelo_ollama} (Ollama)..."):
        try:
            resposta = gerar_resposta(pergunta, contexto, modelo=modelo_ollama, host=host_ollama)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

    st.subheader("Resposta")
    st.write(resposta)
