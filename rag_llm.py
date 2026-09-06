"""
Cliente minimo para o Ollama local (item 8 -- prototipo de RAG). Usa a API
HTTP nativa do Ollama (/api/generate) em vez do pacote `ollama` para nao
adicionar mais uma dependencia ao ambiente -- so precisa de `requests`.
"""
import requests

PROMPT_TEMPLATE = """Voce e um assistente de apoio a pesquisa clinica. Responda \
a pergunta do usuario usando SOMENTE as informacoes do contexto abaixo, \
extraido de um Grafo de Conhecimento Clinico construido a partir do corpus \
SemClinBr. Se o contexto nao for suficiente para responder, diga isso \
explicitamente em vez de inventar informacao.

Contexto (relacoes e conceitos do grafo):
{contexto}

Pergunta: {pergunta}

Resposta:"""


def gerar_resposta(pergunta, contexto, modelo="llama3.2", host="http://localhost:11434", timeout=120):
    """
    Chama o endpoint /api/generate do Ollama local. Lanca RuntimeError com
    mensagem amigavel se o Ollama nao estiver rodando ou o modelo nao
    tiver sido baixado (ollama pull <modelo>).
    """
    prompt = PROMPT_TEMPLATE.format(
        contexto=contexto if contexto else "(nenhum conceito do grafo foi reconhecido na pergunta)",
        pergunta=pergunta,
    )
    try:
        resposta = requests.post(
            f"{host}/api/generate",
            json={"model": modelo, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resposta.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Nao foi possivel conectar ao Ollama em {host}. Ele esta rodando? "
            "(instale em https://ollama.com/download; no Windows ele roda como servico "
            "automaticamente depois de instalado)"
        ) from e
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"Ollama retornou erro ao usar o modelo '{modelo}'. Ele foi baixado? "
            f"(comando: 'ollama pull {modelo}')"
        ) from e

    corpo = resposta.json()
    return corpo.get("response", "").strip()
