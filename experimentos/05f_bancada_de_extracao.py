"""
Experimento 5f: bancada de testes da extração das relações.

Depois do 5d e do teste do 5e, a dúvida era onde está o problema da extração: no jeito de pedir, no modelo de
linguagem ou na falta de uma camada a mais. Aqui cada forma muda uma coisa de cada vez e todas são medidas do
mesmo jeito do 5e, nas mesmas 50 notas de teste da parte 1 e com as mesmas listas de conceitos (marcadas pelo
modelo de conceitos que não viu a parte 1):
  1a  qwen2.5:7b (o modelo do 5d) com os 5 tipos escolhidos depois do 5d (sem dose, valor e tempo)
  1b  o mesmo, com duas notas anotadas do SemClinBr como exemplo no pedido
  2   qwen3.5:9b, com o pedido do 1b
  3a  gemma4:e4b, com o pedido do 1b
  3b  medgemma:4b (treinado para medicina), com o pedido do 1b
  4   qwen3.5:27b, com o pedido do 1b (não cabe na placa de vídeo; roda quase todo no processador)
  5-X a forma X, mais uma segunda leitura: o mesmo modelo relê cada relação e diz se a nota a afirma
  6   qwen3.5:35b-a3b, com o pedido do 1b (do tipo MoE: tem 35 bilhões de parâmetros, mas usa uns 3 bilhões
      para cada palavra; testado depois dos outros, para ver se tem a qualidade do 27b sendo mais rápido)

Como só ficam os 5 tipos, os conceitos de valor e de tempo saem da lista que vai no pedido. Os exemplos são as
notas 9596 e 9792 (da parte 2, curtas e com os 5 tipos), com as relações anotadas entre conceitos, cada uma
com o tipo dado pelas categorias dos dois conceitos (como no 5e). As notas longas vão em pedaços, como no 5d.
Os modelos que pensam antes de responder rodam sem pensar, como os outros.

Saídas (não versionadas):
  results/experimento5f/<forma>.jsonl             relações de cada nota, uma linha por nota
  results/experimento5f_resultados.json           medidas
  results/experimento5f_trata_para_conferir.csv   amostra das relações "trata" de cada forma, para conferir à
  results/experimento5f_trata_para_conferir.json  mão (as anotações quase não têm "trata")

Uso (no Windows, com o Ollama rodando):
    python experimentos/05f_bancada_de_extracao.py rodar 1b       # continua de onde parou
    python experimentos/05f_bancada_de_extracao.py rodar 5-1b     # a segunda leitura da forma 1b
    python experimentos/05f_bancada_de_extracao.py medir
    python experimentos/05f_bancada_de_extracao.py amostra
"""
import argparse
import collections
import csv
import functools
import importlib.util
import json
import os
import random
import re
import sys
import time
import urllib.error

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA = os.path.join(RAIZ, "results", "experimento5f")
SAIDA_JSON = os.path.join(RAIZ, "results", "experimento5f_resultados.json")
PLANILHA_TRATA = os.path.join(RAIZ, "results", "experimento5f_trata_para_conferir.csv")
AMOSTRA_TRATA = os.path.join(RAIZ, "results", "experimento5f_trata_para_conferir.json")
TRATA_POR_FORMA = 30
EXEMPLOS = ("9596.xml", "9792.xml")
TIPOS = ("trata", "tem_resultado", "localizado_em", "nega", "associado_a")
FORMAS = {
    "1a": ("qwen2.5:7b", False),
    "1b": ("qwen2.5:7b", True),
    "2": ("qwen3.5:9b", True),
    "3a": ("gemma4:e4b", True),
    "3b": ("medgemma:4b", True),
    "4": ("qwen3.5:27b", True),
    "6": ("qwen3.5:35b-a3b", True),
}
RELACOES_POR_CONFERENCIA = 40
# Pares de formas comparados nas mesmas notas (5d: o qwen do 5d; 5e: o modelo treinado do 5e), e as medidas:
# (parte, todo) das contagens
COMPARACOES = (("1b", "5d"), ("1a", "1b"), ("2", "1b"), ("3a", "2"), ("3b", "3a"), ("4", "2"), ("4", "3a"),
               ("5-2", "2"), ("5-3a", "3a"), ("5-4", "4"), ("2", "5e"), ("4", "5e"), ("6", "4"), ("6", "2"),
               ("6", "5e"))
MEDIDAS_COMPARADAS = {"anotadas achadas": ("achadas", "anotadas"), "batem c/ anotações": ("batem", "extraidas"),
                      "negações": ("negacoes_achadas", "negacoes")}
REAMOSTRAS = 2000


def carregar(nome, arquivo):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(RAIZ, "experimentos", arquivo))
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# Notas de teste, medidas e formas de referência: as do teste do experimento 5e (e, por ele, do 5d)
ligacoes = carregar("ligacoes", "05e_modelo_de_ligacoes.py")
grafo, ner, piloto = ligacoes.grafo, ligacoes.ner, ligacoes.piloto
conferir_5d = carregar("conferir_5d", "05d_conferir_relacoes.py")

PROMPT = """Você é um especialista em informática clínica. Leia a nota clínica abaixo, escrita em \
português por profissionais de saúde, e extraia as relações entre os conceitos clínicos que ela contém.

Regras:
- Use como origem e destino apenas os conceitos da lista da nota, copiados como estão na lista (a \
categoria entre parênteses é só uma ajuda; não a copie).
- Não invente relações que a nota não afirma.
- Não repita relações. Liste no máximo {maximo} relações, as mais importantes.
- Use um destes tipos de relação:
{tipos}
{exemplos}
Conceitos da nota:
{conceitos}

Nota clínica:
\"\"\"{texto}\"\"\"

Responda em JSON, no formato {{"relacoes": [{{"origem": "...", "relacao": "...", "destino": "..."}}]}}."""

PROMPT_EXEMPLOS = """
Exemplos de notas com as relações que anotadores humanos marcaram nelas:
{exemplos}"""

PROMPT_EXEMPLO = """
Conceitos da nota:
{conceitos}

Nota clínica:
\"\"\"{texto}\"\"\"

Resposta:
{resposta}
"""

PROMPT_CONFERIR = """Você é um especialista em informática clínica. Abaixo estão uma nota clínica, escrita em \
português por profissionais de saúde, e relações entre conceitos que foram extraídas dela. Para cada relação, \
diga se a nota realmente afirma essa relação, com esse tipo e nessa direção (true), ou não (false).

Tipos de relação:
{tipos}

Nota clínica:
\"\"\"{texto}\"\"\"

Relações:
{relacoes}

Responda em JSON, no formato {{"respostas": [{{"numero": 1, "certa": true}}]}}, com uma resposta para cada \
relação."""

ESQUEMA = {
    "type": "object",
    "properties": {"relacoes": {"type": "array", "maxItems": piloto.MAX_RELACOES, "items": {
        "type": "object",
        "properties": {"origem": {"type": "string"}, "relacao": {"type": "string", "enum": list(TIPOS)},
                       "destino": {"type": "string"}},
        "required": ["origem", "relacao", "destino"],
    }}},
    "required": ["relacoes"],
}
ESQUEMA_CONFERIR = {
    "type": "object",
    "properties": {"respostas": {"type": "array", "items": {
        "type": "object", "properties": {"numero": {"type": "integer"}, "certa": {"type": "boolean"}},
        "required": ["numero", "certa"],
    }}},
    "required": ["respostas"],
}


def arquivo_da_forma(chave):
    return os.path.join(PASTA, f"{chave}.jsonl")


def nome_da_forma(chave):
    if chave.startswith("5-"):
        return f"{nome_da_forma(chave[2:])} + segunda leitura"
    modelo, exemplos = FORMAS[chave]
    return f"{modelo}, 5 tipos" + (" + exemplos" if exemplos else "")


def modelo_da_forma(chave):
    return FORMAS[chave[2:] if chave.startswith("5-") else chave][0]


def nota_marcada(item):
    """Uma nota marcada no 5d (um item de conceitos_parte<k>.json), com o texto, os conceitos marcados (ajustados
    às palavras, como no 5d) e a nota lida do XML."""
    caminho = os.path.join(piloto.PASTA_XML, item["arquivo"])
    texto, spans, lista = grafo.conceitos_da_nota(item)
    return {"arquivo": item["arquivo"], "caminho": caminho, "texto": texto, "spans": spans, "lista": lista,
            "marcados": {(a, b): c for a, b, c in spans}, "nota": piloto.ler_nota(caminho)}


def notas_de_teste():
    """As 50 notas de teste do 5e, marcadas, com as relações anotadas."""
    with open(ligacoes.SAIDA_JSON, encoding="utf-8") as f:
        arquivos = json.load(f)["protocolo"]["notas_teste"]
    with open(grafo.arquivo_conceitos(1), encoding="utf-8") as f:
        itens = {n["arquivo"]: n for n in json.load(f)["notas"]}
    return [{**nota_marcada(itens[a]),
             "relacoes_entre_conceitos": ligacoes.relacoes_entre_conceitos(os.path.join(piloto.PASTA_XML, a))}
            for a in arquivos]


def linhas_da_lista(conceitos):
    return "\n".join(f"  - {c['texto']} ({c['categoria']})" for c in conceitos)


def texto_dos_tipos():
    return "\n".join(f"  - {nome}: {piloto.TIPOS[nome]}" for nome in TIPOS)


@functools.cache
def exemplos():
    """O trecho do pedido com as notas de exemplo e as relações anotadas entre conceitos de cada uma."""
    partes = []
    for arquivo in EXEMPLOS:
        caminho = os.path.join(piloto.PASTA_XML, arquivo)
        texto = ner.texto_cru(caminho)
        anotados, relacoes = ligacoes.anotacoes_e_relacoes(caminho, texto)
        conceitos = ner.lista_de_conceitos(texto, [(a, b, c) for (a, b), c in anotados.items()
                                                   if c not in ligacoes.LITERAIS])
        ligados = {par: (anotados[par[0]], anotados[par[1]], classe) for par, classe in relacoes.items()
                   if not {anotados[par[0]], anotados[par[1]]} & ligacoes.LITERAIS}
        resposta = {"relacoes": [{k: r[k] for k in ("origem", "relacao", "destino")}
                                 for r in ligacoes.como_relacoes({"texto": texto}, ligados)]}
        partes.append(PROMPT_EXEMPLO.format(conceitos=linhas_da_lista(conceitos), texto=" ".join(texto.split()),
                                            resposta=json.dumps(resposta, ensure_ascii=False)))
    return PROMPT_EXEMPLOS.format(exemplos="".join(partes))


def pedidos(n, com_exemplos):
    """Os pedidos para uma nota: a nota inteira se a lista (sem valores e tempos) cabe num pedido; se não, um
    pedido por pedaço de frases inteiras, como no 5d."""
    trecho_exemplos = exemplos() if com_exemplos else ""
    spans = [s for s in n["spans"] if s[2] not in ligacoes.LITERAIS]
    conceitos = [c for c in n["lista"] if c["categoria"] not in ligacoes.LITERAIS]

    def pedido(texto, lista):
        return PROMPT.format(maximo=piloto.MAX_RELACOES, tipos=texto_dos_tipos(), exemplos=trecho_exemplos,
                             conceitos=linhas_da_lista(lista), texto=texto)

    if len(conceitos) <= grafo.CONCEITOS_POR_PEDIDO:
        return [pedido(n["nota"]["texto"], conceitos)]
    mencoes = grafo.mencoes_da_lista(n["texto"], spans, conceitos)
    return [pedido(n["texto"][a:b].replace("\r\n", "\n").strip(),
                   [conceitos[i] for i in sorted({c for p, c in mencoes if a <= p < b})])
            for a, b in grafo.pedacos(n["texto"], mencoes)]


def pensa(modelo, host):
    """Se o modelo pensa antes de responder (aí ele roda sem pensar, como os outros)."""
    return "thinking" in piloto.chamar(f"{host}/api/show", {"model": modelo}).get("capabilities", [])


def perguntar(prompt, modelo, host, esquema, sem_pensar, maximo_tokens=2000):
    """Resposta crua do modelo, tokens gerados e se a resposta chegou ao limite de tamanho."""
    corpo = {"model": modelo, "prompt": prompt, "stream": False, "format": esquema, "keep_alive": "30m",
             "options": {"temperature": 0, "seed": piloto.SEMENTE, "num_ctx": 8192, "num_predict": maximo_tokens}}
    if sem_pensar:
        corpo["think"] = False
    for tentativa in range(piloto.TENTATIVAS):
        try:
            resposta = piloto.chamar(f"{host}/api/generate", corpo)
            break
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, ConnectionError):  # o Ollama pode reiniciar: espera e tenta de novo
            if tentativa == piloto.TENTATIVAS - 1:
                raise
            time.sleep(30)
    return resposta.get("response") or "", resposta.get("eval_count"), resposta.get("done_reason") == "length"


def extrair(prompt, modelo, host, sem_pensar):
    """Relações sem repetições, a resposta crua, se ela veio cortada e os tokens (como piloto.extrair)."""
    crua, tokens, cortada = perguntar(prompt, modelo, host, ESQUEMA, sem_pensar)
    try:
        relacoes = json.loads(crua).get("relacoes", [])
    except json.JSONDecodeError:  # resposta cortada: aproveita as relações que chegaram completas
        padrao = r'\{\s*"origem"\s*:\s*"([^"]*)"\s*,\s*"relacao"\s*:\s*"([^"]*)"\s*,\s*"destino"\s*:\s*"([^"]*)"\s*\}'
        relacoes, cortada = [dict(zip(("origem", "relacao", "destino"), m)) for m in re.findall(padrao, crua)], True
    limpas, vistas = [], set()
    for r in relacoes:
        r = {k: str(r.get(k, "")).strip() for k in ("origem", "relacao", "destino")}
        chave = (piloto.normalizar(r["origem"]), r["relacao"], piloto.normalizar(r["destino"]))
        if chave[0] and chave[2] and r["relacao"] in TIPOS and chave not in vistas:
            vistas.add(chave)
            limpas.append(r)
    return limpas, crua, cortada, tokens


def registros(chave):
    """O último registro de cada nota já feita na forma (uma linha cortada é ignorada)."""
    saida = {}
    if os.path.exists(arquivo_da_forma(chave)):
        with open(arquivo_da_forma(chave), encoding="utf-8") as f:
            for linha in f:
                try:
                    r = json.loads(linha)
                except json.JSONDecodeError:
                    continue
                saida[r["nota"]] = r
    return saida


def conferir_relacoes(n, relacoes, modelo, host, sem_pensar):
    """A segunda leitura: as relações que o modelo confirma, as que ele rejeita e as respostas cruas."""
    ficam, saem, cruas = [], [], []
    for i in range(0, len(relacoes), RELACOES_POR_CONFERENCIA):
        lote = relacoes[i:i + RELACOES_POR_CONFERENCIA]
        lista = "\n".join(f"{k}. {r['origem']} -> {r['relacao']} -> {r['destino']}" for k, r in enumerate(lote, 1))
        crua, _, _ = perguntar(PROMPT_CONFERIR.format(tipos=texto_dos_tipos(), texto=n["nota"]["texto"],
                                                      relacoes=lista), modelo, host, ESQUEMA_CONFERIR, sem_pensar,
                               maximo_tokens=1500)
        cruas.append(crua)
        try:
            certas = {x.get("numero"): x.get("certa") for x in json.loads(crua).get("respostas", [])}
        except json.JSONDecodeError:
            certas = {}
        for k, r in enumerate(lote, 1):
            (saem if certas.get(k) is False else ficam).append(r)  # sem resposta, a relação fica
    return ficam, saem, cruas


def carregar_o_modelo(modelo, host, sem_pensar):
    """Carrega o modelo antes de medir o tempo da primeira nota."""
    piloto.chamar(f"{host}/api/generate", {"model": modelo, "prompt": "ok", "stream": False, "keep_alive": "30m",
                                           **({"think": False} if sem_pensar else {}),
                                           "options": {"num_ctx": 8192, "num_predict": 1}})


def ligar_nota(n, modelo, host, sem_pensar, com_exemplos):
    """Os campos do registro de uma nota: as relações de todos os pedidos dela, sem repetir as que outro pedaço da
    mesma nota já deu, as respostas cruas, se alguma veio cortada e os tokens. Uma nota sem conceitos (fora os
    valores e os tempos) não vai para o modelo."""
    if not any(c["categoria"] not in ligacoes.LITERAIS for c in n["lista"]):
        return {"pedidos": 0, "relacoes": [], "respostas": [], "cortada": False, "tokens": 0}
    prompts, extraidas, vistas, cruas, cortada, tokens = pedidos(n, com_exemplos), [], set(), [], False, 0
    for prompt in prompts:
        novas, crua, cortada_aqui, tokens_aqui = extrair(prompt, modelo, host, sem_pensar)
        cruas.append(crua)
        cortada, tokens = cortada or cortada_aqui, tokens + (tokens_aqui or 0)
        for r in novas:
            k = (piloto.normalizar(r["origem"]), r["relacao"], piloto.normalizar(r["destino"]))
            if k not in vistas:
                vistas.add(k)
                extraidas.append(r)
    return {"pedidos": len(prompts), "relacoes": extraidas, "respostas": cruas, "cortada": cortada, "tokens": tokens}


def rodar(chave, host):
    try:
        versao = piloto.chamar(f"{host}/api/version", timeout=10).get("version")
    except (OSError, ValueError) as erro:
        sys.exit(f"Não foi possível falar com o Ollama em {host} ({erro}). Ele está rodando?")
    modelo = modelo_da_forma(chave)
    sem_pensar = pensa(modelo, host)
    notas, feitas = notas_de_teste(), {a for a, r in registros(chave).items() if not r.get("erro")}
    base = registros(chave[2:]) if chave.startswith("5-") else None
    if base is not None and len([a for a, r in base.items() if not r.get("erro")]) < len(notas):
        sys.exit(f"A forma {chave[2:]} ainda não terminou: rode antes a forma {chave[2:]}.")
    pendentes = [n for n in notas if n["arquivo"] not in feitas]
    print(f"Ollama {versao}; forma {chave}: {nome_da_forma(chave)}{' (sem pensar)' if sem_pensar else ''};"
          f" {len(pendentes)} de {len(notas)} notas para fazer")
    if not pendentes:
        return
    carregar_o_modelo(modelo, host, sem_pensar)
    os.makedirs(PASTA, exist_ok=True)
    for numero, n in enumerate(pendentes, 1):
        comeco, registro = time.time(), {"nota": n["arquivo"], "forma": chave, "modelo": modelo}
        try:
            if base is not None:
                registro["base"] = chave[2:]
                ficam, saem, cruas = conferir_relacoes(n, base[n["arquivo"]]["relacoes"], modelo, host, sem_pensar)
                registro.update(relacoes=ficam, rejeitadas=saem, respostas=cruas)
            else:
                registro.update(ligar_nota(n, modelo, host, sem_pensar, FORMAS[chave][1]))
        except (OSError, ValueError) as erro:  # sem conexão, erro do servidor ou resposta que não é JSON
            registro.update(relacoes=[], erro=repr(erro))
        registro["segundos"] = round(time.time() - comeco, 1)
        with open(arquivo_da_forma(chave), "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        print(f"   [{numero}/{len(pendentes)}] {n['arquivo']}: {len(registro['relacoes'])} relações"
              + (f", {len(registro['rejeitadas'])} rejeitadas" if "rejeitadas" in registro else "")
              + f" ({registro['segundos']} s)" + (f" ERRO: {registro['erro']}" if registro.get("erro") else ""))


def chaves_feitas():
    """As formas que já têm registros, na ordem da bancada."""
    todas = [*FORMAS, *(f"5-{k}" for k in FORMAS)]
    return [k for k in todas if os.path.exists(arquivo_da_forma(k))]


def contagens_entre_conceitos(n, relacoes):
    """As contagens de piloto.avaliar de uma nota, só entre conceitos (o mesmo recorte de ligacoes.medir)."""
    nota = {**n["nota"], "relacoes": n["relacoes_entre_conceitos"]}
    relacoes = [r for r in relacoes
                if r["relacao"] not in ligacoes.TIPOS_DE_LITERAIS and not set(r["categorias"]) & ligacoes.LITERAIS]
    a = piloto.avaliar(nota, relacoes, "livre")
    return {"achadas": a["relacoes_anotadas_achadas_parcial"], "anotadas": a["relacoes_anotadas"],
            "batem": a["relacoes_extraidas"] - len(a["novas"]), "extraidas": a["relacoes_extraidas"],
            "negacoes_achadas": a["negacoes_achadas_parcial"], "negacoes": a["negacoes_anotadas"]}


def comparacoes(notas, relacoes):
    """As diferenças entre pares de formas nas mesmas notas, com o intervalo de 95% de um bootstrap pareado: as
    notas sorteadas com reposição, as mesmas para as duas formas. relacoes: {forma: {nota: relações}}."""
    contagens = {k: [contagens_entre_conceitos(n, relacoes[k][n["arquivo"]]) for n in notas] for k in relacoes}
    rng = random.Random(piloto.SEMENTE)
    sorteios = [[rng.randrange(len(notas)) for _ in notas] for _ in range(REAMOSTRAS)]

    def medida(k, indices, parte, todo):
        return piloto.fracao(sum(contagens[k][i][parte] for i in indices), sum(contagens[k][i][todo] for i in indices))

    saida = {}
    for a, b in COMPARACOES:
        if a in contagens and b in contagens:
            saida[f"{a} - {b}"] = {}
            for nome, (parte, todo) in MEDIDAS_COMPARADAS.items():
                diferencas = sorted(medida(a, s, parte, todo) - medida(b, s, parte, todo) for s in sorteios
                                    if sum(contagens[a][i][todo] for i in s) and sum(contagens[b][i][todo] for i in s))
                todas = range(len(notas))
                saida[f"{a} - {b}"][nome] = {
                    "diferenca": medida(a, todas, parte, todo) - medida(b, todas, parte, todo),
                    "de": diferencas[round(0.025 * (len(diferencas) - 1))],
                    "ate": diferencas[round(0.975 * (len(diferencas) - 1))]}
    return saida


def medir():
    notas = notas_de_teste()
    with open(ligacoes.SAIDA_JSON, encoding="utf-8") as f:
        teste_5e = json.load(f)
    referencias = teste_5e["medidas"]
    registros_5d = grafo.registros_de_relacoes()
    completas = {"5d": {n["arquivo"]: ligacoes.do_qwen(n, registros_5d[n["arquivo"]]) for n in notas},
                 "5e": teste_5e["relacoes"]["modelo treinado"]}
    medidas, situacao = {}, {}
    for chave in chaves_feitas():
        regs = registros(chave)
        prontas = [n for n in notas if n["arquivo"] in regs and not regs[n["arquivo"]].get("erro")]
        if not prontas:
            continue
        relacoes = {n["arquivo"]: ligacoes.do_qwen(n, regs[n["arquivo"]]) for n in prontas}
        medidas[chave] = ligacoes.medir(prontas, relacoes)
        if len(prontas) == len(notas):
            completas[chave] = relacoes
        segundos = [regs[n["arquivo"]]["segundos"] for n in prontas]
        situacao[chave] = {"nome": nome_da_forma(chave), "notas": len(prontas),
                           "com_erro": sum(1 for r in regs.values() if r.get("erro")),
                           "segundos_por_nota": sum(segundos) / len(segundos),
                           "respostas_cortadas": sum(1 for n in prontas if regs[n["arquivo"]].get("cortada"))}
    trata = medir_trata()
    diferencas = comparacoes(notas, completas)
    grafo.salvar({"notas_de_teste": len(notas), "formas": situacao, "medidas": medidas,
                  "referencias_do_5e": referencias, "comparacoes": diferencas, "trata_conferido_a_mao": trata},
                 SAIDA_JSON)

    pct = piloto.pct
    linhas = [(f"5e: {nome}", m, None) for nome, m in referencias.items() if nome != "modelo, conceitos anotados"]
    linhas += [(f"{k}: {situacao[k]['nome']}" + ("" if situacao[k]["notas"] == len(notas)
                                                  else f" ({situacao[k]['notas']} notas)"), medidas[k], situacao[k])
               for k in medidas]
    for nome in ("entre conceitos", "todas"):
        titulo = "só entre conceitos (sem valor, dose e tempo)" if nome == "entre conceitos" else "todas as ligações"
        print(f"\n== {len(notas)} notas de teste, {titulo}")
        print(f"   {'forma':56s}{'ligações/nota':>14s}{'anotadas achadas':>18s}{'(exata)':>9s}"
              f"{'batem c/ anotações':>20s}{'negações':>10s}{'trata/nota':>12s}{'s/nota':>8s}")
        for rotulo, m, s in linhas:
            r = m[nome]
            print(f"   {rotulo:56s}{r['relacoes_por_nota']:14.1f}{pct(r['relacoes_anotadas_achadas_parcial']):>18s}"
                  f"{pct(r['relacoes_anotadas_achadas_exata']):>9s}"
                  f"{pct(None if r['relacoes_novas'] is None else 1 - r['relacoes_novas']):>20s}"
                  f"{pct(r['negacoes_achadas_parcial']):>10s}{m['trata_por_nota']:12.1f}"
                  + (f"{s['segundos_por_nota']:8.0f}" if s else ""))
    for k, s in situacao.items():
        if s["com_erro"] or s["respostas_cortadas"]:
            print(f"   {k}: {s['com_erro']} notas com erro, {s['respostas_cortadas']} com resposta cortada")
    print(f"\n== Diferenças nas mesmas {len(notas)} notas, só entre conceitos, em pontos, com o intervalo de 95%"
          f" ({REAMOSTRAS} sorteios das notas; 5d é o qwen do 5d e 5e, o modelo treinado do 5e)")
    print(f"   {'comparação':14s}" + "".join(f"{nome:>28s}" for nome in MEDIDAS_COMPARADAS))
    for par, d in diferencas.items():
        print(f"   {par:14s}" + "".join(f"{100 * m['diferenca']:+.1f} ({100 * m['de']:+.1f} a {100 * m['ate']:+.1f})"
                                       .rjust(28) for m in d.values()))
    if trata:
        print("\n   \"trata\" conferidos à mão (certas / respondidas):")
        for k, t in trata.items():
            print(f"   {k:14s}{t['certas']:4d} / {t['respondidas']:<4d}{pct(t['fracao']):>8s}")
    print(f"\nMedidas em {os.path.relpath(SAIDA_JSON, RAIZ)}")


def chave_do_trata(nota, r):
    """A mesma relação "trata" em formas diferentes: a nota e os dois termos, normalizados."""
    return nota, piloto.normalizar(r["origem"]), piloto.normalizar(r["destino"])


def amostra():
    """A planilha com até TRATA_POR_FORMA relações "trata" de cada forma (e do qwen no 5d), misturadas, para
    conferir à mão; as respostas seguem as letras da conferência do 5d (05d_conferir_relacoes.py). A mesma relação
    sorteada em mais de uma forma vira uma linha só. As formas com segunda leitura não têm sorteio próprio: as
    relações delas são parte das da forma de base, e as linhas da base que a segunda leitura manteve já são uma
    amostra delas (veja medir_trata)."""
    if os.path.exists(PLANILHA_TRATA):
        sys.exit(f"{os.path.relpath(PLANILHA_TRATA, RAIZ)} já existe e pode ter respostas.")
    notas = {n["arquivo"]: n for n in notas_de_teste()}
    rng, linhas, por_chave = random.Random(piloto.SEMENTE), [], {}
    fontes = {"5d": {a: r for a, r in grafo.registros_de_relacoes().items() if a in notas}}
    fontes.update({k: registros(k) for k in chaves_feitas() if not k.startswith("5-")})
    for forma, regs in fontes.items():
        tratas = [(a, r) for a, reg in sorted(regs.items()) if not reg.get("erro")
                  for r in reg["relacoes"] if r["relacao"] == "trata"]
        for a, r in rng.sample(tratas, min(TRATA_POR_FORMA, len(tratas))):
            if chave_do_trata(a, r) not in por_chave:
                por_chave[chave_do_trata(a, r)] = {"nota": a, **r, "formas": []}
                linhas.append(por_chave[chave_do_trata(a, r)])
            if forma not in por_chave[chave_do_trata(a, r)]["formas"]:
                por_chave[chave_do_trata(a, r)]["formas"].append(forma)
    rng.shuffle(linhas)  # mistura as formas, para quem confere não saber de onde veio cada relação
    with open(PLANILHA_TRATA, "w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.writer(f, delimiter=";")
        escritor.writerow(conferir_5d.COLUNAS)
        for numero, x in enumerate(linhas, 1):
            texto = ner.texto_cru(notas[x["nota"]]["caminho"])
            escritor.writerow([numero, x["nota"], conferir_5d.celula(f"{x['origem']} → trata → {x['destino']}"),
                               conferir_5d.celula(conferir_5d.trecho(texto, x["origem"], x["destino"])), "", ""])
    grafo.salvar({"linhas": [{"numero": numero, **x} for numero, x in enumerate(linhas, 1)]}, AMOSTRA_TRATA)
    por_forma = collections.Counter(forma for x in linhas for forma in x["formas"])
    print(f"Planilha: {os.path.relpath(PLANILHA_TRATA, RAIZ)} ({len(linhas)} linhas;"
          f" {', '.join(f'{k}: {por_forma[k]}' for k in fontes)};"
          f" {sum(1 for x in linhas if len(x['formas']) > 1)} linhas sorteadas em mais de uma forma)")


def medir_trata():
    """Quantas relações "trata" de cada forma a pessoa marcou como certas (s) na planilha, se ela foi preenchida.
    Numa forma com segunda leitura contam as linhas da forma de base que ela manteve; as que ela tirou aparecem à
    parte ("tiradas"), para ver se ela tira as erradas."""
    if not (os.path.exists(PLANILHA_TRATA) and os.path.exists(AMOSTRA_TRATA)):
        return None
    with open(AMOSTRA_TRATA, encoding="utf-8") as f:
        linhas = {x["numero"]: x for x in json.load(f)["linhas"]}
    respostas, _ = conferir_5d.ler_respostas(PLANILHA_TRATA)
    segundas = {k: registros(k) for k in chaves_feitas() if k.startswith("5-")}
    contagem = collections.defaultdict(collections.Counter)
    for numero, resposta in respostas.items():
        x = linhas[numero]
        for forma in x["formas"]:
            contagem[forma][resposta] += 1
            reg = segundas.get(f"5-{forma}", {}).get(x["nota"])
            if reg and not reg.get("erro"):
                ficou = chave_do_trata(x["nota"], x) in {chave_do_trata(x["nota"], r) for r in reg["relacoes"]
                                                         if r["relacao"] == "trata"}
                contagem[f"5-{forma}" + ("" if ficou else ", tiradas")][resposta] += 1
    ordem = [k for c in ("5d", *chaves_feitas()) for k in (c, f"{c}, tiradas")]
    return {k: {"certas": contagem[k]["s"], "respondidas": sum(contagem[k].values()),
                "fracao": piloto.fracao(contagem[k]["s"], sum(contagem[k].values()))}
            for k in ordem if k in contagem} or None


def main():
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    argumentos.add_argument("parte", choices=("rodar", "medir", "amostra"))
    argumentos.add_argument("forma", nargs="?", help="a forma da bancada (parte rodar): " + ", ".join(FORMAS)
                            + " ou 5-<forma> para a segunda leitura")
    argumentos.add_argument("--host", default="http://127.0.0.1:11434", help="endereço do Ollama")
    args = argumentos.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if args.parte == "rodar":
        forma = args.forma or ""
        if (forma[2:] if forma.startswith("5-") else forma) not in FORMAS:
            sys.exit(f"Forma desconhecida: {args.forma}")
        rodar(forma, args.host)
    elif args.parte == "medir":
        medir()
    else:
        amostra()


if __name__ == "__main__":
    main()
