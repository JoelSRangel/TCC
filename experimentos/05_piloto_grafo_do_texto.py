"""
Experimento 5 (piloto): montar o grafo a partir do texto das notas.

O grafo dos experimentos anteriores vem das anotações do SemClinBr, feitas
à mão: só tem dois tipos de relação ("associado a" e "negação de"), e 65%
dos conceitos ligados têm um único vizinho. Esta é uma metodologia
adicional: um modelo de linguagem lê o texto de cada nota e extrai as
relações, com tipos mais ricos. As anotações do SemClinBr servem de
gabarito.

Duas formas de extração, com cada modelo de linguagem:
  livre    o modelo recebe só o texto e encontra os conceitos e as relações
  guiada   o modelo recebe também os conceitos marcados pelos anotadores e
           só decide as relações entre eles

Medidas, comparando com as anotações das mesmas notas:
  conceitos no texto        fração dos conceitos extraídos que aparecem no
                            texto da nota (os outros foram inventados)
  conceitos anotados achados  fração dos conceitos anotados que aparecem
                            entre os extraídos (só na forma livre)
  relações anotadas achadas fração das relações anotadas cujas duas pontas
                            aparecem ligadas na extração, em qualquer
                            sentido e com qualquer tipo
  negações achadas          o mesmo, só para as negações, e com o tipo
                            "nega" na extração
  relações novas            extraídas que não estão nas anotações; uma
                            amostra vai para um arquivo de avaliação
                            humana, porque as anotações não cobrem tudo

Comparação de textos: maiúsculas, sem acentos, sem pontuação e sem o que
estiver entre parênteses. "Exata": textos iguais; "parcial": um contém o
outro em palavras inteiras e cobre pelo menos metade das palavras dele
(por exemplo, "acesso venoso periférico heparinizado" e "acesso venoso
periférico"; uma frase inteira não casa com cada palavra que contém).

Os modelos rodam no Ollama local (as notas não saem do computador), com
temperatura 0 e resposta em JSON no formato pedido.

Saídas (não versionadas):
  results/experimento5_piloto.json
  results/experimento5_relacoes_para_avaliar.csv   (abre no Excel)

Uso, com o Ollama rodando e os modelos baixados (ollama pull llama3.2;
ollama pull qwen2.5:7b). O script só usa a biblioteca padrão do Python, então
roda no WSL e no Windows; o Ollama do Windows usa a placa de vídeo e é bem mais
rápido que o do WSL, que só usa o processador. As respostas mudam um pouco de
uma máquina para a outra, mesmo com temperatura 0 (na placa, as medidas do
qwen2.5:7b ficaram até cerca de 2 pontos longe das do processador):
    python experimentos/05_piloto_grafo_do_texto.py --notas 2   # teste rápido
    python experimentos/05_piloto_grafo_do_texto.py             # 20 notas
"""
import argparse
import collections
import csv
import glob
import json
import os
import random
import re
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_XML = os.path.join(RAIZ, "data", "raw")
SAIDA_JSON = os.path.join(RAIZ, "results", "experimento5_piloto.json")
SAIDA_CSV = os.path.join(RAIZ, "results", "experimento5_relacoes_para_avaliar.csv")
MODELOS = ("llama3.2", "qwen2.5:7b")
FORMAS = ("livre", "guiada")
SEMENTE = 0
NOVAS_PARA_AVALIAR = 10  # por modelo e forma
TENTATIVAS = 20  # se o Ollama cair, tenta de novo a cada 30 s, por até 10 minutos

TIPOS = {
    "localizado_em": "achado, sintoma, lesão, dispositivo ou procedimento -> parte do corpo ou região"
                     " (ex.: dor -> MIE)",
    "tem_dose": "medicamento, dieta ou terapia -> dose, via ou frequência (ex.: dipirona -> 1 g 6/6h)",
    "tem_valor": "medida, sinal vital ou exame -> valor numérico (ex.: PA -> 120/80 mmHg)",
    "tem_resultado": "exame ou avaliação -> resultado ou achado (ex.: RX de tórax -> sem alterações)",
    "trata": "medicamento ou procedimento -> doença ou sintoma que trata (ex.: dipirona -> febre)",
    "tem_tempo": "evento, achado ou procedimento -> quando ocorreu ou quanto durou (ex.: cirurgia -> há 2 dias)",
    "nega": "palavra de negação -> o que foi negado (ex.: sem -> febre)",
    "associado_a": "outra relação clínica entre os dois conceitos",
}

MAX_RELACOES = 30  # sem um limite, os modelos podem repetir relações até estourar o tamanho da resposta
ESQUEMA = {
    "type": "object",
    "properties": {"relacoes": {"type": "array", "maxItems": MAX_RELACOES, "items": {
        "type": "object",
        "properties": {"origem": {"type": "string"}, "relacao": {"type": "string", "enum": list(TIPOS)},
                       "destino": {"type": "string"}},
        "required": ["origem", "relacao", "destino"],
    }}},
    "required": ["relacoes"],
}

PROMPT = """Você é um especialista em informática clínica. Leia a nota clínica abaixo, escrita em \
português por profissionais de saúde, e extraia as relações entre os conceitos clínicos que ela contém.

Regras:
- Cada relação liga dois conceitos que aparecem no texto. Copie cada conceito exatamente como está \
escrito na nota.
- Cada conceito deve ser curto, de uma a quatro palavras (por exemplo: "dor", "MIE", "acesso venoso \
periférico"), e não uma frase inteira.
- Não invente conceitos nem relações que a nota não afirma. Os exemplos abaixo só explicam os tipos; \
não os copie.
- Não repita relações. Liste no máximo {maximo} relações, as mais importantes.
{regra_lista}- Use um destes tipos de relação:
{tipos}

Nota clínica:
\"\"\"{texto}\"\"\"

Responda em JSON, no formato {{"relacoes": [{{"origem": "...", "relacao": "...", "destino": "..."}}]}}."""

REGRA_LISTA = """- Use como origem e destino apenas os conceitos desta lista, já marcados por anotadores \
humanos (a categoria entre parênteses é só uma ajuda; não a copie):
{conceitos}
"""


def normalizar(t):
    """Maiúsculas, sem acentos, sem pontuação e sem o que estiver entre parênteses."""
    t = re.sub(r"\([^)]*\)", " ", t or "")
    t = unicodedata.normalize("NFKD", t.upper())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^A-Z0-9]+", " ", t).split())


def casa(a, b, parcial):
    """Os dois textos (já normalizados) são o mesmo conceito: iguais, ou um contém o outro e cobre
    pelo menos metade das palavras dele (uma frase inteira não casa com cada palavra que contém)."""
    if not a or not b:
        return False
    if a == b:
        return True
    curto, longo = sorted((a, b), key=len)
    return parcial and f" {curto} " in f" {longo} " and len(curto.split()) >= len(longo.split()) / 2


def ler_nota(arquivo):
    raiz = ET.parse(arquivo).getroot()
    anotacoes = {a.get("id"): {"texto": a.get("text") or "", "categoria": a.get("tag") or ""}
                 for a in raiz.findall(".//annotation")}
    relacoes = [(anotacoes[r.get("annotation1")]["texto"], r.get("reltype"), anotacoes[r.get("annotation2")]["texto"])
                for r in raiz.findall(".//rel")
                if r.get("annotation1") in anotacoes and r.get("annotation2") in anotacoes]
    return {"arquivo": os.path.basename(arquivo), "texto": raiz.findtext("TEXT") or "",
            "anotacoes": list(anotacoes.values()), "relacoes": relacoes}


def prompt_para(nota, forma):
    regra_lista = ""
    if forma == "guiada":
        vistos, linhas = set(), []
        for a in nota["anotacoes"]:
            if normalizar(a["texto"]) and normalizar(a["texto"]) not in vistos:
                vistos.add(normalizar(a["texto"]))
                linhas.append(f"  - {a['texto']} ({a['categoria']})")
        regra_lista = REGRA_LISTA.format(conceitos="\n".join(linhas))
    tipos = "\n".join(f"  - {nome}: {descricao}" for nome, descricao in TIPOS.items())
    return PROMPT.format(regra_lista=regra_lista, tipos=tipos, texto=nota["texto"], maximo=MAX_RELACOES)


def chamar(url, corpo=None, timeout=1800):
    """Resposta em JSON do Ollama: POST quando há corpo, GET quando não há."""
    dados = None if corpo is None else json.dumps(corpo).encode("utf-8")
    pedido = urllib.request.Request(url, data=dados, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(pedido, timeout=timeout) as resposta:
        return json.loads(resposta.read().decode("utf-8"))


def perguntar(prompt, modelo, host, esquema, maximo_tokens=2000):
    """Resposta crua do modelo (JSON no formato do esquema) e quantos tokens ele gerou."""
    for tentativa in range(TENTATIVAS):
        try:
            corpo = chamar(f"{host}/api/generate",
                           {"model": modelo, "prompt": prompt, "stream": False, "format": esquema, "keep_alive": "30m",
                            "options": {"temperature": 0, "seed": SEMENTE, "num_ctx": 8192,
                                        "num_predict": maximo_tokens}})
            break
        except urllib.error.HTTPError:
            raise  # o servidor respondeu com erro: tentar de novo não adianta
        except (urllib.error.URLError, ConnectionError):
            # o servidor do Ollama pode reiniciar no meio do piloto (aconteceu na primeira rodada): espera e tenta de novo
            if tentativa == TENTATIVAS - 1:
                raise
            time.sleep(30)
    return corpo.get("response") or "", corpo.get("eval_count")


def extrair(prompt, modelo, host):
    """Relações extraídas (sem repetições), a resposta crua e se ela veio cortada."""
    crua, tokens = perguntar(prompt, modelo, host, ESQUEMA)
    try:
        relacoes, cortada = json.loads(crua).get("relacoes", []), False
    except json.JSONDecodeError:
        # resposta cortada no limite de tamanho: aproveita as relações que chegaram completas
        padrao = r'\{\s*"origem"\s*:\s*"([^"]*)"\s*,\s*"relacao"\s*:\s*"([^"]*)"\s*,\s*"destino"\s*:\s*"([^"]*)"\s*\}'
        relacoes = [dict(zip(("origem", "relacao", "destino"), m)) for m in re.findall(padrao, crua)]
        cortada = True
    limpas, vistas = [], set()
    for r in relacoes:
        r = {k: str(r.get(k, "")).strip() for k in ("origem", "relacao", "destino")}
        chave = (normalizar(r["origem"]), r["relacao"], normalizar(r["destino"]))
        if chave[0] and chave[2] and chave not in vistas:
            vistas.add(chave)
            limpas.append(r)
    return limpas, crua, cortada, tokens


def avaliar(nota, extraidas, forma):
    """Compara a extração de uma nota com as anotações dela."""
    texto = f" {normalizar(nota['texto'])} "
    anotados = {normalizar(a["texto"]) for a in nota["anotacoes"]} - {""}
    conceitos = {normalizar(r[k]) for r in extraidas for k in ("origem", "destino")}
    pares = [(normalizar(r["origem"]), normalizar(r["destino"]), r["relacao"]) for r in extraidas]
    gabarito = {}
    for a, tipo, b in nota["relacoes"]:
        a, b = normalizar(a), normalizar(b)
        if a and b and a != b:
            gabarito[frozenset((a, b))] = (a, b, tipo)
    resultado = {"relacoes_extraidas": len(extraidas), "conceitos_extraidos": len(conceitos),
                 "conceitos_no_texto": sum(f" {c} " in texto for c in conceitos),
                 "relacoes_anotadas": len(gabarito),
                 "negacoes_anotadas": sum(tipo == "negation_of" for _, _, tipo in gabarito.values()),
                 "tipos": dict(collections.Counter(r["relacao"] for r in extraidas))}
    if forma == "guiada":
        resultado["conceitos_da_lista"] = sum(c in anotados for c in conceitos)
    for modo in ("exata", "parcial"):
        p = modo == "parcial"

        def liga(a, b, so_negacao=False):
            return any(((casa(o, a, p) and casa(d, b, p)) or (casa(o, b, p) and casa(d, a, p)))
                       and (not so_negacao or t == "nega") for o, d, t in pares)

        if forma == "livre":
            resultado[f"conceitos_anotados_achados_{modo}"] = sum(any(casa(a, c, p) for c in conceitos) for a in anotados)
        resultado[f"relacoes_anotadas_achadas_{modo}"] = sum(liga(a, b) for a, b, _ in gabarito.values())
        resultado[f"negacoes_achadas_{modo}"] = sum(liga(a, b, True) for a, b, tipo in gabarito.values()
                                                    if tipo == "negation_of")
    resultado["novas"] = [r for r, (o, d, _) in zip(extraidas, pares)
                          if not any((casa(o, a, True) and casa(d, b, True)) or (casa(o, b, True) and casa(d, a, True))
                                     for a, b, _ in gabarito.values())]
    resultado["anotados_distintos"] = len(anotados)
    return resultado


def trecho(texto, *termos, margem=90):
    """Trecho da nota em volta do primeiro termo encontrado, para quem for avaliar."""
    for termo in termos:
        i = texto.lower().find(termo.lower())
        if i >= 0:
            inicio, fim = max(0, i - margem), min(len(texto), i + len(termo) + margem)
            return ("…" if inicio else "") + texto[inicio:fim].replace("\n", " ") + ("…" if fim < len(texto) else "")
    return "(nenhum dos dois termos aparece igual no texto)"


def fracao(parte, todo):
    return parte / todo if todo else None


def resumir(execucoes, forma):
    soma = collections.Counter()
    tipos = collections.Counter()
    for e in execucoes:
        for chave, valor in e["avaliacao"].items():
            if isinstance(valor, (int, float)) and not isinstance(valor, bool):
                soma[chave] += valor
        tipos.update(e["avaliacao"]["tipos"])
    resumo = {
        "notas": len(execucoes), "falhas": sum(1 for e in execucoes if e.get("erro")),
        "respostas_cortadas": sum(1 for e in execucoes if e.get("cortada")),
        "relacoes_por_nota": fracao(soma["relacoes_extraidas"], len(execucoes)),
        "conceitos_no_texto": fracao(soma["conceitos_no_texto"], soma["conceitos_extraidos"]),
        "relacoes_novas": fracao(sum(len(e["avaliacao"]["novas"]) for e in execucoes), soma["relacoes_extraidas"]),
        "segundos_por_nota": statistics.fmean(e["segundos"] for e in execucoes) if execucoes else None,
        "tipos": dict(tipos.most_common()),
    }
    for modo in ("exata", "parcial"):
        resumo[f"relacoes_anotadas_achadas_{modo}"] = fracao(soma[f"relacoes_anotadas_achadas_{modo}"],
                                                             soma["relacoes_anotadas"])
        resumo[f"negacoes_achadas_{modo}"] = fracao(soma[f"negacoes_achadas_{modo}"], soma["negacoes_anotadas"])
        if forma == "livre":
            resumo[f"conceitos_anotados_achados_{modo}"] = fracao(soma[f"conceitos_anotados_achados_{modo}"],
                                                                  soma["anotados_distintos"])
    if forma == "guiada":
        resumo["conceitos_da_lista"] = fracao(soma["conceitos_da_lista"], soma["conceitos_extraidos"])
    return resumo


def pct(v):
    return "    -" if v is None else f"{100 * v:5.1f}%"


def main():
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    argumentos.add_argument("--notas", type=int, default=20, help="notas sorteadas para o piloto")
    argumentos.add_argument("--modelos", nargs="+", default=list(MODELOS), help="modelos do Ollama")
    argumentos.add_argument("--host", default="http://127.0.0.1:11434", help="endereço do Ollama")
    args = argumentos.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    try:
        instalados = {m["name"] for m in chamar(f"{args.host}/api/tags", timeout=10).get("models", [])}
    except (OSError, ValueError) as erro:
        sys.exit(f"Não foi possível falar com o Ollama em {args.host} ({erro}). Ele está rodando?")
    faltando = [m for m in args.modelos if m not in instalados and f"{m}:latest" not in instalados]
    if faltando:
        sys.exit(f"Modelos que não estão no Ollama: {faltando}. Rode: ollama pull <modelo>")

    arquivos = sorted(glob.glob(os.path.join(PASTA_XML, "*.xml")))
    sorteadas = sorted(random.Random(SEMENTE).sample(arquivos, min(args.notas, len(arquivos))))
    notas = [ler_nota(a) for a in sorteadas]
    print(f"{len(notas)} notas sorteadas (semente {SEMENTE}): {sum(len(n['texto']) for n in notas)} caracteres,"
          f" {sum(len(n['anotacoes']) for n in notas)} anotações, {sum(len(n['relacoes']) for n in notas)} relações")

    dados = {"protocolo": {"semente": SEMENTE, "modelos": args.modelos, "formas": FORMAS, "tipos": TIPOS,
                           "prompt": PROMPT, "regra_lista": REGRA_LISTA, "esquema": ESQUEMA},
             "notas": [n["arquivo"] for n in notas], "execucoes": [], "resumo": {}}
    for modelo in args.modelos:
        for forma in FORMAS:
            print(f"\n== {modelo}, extração {forma}")
            for numero, nota in enumerate(notas, 1):
                comeco = time.time()
                registro = {"modelo": modelo, "forma": forma, "nota": nota["arquivo"]}
                try:
                    extraidas, crua, cortada, tokens = extrair(prompt_para(nota, forma), modelo, args.host)
                except (OSError, ValueError) as erro:  # sem conexão, erro do servidor ou resposta que não é JSON
                    extraidas, crua, cortada, tokens = [], None, None, None
                    registro["erro"] = repr(erro)  # registra a falha em vez de esconder
                registro.update(relacoes=extraidas, resposta=crua, cortada=cortada, tokens=tokens,
                                segundos=round(time.time() - comeco, 1), avaliacao=avaliar(nota, extraidas, forma))
                dados["execucoes"].append(registro)
                with open(SAIDA_JSON, "w", encoding="utf-8") as f:
                    json.dump(dados, f, ensure_ascii=False, indent=2)
                a = registro["avaliacao"]
                situacao = " FALHOU" if registro.get("erro") else " (resposta cortada)" if cortada else ""
                print(f"   [{numero}/{len(notas)}] {nota['arquivo']}: {a['relacoes_extraidas']} relações,"
                      f" {a['relacoes_anotadas_achadas_parcial']}/{a['relacoes_anotadas']} anotadas achadas"
                      f"{situacao} ({registro['segundos']} s)")
            execucoes = [e for e in dados["execucoes"] if e["modelo"] == modelo and e["forma"] == forma]
            dados["resumo"][f"{modelo} | {forma}"] = resumir(execucoes, forma)
            with open(SAIDA_JSON, "w", encoding="utf-8") as f:
                json.dump(dados, f, ensure_ascii=False, indent=2)

    # Amostra de relações novas (fora das anotações) para avaliação humana
    rng = random.Random(SEMENTE)
    texto_da_nota = {n["arquivo"]: n["texto"] for n in notas}
    linhas = []
    for modelo in args.modelos:
        for forma in FORMAS:
            novas = [(e["nota"], r) for e in dados["execucoes"]
                     if e["modelo"] == modelo and e["forma"] == forma for r in e["avaliacao"]["novas"]]
            for nota, r in rng.sample(novas, min(NOVAS_PARA_AVALIAR, len(novas))):
                linhas.append([modelo, forma, nota, r["origem"], r["relacao"], r["destino"],
                               trecho(texto_da_nota[nota], r["origem"], r["destino"]), "", ""])
    rng.shuffle(linhas)  # mistura modelos e formas, para quem avalia não saber de onde veio cada uma
    with open(SAIDA_CSV, "w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.writer(f, delimiter=";")
        escritor.writerow(["modelo", "forma", "nota", "origem", "relacao", "destino", "trecho da nota",
                           "faz sentido? (s/n)", "observação"])
        escritor.writerows(linhas)

    print("\n== Resumo (somando as notas)")
    print(f"   {'modelo | forma':24s}{'relações/nota':>14s}{'conceitos no texto':>20s}{'anotadas achadas':>18s}"
          f"{'(exata)':>9s}{'negações':>10s}{'novas':>8s}{'s/nota':>8s}")
    for nome, r in dados["resumo"].items():
        print(f"   {nome:24s}{r['relacoes_por_nota']:14.1f}{pct(r['conceitos_no_texto']):>20s}"
              f"{pct(r['relacoes_anotadas_achadas_parcial']):>18s}{pct(r['relacoes_anotadas_achadas_exata']):>9s}"
              f"{pct(r['negacoes_achadas_parcial']):>10s}{pct(r['relacoes_novas']):>8s}{r['segundos_por_nota']:8.0f}")
    for nome, r in dados["resumo"].items():
        extra = (f"conceitos anotados achados {pct(r['conceitos_anotados_achados_parcial']).strip()}"
                 if "conceitos_anotados_achados_parcial" in r else
                 f"conceitos dentro da lista {pct(r['conceitos_da_lista']).strip()}")
        print(f"   {nome}: {extra}; falhas {r['falhas']}; respostas cortadas {r['respostas_cortadas']};"
              f" tipos {r['tipos']}")
    print(f"\nResultados em {os.path.relpath(SAIDA_JSON, RAIZ)}")
    print(f"Relações novas para avaliar à mão: {os.path.relpath(SAIDA_CSV, RAIZ)} ({len(linhas)} linhas)")


if __name__ == "__main__":
    main()
