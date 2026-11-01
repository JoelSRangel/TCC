"""
Experimento 5b (piloto): a versão com categorias.

No experimento 5, o modelo de linguagem extraiu relações do texto das notas de
duas formas: "livre" (acha os conceitos e as relações sozinho) e "guiada"
(recebe os conceitos marcados pelos anotadores). Esta é a versão com
categorias, que também roda sem as anotações, em duas etapas:
  1. o modelo lista os conceitos da nota, cada um com uma categoria;
  2. o modelo liga os conceitos dessa lista, com o mesmo pedido da forma guiada.
Para comparar, a forma guiada roda de novo com as categorias agrupadas
("guiada com grupos"): a única diferença entre ela e a versão com categorias é
quem fez a lista, os anotadores ou o modelo. As formas livre e guiada também
rodam de novo, na mesma máquina, para a comparação ser justa.

Categorias: o SemClinBr marca cada conceito com tipos semânticos da UMLS (89
tipos, às vezes mais de um por conceito, como "Abbreviation|Finding"). Eles
foram agrupados em 8 grupos clínicos, mais "negacao" e "outro", com base nos
grupos semânticos da UMLS; a diferença é que doença fica separada de sinal ou
sintoma, e exame fica separado de procedimento. "Abbreviation" só diz que o
conceito é uma abreviatura e não entra em grupo nenhum. As anotações agrupadas
servem de gabarito.

Medidas, além das do experimento 5:
  lista de conceitos   etapa 1: fração dos conceitos listados que aparecem no
                       texto, e fração dos conceitos anotados que aparecem na
                       lista
  categoria certa      etapa 1: dos conceitos listados que casam com um
                       conceito anotado, a fração que recebeu o grupo do
                       gabarito (os marcados só como abreviatura ficam de fora)
  regras de categoria  cada tipo de relação só faz sentido entre certos grupos
                       (a dose sai de um medicamento e vai para um valor, o
                       local vai para uma parte do corpo...). Conta as relações
                       que respeitam as regras, as que só estão com a direção
                       trocada, as que as violam e as que têm uma ponta sem
                       categoria, separando as que batem com as anotações das
                       novas: um bom filtro descarta muitas novas e quase
                       nenhuma das que batem

Saída (não versionada):
  results/experimento5b_categorias.json

Uso, com o Ollama rodando e o qwen2.5:7b baixado (o Ollama do Windows usa a
placa de vídeo); como o experimento 5, só usa a biblioteca padrão do Python:
    python experimentos/05b_piloto_com_categorias.py
"""
import argparse
import collections
import glob
import importlib.util
import json
import os
import random
import re
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAIDA_JSON = os.path.join(RAIZ, "results", "experimento5b_categorias.json")
MODELO = "qwen2.5:7b"
FORMAS = ("livre", "guiada", "guiada com grupos", "com categorias")
MAX_CONCEITOS = 100  # das 20 notas do piloto, a que tem mais conceitos anotados tem 90


def carregar(nome, arquivo):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(RAIZ, "experimentos", arquivo))
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# Notas, pedidos, comparação de textos e medidas: os mesmos do experimento 5
piloto = carregar("piloto", "05_piloto_grafo_do_texto.py")

# Descrições ajustadas depois de um teste com 2 das 20 notas, em que o modelo chamava de "exame" os achados
# do exame físico (que a UMLS põe em "Finding") e de "outro" o estado do paciente
GRUPOS = {
    "doenca": "doença, diagnóstico ou lesão (ex.: HAS, pneumonia, fratura)",
    "sinal_sintoma": "sinal, sintoma ou achado, inclusive os achados do exame físico e o estado do paciente"
                     " (ex.: dor, febre, abdome flácido, consciente)",
    "medicamento": "medicamento, substância ou dieta (ex.: dipirona, soro fisiológico, dieta enteral)",
    "procedimento": "procedimento, cuidado ou dispositivo (ex.: curativo, cirurgia, sonda vesical)",
    "exame": "exame ou medida feita no paciente, ou resultado de laboratório (ex.: RX de tórax, hemograma, PA,"
             " creatinina)",
    "anatomia": "parte do corpo ou local (ex.: MIE, abdome, tórax)",
    "valor": "valor, quantidade ou qualidade (ex.: 120/80 mmHg, 1 g, moderado)",
    "tempo": "tempo, data ou duração (ex.: há 2 dias, 05/2013)",
    "negacao": "palavra de negação (ex.: nega, sem, ausente, ausência de)",
    "outro": "qualquer outro conceito, como pessoas, profissionais, setores e ideias (ex.: paciente, enfermeiro, UTI)",
}

# Os 88 tipos semânticos da UMLS usados no SemClinBr (fora "Abbreviation"), por grupo
TIPOS_DO_GRUPO = {
    "doenca": ("Disease or Syndrome", "Neoplastic Process", "Injury or Poisoning", "Mental or Behavioral Dysfunction",
               "Pathologic Function", "Anatomical Abnormality", "Congenital Abnormality", "Acquired Abnormality",
               "Cell or Molecular Dysfunction"),
    "sinal_sintoma": ("Sign or Symptom", "Finding", "Physiologic Function", "Organism Function",
                      "Organ or Tissue Function", "Mental Process", "Molecular Function"),
    "medicamento": ("Pharmacologic Substance", "Organic Chemical", "Antibiotic", "Clinical Drug", "Hormone", "Vitamin",
                    "Amino Acid, Peptide, or Protein", "Biologically Active Substance", "Inorganic Chemical",
                    "Immunologic Factor", "Hazardous or Poisonous Substance", "Enzyme", "Element, Ion, or Isotope",
                    "Nucleic Acid, Nucleoside, or Nucleotide", "Biomedical or Dental Material", "Substance", "Food"),
    "procedimento": ("Therapeutic or Preventive Procedure", "Health Care Activity", "Educational Activity",
                     "Research Activity", "Medical Device", "Drug Delivery Device", "Manufactured Object"),
    "exame": ("Diagnostic Procedure", "Laboratory Procedure", "Laboratory or Test Result", "Clinical Attribute",
              "Organism Attribute"),
    "anatomia": ("Body Location or Region", "Body Part, Organ, or Organ Component", "Body Space or Junction",
                 "Body System", "Body Substance", "Tissue", "Cell", "Spatial Concept"),
    "valor": ("Quantitative Concept", "Qualitative Concept"),
    "tempo": ("Temporal Concept",),
    "negacao": ("Negation",),
    "outro": ("Patient or Disabled Group", "Professional or Occupational Group", "Health Care Related Organization",
              "Population Group", "Family Group", "Age Group", "Group", "Organization", "Intellectual Product",
              "Functional Concept", "Idea or Concept", "Classification", "Regulation or Law", "Event", "Activity",
              "Daily or Recreational Activity", "Occupational Activity", "Machine Activity", "Individual Behavior",
              "Social Behavior", "Behavior", "Biomedical Occupation or Discipline", "Physical Object",
              "Phenomenon or Process", "Natural Phenomenon or Process", "Bacterium", "Virus", "Fungus", "Plant",
              "Fish", "Amino Acid Sequence"),
}
GRUPO_DO_TIPO = {tipo: grupo for grupo, tipos in TIPOS_DO_GRUPO.items() for tipo in tipos}
SEM_TIPO = "Abbreviation"

# Grupos de origem e de destino que fazem sentido em cada tipo de relação (None: qualquer par)
CLINICOS = {"doenca", "sinal_sintoma", "medicamento", "procedimento", "exame"}
REGRAS = {
    "localizado_em": ({"sinal_sintoma", "doenca", "procedimento", "exame"}, {"anatomia"}),
    "tem_dose": ({"medicamento", "procedimento"}, {"valor", "tempo"}),
    "tem_valor": ({"exame", "sinal_sintoma"}, {"valor"}),
    "tem_resultado": ({"exame", "procedimento"}, {"sinal_sintoma", "doenca", "valor"}),
    "trata": ({"medicamento", "procedimento"}, {"doenca", "sinal_sintoma"}),
    "tem_tempo": (CLINICOS, {"tempo"}),
    "nega": ({"negacao"}, CLINICOS),
    "associado_a": None,
}
SITUACOES = ("respeita", "direção trocada", "viola", "sem categoria")

PROMPT_CONCEITOS = """Você é um especialista em informática clínica. Leia a nota clínica abaixo, escrita em \
português por profissionais de saúde, e liste todos os conceitos clínicos que ela contém (inclusive valores, \
tempos e palavras de negação), cada um com a sua categoria.

Regras:
- Copie cada conceito exatamente como está escrito na nota.
- Cada conceito deve ser curto, de uma a quatro palavras (por exemplo: "dor", "MIE", "acesso venoso \
periférico"), e não uma frase inteira.
- Não invente conceitos que a nota não contém. Os exemplos abaixo só explicam as categorias; não os copie.
- Não repita conceitos. Liste no máximo {maximo} conceitos.
- O que o profissional encontrou ao examinar o paciente é sinal_sintoma; exame é o exame ou a medida em si.
- Use uma destas categorias:
{categorias}

Nota clínica:
\"\"\"{texto}\"\"\"

Responda em JSON, no formato {{"conceitos": [{{"texto": "...", "categoria": "..."}}]}}."""

REGRA_LISTA_GRUPOS = """- Use como origem e destino apenas os conceitos desta lista, encontrados na nota (a \
categoria entre parênteses é só uma ajuda; não a copie):
{conceitos}
"""

ESQUEMA_CONCEITOS = {
    "type": "object",
    "properties": {"conceitos": {"type": "array", "maxItems": MAX_CONCEITOS, "items": {
        "type": "object",
        "properties": {"texto": {"type": "string"}, "categoria": {"type": "string", "enum": list(GRUPOS)}},
        "required": ["texto", "categoria"],
    }}},
    "required": ["conceitos"],
}


def grupos(tag):
    """Grupos dos tipos de uma anotação (vazio quando ela só está marcada como abreviatura)."""
    tipos = [t.strip() for t in (tag or "").split("|") if t.strip() and t.strip() != SEM_TIPO]
    return {GRUPO_DO_TIPO.get(t, "outro") for t in tipos}


def gabarito(nota):
    """Grupos de cada conceito anotado (texto normalizado), juntando as anotações do mesmo texto."""
    certo = collections.defaultdict(set)
    for a in nota["anotacoes"]:
        texto = piloto.normalizar(a["texto"])
        if texto:
            certo[texto] |= grupos(a["categoria"])
    return certo


def lista_anotada(nota):
    """Conceitos dos anotadores com os grupos do gabarito, na mesma ordem e sem as repetições da forma guiada."""
    certo, vistos, lista = gabarito(nota), set(), []
    for a in nota["anotacoes"]:
        texto = piloto.normalizar(a["texto"])
        if texto and texto not in vistos:
            vistos.add(texto)
            lista.append((a["texto"], certo[texto]))
    return lista


def prompt_conceitos(nota):
    categorias = "\n".join(f"  - {nome}: {descricao}" for nome, descricao in GRUPOS.items())
    return PROMPT_CONCEITOS.format(maximo=MAX_CONCEITOS, categorias=categorias, texto=nota["texto"])


def prompt_relacoes(nota, lista):
    """O pedido da forma guiada, com a lista dada e as categorias agrupadas."""
    linhas = [f"  - {texto} ({'/'.join(sorted(g)) or 'abreviatura'})" for texto, g in lista]
    tipos = "\n".join(f"  - {nome}: {descricao}" for nome, descricao in piloto.TIPOS.items())
    return piloto.PROMPT.format(regra_lista=REGRA_LISTA_GRUPOS.format(conceitos="\n".join(linhas)), tipos=tipos,
                                texto=nota["texto"], maximo=piloto.MAX_RELACOES)


def extrair_conceitos(nota, host):
    """Etapa 1: conceitos listados (sem repetições), a resposta crua, se ela veio cortada e os tokens gerados."""
    crua, tokens = piloto.perguntar(prompt_conceitos(nota), MODELO, host, ESQUEMA_CONCEITOS, maximo_tokens=3000)
    try:
        conceitos, cortada = json.loads(crua).get("conceitos", []), False
    except json.JSONDecodeError:
        # resposta cortada no limite de tamanho: aproveita os conceitos que chegaram completos
        padrao = r'\{\s*"texto"\s*:\s*"([^"]*)"\s*,\s*"categoria"\s*:\s*"([^"]*)"\s*\}'
        conceitos = [dict(zip(("texto", "categoria"), m)) for m in re.findall(padrao, crua)]
        cortada = True
    limpos, vistos = [], set()
    for c in conceitos:
        texto, categoria = str(c.get("texto", "")).strip(), str(c.get("categoria", "")).strip()
        if piloto.normalizar(texto) and categoria in GRUPOS and piloto.normalizar(texto) not in vistos:
            vistos.add(piloto.normalizar(texto))
            limpos.append({"texto": texto, "categoria": categoria})
    return limpos, crua, cortada, tokens


def avaliar_conceitos(nota, conceitos):
    """Etapa 1: conceitos listados x conceitos anotados, e a categoria de cada um."""
    texto = f" {piloto.normalizar(nota['texto'])} "
    certo = gabarito(nota)
    listados = {piloto.normalizar(c["texto"]): c["categoria"] for c in conceitos}
    resultado = {"conceitos_listados": len(listados), "conceitos_no_texto": sum(f" {c} " in texto for c in listados),
                 "anotados_distintos": len(certo), "categorias_avaliaveis": 0, "categorias_certas": 0}
    for modo in ("exata", "parcial"):
        resultado[f"anotados_na_lista_{modo}"] = sum(any(piloto.casa(a, c, modo == "parcial") for c in listados)
                                                     for a in certo)
    erros = collections.Counter()
    for c, categoria in listados.items():
        casados = [a for a in certo if a == c] or [a for a in certo if piloto.casa(a, c, True)]
        esperados = set().union(*(certo[a] for a in casados))
        if esperados:
            resultado["categorias_avaliaveis"] += 1
            if categoria in esperados:
                resultado["categorias_certas"] += 1
            else:
                erros[f"{'/'.join(sorted(esperados))} -> {categoria}"] += 1
    resultado["erros_de_categoria"] = dict(erros)
    return resultado


def grupos_da_ponta(texto, categorias):
    """Grupos de uma ponta da relação: os do conceito da lista com o mesmo texto ou, se não houver, de um que case."""
    texto = piloto.normalizar(texto)
    if texto in categorias:
        return categorias[texto]
    return next((g for c, g in categorias.items() if piloto.casa(c, texto, True)), set())


def situacao(relacao, categorias):
    origem, destino = grupos_da_ponta(relacao["origem"], categorias), grupos_da_ponta(relacao["destino"], categorias)
    if not origem or not destino:
        return "sem categoria"
    regra = REGRAS[relacao["relacao"]]
    if regra is None:
        return "respeita"
    origens, destinos = regra
    if origem & origens and destino & destinos:
        return "respeita"
    if destino & origens and origem & destinos:
        return "direção trocada"
    return "viola"


def avaliar_regras(relacoes, novas, categorias):
    """Situação de cada relação diante das regras, separando as que batem com as anotações das novas,
    e quais combinações de grupos e tipo de relação violam as regras."""
    contagem = {"batem": collections.Counter(), "novas": collections.Counter()}
    violacoes = {"batem": collections.Counter(), "novas": collections.Counter()}
    for r in relacoes:
        grupo, s = "novas" if r in novas else "batem", situacao(r, categorias)
        contagem[grupo][s] += 1
        if s == "viola":
            origem = "/".join(sorted(grupos_da_ponta(r["origem"], categorias)))
            destino = "/".join(sorted(grupos_da_ponta(r["destino"], categorias)))
            violacoes[grupo][f"{origem} -{r['relacao']}-> {destino}"] += 1
    return {grupo: {"situacoes": dict(contagem[grupo]), "violacoes": dict(violacoes[grupo])} for grupo in contagem}


def resumir_conceitos(execucoes):
    soma, erros = collections.Counter(), collections.Counter()
    for e in execucoes:
        if e.get("avaliacao_conceitos"):
            soma.update({k: v for k, v in e["avaliacao_conceitos"].items() if isinstance(v, int)})
            erros.update(e["avaliacao_conceitos"]["erros_de_categoria"])
    return {"conceitos_listados_por_nota": piloto.fracao(soma["conceitos_listados"], len(execucoes)),
            "lista_no_texto": piloto.fracao(soma["conceitos_no_texto"], soma["conceitos_listados"]),
            "anotados_na_lista_exata": piloto.fracao(soma["anotados_na_lista_exata"], soma["anotados_distintos"]),
            "anotados_na_lista_parcial": piloto.fracao(soma["anotados_na_lista_parcial"], soma["anotados_distintos"]),
            "categoria_certa": piloto.fracao(soma["categorias_certas"], soma["categorias_avaliaveis"]),
            "categorias_avaliadas": soma["categorias_avaliaveis"],
            "listas_cortadas": sum(1 for e in execucoes if e.get("conceitos_cortada")),
            "erros_de_categoria_mais_comuns": dict(erros.most_common(8))}


def resumir_regras(execucoes):
    soma = {"batem": collections.Counter(), "novas": collections.Counter()}
    violacoes = {"batem": collections.Counter(), "novas": collections.Counter()}
    for e in execucoes:
        for grupo, r in e.get("regras", {}).items():
            soma[grupo].update(r["situacoes"])
            violacoes[grupo].update(r["violacoes"])
    return {grupo: {"relacoes": sum(c.values()), **{s: piloto.fracao(c[s], sum(c.values())) for s in SITUACOES},
                    "violacoes_mais_comuns": dict(violacoes[grupo].most_common(5))}
            for grupo, c in soma.items()}


def main():
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    argumentos.add_argument("--notas", type=int, default=20, help="notas sorteadas (as mesmas do experimento 5)")
    argumentos.add_argument("--host", default="http://127.0.0.1:11434", help="endereço do Ollama")
    args = argumentos.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    try:
        instalados = {m["name"] for m in piloto.chamar(f"{args.host}/api/tags", timeout=10).get("models", [])}
        versao = piloto.chamar(f"{args.host}/api/version", timeout=10).get("version")
    except (OSError, ValueError) as erro:
        sys.exit(f"Não foi possível falar com o Ollama em {args.host} ({erro}). Ele está rodando?")
    if MODELO not in instalados:
        sys.exit(f"O {MODELO} não está no Ollama. Rode: ollama pull {MODELO}")

    arquivos = sorted(glob.glob(os.path.join(piloto.PASTA_XML, "*.xml")))
    tipos_no_corpus = {t.strip() for arquivo in arquivos for a in piloto.ler_nota(arquivo)["anotacoes"]
                       for t in a["categoria"].split("|") if t.strip()}
    sem_grupo = sorted(tipos_no_corpus - set(GRUPO_DO_TIPO) - {SEM_TIPO})
    print(f"Tipos semânticos no corpus: {len(tipos_no_corpus)}; sem grupo definido (contam como outro): {sem_grupo}")

    sorteadas = sorted(random.Random(piloto.SEMENTE).sample(arquivos, min(args.notas, len(arquivos))))
    notas = [piloto.ler_nota(a) for a in sorteadas]
    por_grupo = collections.Counter(g for n in notas for gs in gabarito(n).values() for g in (gs or {"(só abreviatura)"}))
    print(f"{len(notas)} notas sorteadas (semente {piloto.SEMENTE}); conceitos anotados distintos por grupo:"
          f" {dict(por_grupo.most_common())}")

    # Carrega o modelo antes de medir o tempo, com o mesmo tamanho de contexto do piloto
    piloto.chamar(f"{args.host}/api/generate", {"model": MODELO, "prompt": "ok", "stream": False, "keep_alive": "30m",
                                                "options": {"num_ctx": 8192, "num_predict": 1}})
    carregado = next(m for m in piloto.chamar(f"{args.host}/api/ps", timeout=10)["models"] if m["name"] == MODELO)
    na_placa = carregado["size_vram"] / carregado["size"]
    print(f"Ollama {versao}; {MODELO} com {100 * na_placa:.0f}% na placa de vídeo")

    dados = {"protocolo": {"semente": piloto.SEMENTE, "modelo": MODELO, "ollama": versao, "fracao_na_placa": na_placa,
                           "formas": FORMAS, "grupos": GRUPOS, "tipos_do_grupo": TIPOS_DO_GRUPO,
                           "regras": {t: r and [sorted(r[0]), sorted(r[1])] for t, r in REGRAS.items()},
                           "max_conceitos": MAX_CONCEITOS, "prompt_conceitos": PROMPT_CONCEITOS,
                           "regra_lista_grupos": REGRA_LISTA_GRUPOS, "esquema_conceitos": ESQUEMA_CONCEITOS},
             "notas": [n["arquivo"] for n in notas], "execucoes": [], "resumo": {}}
    for forma in FORMAS:
        print(f"\n== {MODELO}, {forma}")
        for numero, nota in enumerate(notas, 1):
            comeco = time.time()
            registro, categorias = {"forma": forma, "nota": nota["arquivo"]}, None
            try:
                if forma in ("livre", "guiada"):
                    prompt = piloto.prompt_para(nota, forma)
                else:
                    if forma == "guiada com grupos":
                        lista = lista_anotada(nota)
                    else:
                        conceitos, crua, cortada, tokens = extrair_conceitos(nota, args.host)
                        registro.update(conceitos=conceitos, resposta_conceitos=crua, conceitos_cortada=cortada,
                                        tokens_conceitos=tokens, avaliacao_conceitos=avaliar_conceitos(nota, conceitos))
                        lista = [(c["texto"], {c["categoria"]}) for c in conceitos]
                    categorias = {piloto.normalizar(texto): g for texto, g in lista}
                    prompt = prompt_relacoes(nota, lista)
                extraidas, crua, cortada, tokens = piloto.extrair(prompt, MODELO, args.host)
            except (OSError, ValueError) as erro:  # sem conexão, erro do servidor ou resposta que não é JSON
                extraidas, crua, cortada, tokens, categorias = [], None, None, None, None
                registro["erro"] = repr(erro)
            registro.update(relacoes=extraidas, resposta=crua, cortada=cortada, tokens=tokens,
                            segundos=round(time.time() - comeco, 1),
                            avaliacao=piloto.avaliar(nota, extraidas, "guiada" if forma.startswith("guiada") else "livre"))
            if categorias is not None:
                registro["regras"] = avaliar_regras(extraidas, registro["avaliacao"]["novas"], categorias)
            dados["execucoes"].append(registro)
            with open(SAIDA_JSON, "w", encoding="utf-8") as f:
                json.dump(dados, f, ensure_ascii=False, indent=2)
            a, c = registro["avaliacao"], registro.get("avaliacao_conceitos")
            etapa1 = (f"{c['conceitos_listados']} conceitos ({c['categorias_certas']}/{c['categorias_avaliaveis']}"
                      f" categorias certas), " if c else "")
            situacao_texto = " FALHOU" if registro.get("erro") else " (resposta cortada)" if cortada else ""
            print(f"   [{numero}/{len(notas)}] {nota['arquivo']}: {etapa1}{a['relacoes_extraidas']} relações,"
                  f" {a['relacoes_anotadas_achadas_parcial']}/{a['relacoes_anotadas']} anotadas achadas"
                  f"{situacao_texto} ({registro['segundos']} s)")
        execucoes = [e for e in dados["execucoes"] if e["forma"] == forma]
        resumo = piloto.resumir(execucoes, "guiada" if forma.startswith("guiada") else "livre")
        if forma == "com categorias":
            resumo["etapa_1"] = resumir_conceitos(execucoes)
        if forma in ("guiada com grupos", "com categorias"):
            resumo["regras"] = resumir_regras(execucoes)
        dados["resumo"][forma] = resumo
        with open(SAIDA_JSON, "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

    pct = piloto.pct
    print("\n== Resumo (somando as notas)")
    print(f"   {'forma':20s}{'relações/nota':>14s}{'conceitos no texto':>20s}{'anotadas achadas':>18s}"
          f"{'(exata)':>9s}{'negações':>10s}{'novas':>8s}{'s/nota':>8s}")
    for forma, r in dados["resumo"].items():
        print(f"   {forma:20s}{r['relacoes_por_nota']:14.1f}{pct(r['conceitos_no_texto']):>20s}"
              f"{pct(r['relacoes_anotadas_achadas_parcial']):>18s}{pct(r['relacoes_anotadas_achadas_exata']):>9s}"
              f"{pct(r['negacoes_achadas_parcial']):>10s}{pct(r['relacoes_novas']):>8s}{r['segundos_por_nota']:8.0f}")
    for forma in ("livre", "com categorias"):
        r = dados["resumo"][forma]
        print(f"   {forma}: conceitos anotados achados nas relações {pct(r['conceitos_anotados_achados_parcial']).strip()}"
              f"; falhas {r['falhas']}; respostas cortadas {r['respostas_cortadas']}")
    e1 = dados["resumo"]["com categorias"]["etapa_1"]
    print(f"\n   com categorias, etapa 1: {e1['conceitos_listados_por_nota']:.1f} conceitos por nota;"
          f" {pct(e1['lista_no_texto']).strip()} no texto; {pct(e1['anotados_na_lista_parcial']).strip()} dos anotados"
          f" na lista ({pct(e1['anotados_na_lista_exata']).strip()} exata); categoria certa"
          f" {pct(e1['categoria_certa']).strip()} de {e1['categorias_avaliadas']} conceitos; listas cortadas"
          f" {e1['listas_cortadas']}")
    print(f"   erros de categoria mais comuns (gabarito -> modelo): {e1['erros_de_categoria_mais_comuns']}")
    print(f"\n   regras de categoria (fração das relações){''.join(f'{s:>17s}' for s in SITUACOES)}")
    for forma in ("guiada com grupos", "com categorias"):
        for grupo, r in dados["resumo"][forma]["regras"].items():
            nome = f"{forma}, {'batem com as anotações' if grupo == 'batem' else 'novas'} ({r['relacoes']})"
            print(f"   {nome:43s}{''.join(f'{pct(r[s]):>17s}' for s in SITUACOES)}")
    for grupo, r in dados["resumo"]["com categorias"]["regras"].items():
        print(f"   com categorias, violações mais comuns entre as que {'batem' if grupo == 'batem' else 'são novas'}:"
              f" {r['violacoes_mais_comuns']}")
    print(f"\nResultados em {os.path.relpath(SAIDA_JSON, RAIZ)}")


if __name__ == "__main__":
    main()
