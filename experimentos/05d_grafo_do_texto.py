"""
Experimento 5d: o grafo do texto das 998 notas.

O experimento 5c mostrou, nas 20 notas do piloto, que o caminho em duas etapas
funciona: um modelo treinado (o BioBERTpt) faz a lista de conceitos de cada
nota, com a categoria, e o modelo de linguagem (o qwen2.5:7b) liga esses
conceitos. Aqui o mesmo caminho é aplicado às 998 notas para montar o grafo.

Um modelo que marcasse as notas em que foi treinado só repetiria as
anotações. Por isso as notas são sorteadas em 5 partes e são treinados 5
modelos: cada um treina nas outras 4 partes (90% para treinar e 10% para
escolher a época, como no 5c) e marca a parte que não viu. Assim cada nota é
marcada como seria uma nota nova, que o modelo nunca viu.

Duas diferenças do 5c, vistas nas primeiras notas da parte 1:
  - O modelo às vezes marca só um pedaço de uma palavra ("ONGE" em "LONGE"),
    o que as anotações nunca fazem. Os trechos marcados são estendidos até
    as bordas das palavras antes de virar a lista de conceitos.
  - Cada pedido ao modelo de linguagem tem até 40 conceitos da lista: as
    notas com listas mais longas vão em pedaços de frases inteiras, cada um
    com o próprio limite de 30 relações; as outras vão inteiras, como no 5c.
    O limite começou em 55 conceitos, mas as notas da parte 1 ligadas
    inteiras com 41 a 55 conceitos chegaram muito mais vezes ao limite de 30
    relações que os pedaços, e foram ligadas de novo em pedaços.
O resultado de montar mostra a marcação com e sem o ajuste às palavras, e as
relações das notas inteiras, das notas em pedaços e das notas ligadas das
duas formas.

O script tem três partes:
  treinar   (WSL) treina os 5 modelos e salva os conceitos de cada parte
  ligar     (Windows) manda a lista de cada nota para o modelo de linguagem;
            espera cada parte ficar pronta, então pode rodar junto com o treino
  montar    (qualquer Python) junta as notas no grafo e mede o resultado
Se forem interrompidas, treinar e ligar continuam de onde pararam.

O grafo: cada entidade é um conceito marcado, com a chave "texto normalizado
|categoria" (como a do grafo das anotações, que usa "texto|tipo"), e cada
relação do modelo de linguagem vira uma tripla (origem, tipo, destino). As
pontas de cada relação são ligadas ao conceito da lista da nota com o mesmo
texto ou, se não houver, ao primeiro que case (a mesma escolha das regras de
categoria do experimento 5b); a ponta que não casa com nenhum fica com a
categoria "sem_categoria". Saem dois grafos:
  completo  todas as relações
  filtrado  só as que respeitam as regras de categoria do experimento 5b
            (as de direção trocada entram invertidas)
Para comparar, montar mede também o grafo das anotações, com a chave do
experimento 1 e com a mesma chave do grafo do texto. As medidas da marcação e
das relações são dadas sem as 20 notas do piloto, que serviram para ajustar
os pedidos ao modelo de linguagem, e só nelas, para comparar com o 5c.

Saídas (não versionadas):
  results/experimento5d/conceitos_parte<k>.json  conceitos marcados nas notas da parte k
  results/experimento5d/relacoes.jsonl           relações de cada nota, uma linha por nota
  results/experimento5d/grafo_completo.tsv       triplas (origem, tipo, destino), sem cabeçalho,
  results/experimento5d/grafo_filtrado.tsv       no formato que o PyKEEN lê
  results/experimento5d/entidades.json           texto, categoria e notas de cada entidade
  results/experimento5d_resultados.json          medidas
  ~/tcc_modelos/experimento5d_parte<k>           os 5 modelos (fora do repositório)

Uso:
    # no WSL, no ambiente conda do projeto
    conda activate tcc_graphs
    python experimentos/05d_grafo_do_texto.py treinar
    # no Windows, com o Ollama rodando e o qwen2.5:7b baixado
    python experimentos/05d_grafo_do_texto.py ligar
    # em qualquer Python, depois das duas
    python experimentos/05d_grafo_do_texto.py montar
"""
import argparse
import collections
import glob
import html
import importlib.util
import json
import os
import random
import re
import statistics
import sys
import time
import xml.etree.ElementTree as ET

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA = os.path.join(RAIZ, "results", "experimento5d")
SAIDA_RELACOES = os.path.join(PASTA, "relacoes.jsonl")
SAIDA_JSON = os.path.join(RAIZ, "results", "experimento5d_resultados.json")
PASTA_MODELOS = os.path.join(os.path.expanduser("~"), "tcc_modelos")
PARTES = 5
SEM_CATEGORIA = "sem_categoria"
# No piloto do 5c, o modelo de linguagem achou 52% das relações anotadas nas notas com menos de 55 conceitos e
# 28% nas outras, e as notas com mais conceitos chegam mais vezes ao limite de relações por resposta. Por isso
# a nota vai inteira só quando a lista cabe num pedido; se não, vai em pedaços de frases inteiras
CONCEITOS_POR_PEDIDO = 40


def carregar(nome, arquivo):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(RAIZ, "experimentos", arquivo))
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# Treino, etapa 2 e medidas: os mesmos do experimento 5c (e, por ele, do 5b e do 5)
ner = carregar("ner", "05c_piloto_com_ner.py")
categorias = ner.categorias
piloto = ner.piloto


def arquivo_conceitos(k):
    return os.path.join(PASTA, f"conceitos_parte{k}.json")


def partes():
    """As notas sorteadas em 5 partes de 199 ou 200 notas."""
    arquivos = sorted(glob.glob(os.path.join(piloto.PASTA_XML, "*.xml")))
    random.Random(ner.SEMENTE).shuffle(arquivos)
    return [sorted(arquivos[k::PARTES]) for k in range(PARTES)]


def salvar(dados, caminho):
    """Grava num arquivo temporário e só então troca, para a parte ligar nunca ler um arquivo pela metade."""
    with open(caminho + ".tmp", "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=1)
    os.replace(caminho + ".tmp", caminho)


def ajustar_as_palavras(texto, spans):
    """Os trechos marcados, estendidos até as bordas das palavras. O modelo às vezes marca só um pedaço de uma
    palavra (por exemplo, "ONGE" em "LONGE" ou "pre" e "scrição medica" em "prescrição medica"), o que as
    anotações nunca fazem. Trechos que passam a se sobrepor viram um só, com a categoria do mais longo."""
    ajustados = []
    for inicio, fim, categoria in sorted(spans):
        while inicio > 0 and texto[inicio - 1].isalnum() and texto[inicio].isalnum():
            inicio -= 1
        while fim < len(texto) and texto[fim].isalnum() and texto[fim - 1].isalnum():
            fim += 1
        if ajustados and inicio < ajustados[-1][1]:
            inicio_antes, fim_antes, categoria_antes = ajustados[-1]
            if fim - inicio > fim_antes - inicio_antes:
                categoria_antes = categoria
            ajustados[-1] = (inicio_antes, max(fim, fim_antes), categoria_antes)
        else:
            ajustados.append((inicio, fim, categoria))
    return ajustados


def conceitos_da_nota(item):
    """Texto cru, trechos ajustados às palavras e lista de conceitos de uma nota marcada."""
    texto = ner.texto_cru(os.path.join(piloto.PASTA_XML, item["arquivo"]))
    spans = ajustar_as_palavras(texto, [tuple(s) for s in item["spans"]])
    return texto, spans, ner.lista_de_conceitos(texto, spans)


def treinar():
    import torch
    import transformers

    transformers.logging.set_verbosity_error()
    tokenizador = transformers.AutoTokenizer.from_pretrained(ner.MODELO_BASE)
    os.makedirs(PASTA, exist_ok=True)
    divisao = partes()
    notas = {a: ner.preparar(tokenizador, a) for parte in divisao for a in parte}
    contagem = {k: sum(n["contagem"][k] for n in notas.values()) for k in ("no_lugar", "achadas_perto", "fora")}
    print(f"{len(notas)} notas em {PARTES} partes de {', '.join(str(len(p)) for p in divisao)}")
    print(f"Anotações: {contagem['no_lugar']} no lugar marcado, {contagem['achadas_perto']} achadas perto,"
          f" {contagem['fora']} de fora")

    for k, arquivos_marcados in enumerate(divisao, 1):
        if os.path.exists(arquivo_conceitos(k)):
            print(f"\n== Parte {k}: já treinada")
            continue
        resto = sorted(a for j, parte in enumerate(divisao, 1) if j != k for a in parte)
        arquivos_validacao = random.Random(ner.SEMENTE).sample(resto, round(ner.FRACAO_VALIDACAO * len(resto)))
        validacao = [notas[a] for a in sorted(arquivos_validacao)]
        treino = [notas[a] for a in resto if a not in arquivos_validacao]
        marcadas = [notas[a] for a in arquivos_marcados]
        print(f"\n== Parte {k}: {len(treino)} notas de treino, {len(validacao)} de validação, {len(marcadas)} marcadas")
        comeco, pasta = time.time(), os.path.join(PASTA_MODELOS, f"experimento5d_parte{k}")
        historico = ner.treinar_modelo(tokenizador, treino, validacao, pasta)
        modelo = transformers.AutoModelForTokenClassification.from_pretrained(pasta)
        _, medidas_validacao = ner.avaliar_notas(modelo, tokenizador, validacao)
        previstos, medidas_marcadas = ner.avaliar_notas(modelo, tokenizador, marcadas)
        dados = {"parte": k,
                 "protocolo": {"modelo_base": ner.MODELO_BASE, "semente": ner.SEMENTE, "epocas_maximas": ner.EPOCAS,
                               "paciencia": ner.PACIENCIA, "lote": ner.LOTE, "taxa": ner.TAXA,
                               "tokens_por_janela": ner.TOKENS_POR_JANELA, "sobreposicao": ner.SOBREPOSICAO,
                               "torch": torch.__version__, "transformers": transformers.__version__},
                 "notas_treino": [n["arquivo"] for n in treino], "notas_validacao": [n["arquivo"] for n in validacao],
                 "historico": historico, "minutos": round((time.time() - comeco) / 60, 1),
                 "validacao": medidas_validacao, "marcadas": medidas_marcadas,
                 "notas": [{"arquivo": n["arquivo"], "spans": previstos[n["arquivo"]],
                            "conceitos": ner.lista_de_conceitos(n["texto"], previstos[n["arquivo"]])}
                           for n in marcadas]}
        salvar(dados, arquivo_conceitos(k))
        pct = piloto.pct
        for nome, m in (("validação", medidas_validacao), (f"parte {k} (marcada)", medidas_marcadas)):
            print(f"   {nome:20s} precisão {pct(m['posicao_e_categoria']['precisao']).strip()},"
                  f" revocação {pct(m['posicao_e_categoria']['revocacao']).strip()},"
                  f" F1 {pct(m['posicao_e_categoria']['f1']).strip()}"
                  f" (só posição: F1 {pct(m['so_posicao']['f1']).strip()})")
        print(f"   {dados['minutos']} min; conceitos em {os.path.relpath(arquivo_conceitos(k), RAIZ)}")


def todos_os_registros():
    """Os registros das notas ligadas, na ordem em que foram gravados. Uma linha cortada (se o computador
    desligou no meio da gravação) é ignorada, e a nota é ligada de novo."""
    registros = []
    if os.path.exists(SAIDA_RELACOES):
        with open(SAIDA_RELACOES, encoding="utf-8") as f:
            for linha in f:
                try:
                    registros.append(json.loads(linha))
                except json.JSONDecodeError:
                    continue
    return registros


def registros_de_relacoes():
    """O último registro de cada nota já ligada."""
    return {r["nota"]: r for r in todos_os_registros()}


def em_dia(registro, item):
    """Se a nota marcada (item) foi ligada sem erro e do jeito de agora. As notas com mais de CONCEITOS_POR_PEDIDO
    conceitos ligadas inteiras, de quando o limite era maior, são ligadas de novo em pedaços (a não ser a nota
    que não dá para cortar em frases, que vai inteira mesmo)."""
    if registro.get("erro"):
        return False
    if registro["pedidos"] > 1 or len(registro["conceitos"]) <= CONCEITOS_POR_PEDIDO:
        return True
    texto, spans, conceitos = conceitos_da_nota(item)
    return len(pedacos(texto, mencoes_da_lista(texto, spans, conceitos))) == 1


def ligar(host):
    try:
        versao = piloto.chamar(f"{host}/api/version", timeout=10).get("version")
    except (OSError, ValueError) as erro:
        sys.exit(f"Não foi possível falar com o Ollama em {host} ({erro}). Ele está rodando?")
    modelo, registros = categorias.MODELO, registros_de_relacoes()
    print(f"Ollama {versao}; etapa 2 com o {modelo}; {sum(not r.get('erro') for r in registros.values())} notas"
          f" já ligadas antes")
    for k in range(1, PARTES + 1):
        if not os.path.exists(arquivo_conceitos(k)):
            print(f"\n   esperando o treino da parte {k}...")
            while not os.path.exists(arquivo_conceitos(k)):
                time.sleep(60)
        with open(arquivo_conceitos(k), encoding="utf-8") as f:
            parte = json.load(f)
        pendentes = [item for item in parte["notas"]
                     if item["arquivo"] not in registros or not em_dia(registros[item["arquivo"]], item)]
        print(f"\n== Parte {k}: {len(pendentes)} de {len(parte['notas'])} notas para ligar")
        if not pendentes:
            continue
        # Carrega o modelo antes de medir o tempo, com o mesmo tamanho de contexto do piloto
        piloto.chamar(f"{host}/api/generate", {"model": modelo, "prompt": "ok", "stream": False, "keep_alive": "30m",
                                               "options": {"num_ctx": 8192, "num_predict": 1}})
        for numero, item in enumerate(pendentes, 1):
            nota = piloto.ler_nota(os.path.join(piloto.PASTA_XML, item["arquivo"]))
            prompts, conceitos = pedidos(item, nota)
            comeco, registro = time.time(), {"nota": item["arquivo"], "parte": k, "conceitos": conceitos}
            extraidas, vistas, respostas, cortada, tokens = [], set(), [], False, 0
            try:
                for prompt in prompts:
                    novas, crua, cortada_aqui, tokens_aqui = piloto.extrair(prompt, modelo, host)
                    respostas.append(crua)
                    cortada, tokens = cortada or cortada_aqui, tokens + (tokens_aqui or 0)
                    for r in novas:  # sem repetir as relações que outro pedaço da mesma nota já deu
                        chave = (piloto.normalizar(r["origem"]), r["relacao"], piloto.normalizar(r["destino"]))
                        if chave not in vistas:
                            vistas.add(chave)
                            extraidas.append(r)
            except (OSError, ValueError) as erro:  # sem conexão, erro do servidor ou resposta que não é JSON
                extraidas, cortada, tokens = [], None, None
                registro["erro"] = repr(erro)
            avaliacao = piloto.avaliar(nota, extraidas, "livre")
            grupos = {piloto.normalizar(c["texto"]): grupos_do_conceito(c) for c in conceitos}
            registro.update(pedidos=len(prompts), relacoes=extraidas, respostas=respostas, cortada=cortada, tokens=tokens,
                            segundos=round(time.time() - comeco, 1), avaliacao=avaliacao,
                            regras=categorias.avaliar_regras(extraidas, avaliacao["novas"], grupos))
            with open(SAIDA_RELACOES, "a", encoding="utf-8") as f:
                f.write(json.dumps(registro, ensure_ascii=False) + "\n")
            print(f"   [{numero}/{len(pendentes)}] {item['arquivo']}: {len(conceitos)} conceitos"
                  + (f" em {len(prompts)} pedaços" if len(prompts) > 1 else "")
                  + f", {avaliacao['relacoes_extraidas']} relações, {avaliacao['relacoes_anotadas_achadas_parcial']}/"
                  f"{avaliacao['relacoes_anotadas']} anotadas achadas ({registro['segundos']} s)"
                  + (f" ERRO: {registro['erro']}" if registro.get("erro") else ""))
    print(f"\nRelações em {os.path.relpath(SAIDA_RELACOES, RAIZ)}")


def grupos_do_conceito(c):
    """Os grupos de um conceito da lista para as regras de categoria (nenhum para as abreviaturas sem tipo)."""
    return set() if c["categoria"] == "abreviatura" else {c["categoria"]}


def pedacos(texto, mencoes):
    """Pedaços (início, fim) de frases inteiras da nota, com até CONCEITOS_POR_PEDIDO conceitos distintos cada;
    uma frase com mais conceitos que isso é cortada nas vírgulas. mencoes: (início, conceito) de cada menção."""
    def conta(a, b):
        return len({c for p, c in mencoes if a <= p < b})

    def unidades(a, b, separador):
        cortes = [a] + [a + x.end() for x in re.finditer(separador, texto[a:b])] + [b]
        return [(x, y) for x, y in zip(cortes, cortes[1:]) if texto[x:y].strip()]

    frases = []
    for a, b in unidades(0, len(texto), r"[.;!?\n]\s+"):
        frases += unidades(a, b, r",\s+") if conta(a, b) > CONCEITOS_POR_PEDIDO else [(a, b)]
    saida = []
    for a, b in frases:
        if saida and conta(saida[-1][0], b) <= CONCEITOS_POR_PEDIDO:
            saida[-1] = (saida[-1][0], b)
        else:
            saida.append((a, b))
    return saida


def mencoes_da_lista(texto, spans, conceitos):
    """(início, conceito) de cada trecho marcado na nota, com o índice do conceito na lista."""
    indice = {piloto.normalizar(c["texto"]): i for i, c in enumerate(conceitos)}
    mencoes = []
    for a, b, _ in spans:
        t = piloto.normalizar(" ".join(texto[a:b].split()))
        if t in indice:
            mencoes.append((a, indice[t]))
    return mencoes


def pedidos(item, nota):
    """Os pedidos ao modelo de linguagem para uma nota marcada e a lista de conceitos dela. Com até
    CONCEITOS_POR_PEDIDO conceitos, é um pedido só, com a nota inteira, como no 5c; com mais, um pedido por
    pedaço de frases inteiras, cada um com os conceitos marcados nele."""
    texto, spans, conceitos = conceitos_da_nota(item)
    if len(conceitos) <= CONCEITOS_POR_PEDIDO:
        return [categorias.prompt_relacoes(nota, [(c["texto"], grupos_do_conceito(c)) for c in conceitos])], conceitos
    mencoes = mencoes_da_lista(texto, spans, conceitos)
    prompts = []
    for a, b in pedacos(texto, mencoes):
        no_pedaco = sorted({c for p, c in mencoes if a <= p < b})
        lista = [(conceitos[i]["texto"], grupos_do_conceito(conceitos[i])) for i in no_pedaco]
        prompts.append(categorias.prompt_relacoes({"texto": texto[a:b].replace("\r\n", "\n").strip()}, lista))
    return prompts, conceitos


def ponta(texto, conceitos):
    """Chave da entidade de uma ponta da relação: o conceito da lista com o mesmo texto ou, se não houver, o
    primeiro que case (a mesma escolha de grupos_da_ponta no experimento 5b); sem nenhum, o próprio texto."""
    t = piloto.normalizar(texto)
    if t not in conceitos:
        t = next((c for c in conceitos if piloto.casa(c, t, True)), t)
    return f"{t}|{conceitos.get(t, SEM_CATEGORIA)}"


def triplas_da_nota(relacoes, conceitos):
    """Triplas de uma nota no grafo completo e no filtrado, e o texto das pontas sem conceito da lista."""
    grupos = {t: set() if c == "abreviatura" else {c} for t, c in conceitos.items()}
    completo, filtrado, soltas = set(), set(), {}
    for r in relacoes:
        origem, destino = ponta(r["origem"], conceitos), ponta(r["destino"], conceitos)
        for chave, texto in ((origem, r["origem"]), (destino, r["destino"])):
            if chave.endswith(f"|{SEM_CATEGORIA}"):
                soltas[chave] = texto
        if origem == destino:
            continue
        completo.add((origem, r["relacao"], destino))
        situacao = categorias.situacao(r, grupos)
        if situacao == "respeita":
            filtrado.add((origem, r["relacao"], destino))
        elif situacao == "direção trocada":
            filtrado.add((destino, r["relacao"], origem))
    return completo, filtrado, soltas


def chave_experimento1(texto, tag):
    """A chave de entidade do grafo das anotações nos experimentos 1 a 4."""
    return " ".join((texto or "").strip().upper().split()) + "|" + str(tag)


def chave_do_texto(texto, tag):
    """A chave de entidade do grafo do texto, aplicada a uma anotação (nenhuma se o texto normalizado é vazio)."""
    t = piloto.normalizar(html.unescape(texto or ""))
    return f"{t}|{ner.rotulo(tag)}" if t else None


def grafo_das_anotacoes(arquivos, chave):
    """Triplas das anotações, com a chave de entidade dada, e as notas em que cada uma aparece."""
    triplas = collections.defaultdict(set)
    for arquivo in arquivos:
        raiz = ET.parse(arquivo).getroot()
        local = {a.get("id"): chave(a.get("text"), a.get("tag")) for a in raiz.findall(".//annotation")}
        for rel in raiz.findall(".//rel"):
            origem, destino = local.get(rel.get("annotation1")), local.get(rel.get("annotation2"))
            if origem and destino:
                triplas[(origem, rel.get("reltype"), destino)].add(os.path.basename(arquivo))
    return triplas


def vizinhos_de(triplas):
    """Entidades distintas ligadas a cada entidade, em qualquer direção e por qualquer relação."""
    vizinhos = collections.defaultdict(set)
    for origem, _, destino in triplas:
        vizinhos[origem].add(destino)
        vizinhos[destino].add(origem)
    return {e: v - {e} for e, v in vizinhos.items()}  # uma tripla de uma entidade com ela mesma não é vizinho


def estatisticas(triplas):
    """Tamanho e ligações de um grafo dado como {tripla: notas em que ela aparece}."""
    vizinhos = vizinhos_de(triplas)
    pai = {e: e for e in vizinhos}

    def raiz(e):
        while pai[e] != e:
            pai[e] = pai[pai[e]]
            e = pai[e]
        return e

    for origem, _, destino in triplas:
        pai[raiz(origem)] = raiz(destino)
    componentes = collections.Counter(raiz(e) for e in vizinhos)
    graus = [len(v) for v in vizinhos.values()]
    return {"entidades": len(vizinhos), "triplas": len(triplas),
            "tipos": dict(collections.Counter(r for _, r, _ in triplas).most_common()),
            "vizinhos_media": statistics.fmean(graus), "vizinhos_mediana": statistics.median(graus),
            "com_um_vizinho": piloto.fracao(sum(g == 1 for g in graus), len(graus)),
            "maior_componente": piloto.fracao(max(componentes.values()), len(vizinhos)),
            "triplas_em_duas_notas_ou_mais": piloto.fracao(sum(len(n) >= 2 for n in triplas.values()), len(triplas))}


def respostas_no_limite(respostas):
    """Quantas respostas do modelo de linguagem chegaram ao limite de relações por pedido."""
    contagem = 0
    for crua in respostas:
        try:
            contagem += len(json.loads(crua).get("relacoes", [])) >= piloto.MAX_RELACOES
        except (TypeError, ValueError, AttributeError):  # sem resposta, cortada ou que não é um objeto JSON
            pass
    return contagem


def resumir_etapa_2(execucoes):
    """As medidas da etapa 2 num grupo de notas ligadas."""
    respostas = [crua for e in execucoes for crua in e["respostas"]]
    return {**piloto.resumir(execucoes, "livre"), "regras": categorias.resumir_regras(execucoes),
            "pedidos_por_nota": piloto.fracao(len(respostas), len(execucoes)),
            "pedidos_no_limite": piloto.fracao(respostas_no_limite(respostas), len(respostas))}


def montar():
    partes_lidas = []
    for k in range(1, PARTES + 1):
        if os.path.exists(arquivo_conceitos(k)):
            with open(arquivo_conceitos(k), encoding="utf-8") as f:
                partes_lidas.append(json.load(f))
    if not partes_lidas:
        sys.exit("Nenhuma parte treinada ainda: rode antes a parte treinar.")
    marcadas = {n["arquivo"]: n for p in partes_lidas for n in p["notas"]}
    registros = registros_de_relacoes()
    ligadas_sem_erro = {a: r for a, r in registros.items() if a in marcadas and em_dia(r, marcadas[a])}
    do_piloto = {os.path.basename(a) for a in ner.divisao()[2]}
    if len(marcadas) < sum(len(p) for p in partes()) or len(ligadas_sem_erro) < len(marcadas):
        print(f"Atenção: resultado parcial. {len(marcadas)} notas marcadas, {len(ligadas_sem_erro)} ligadas sem erro")

    # Medidas por nota: a marcação de conceitos x anotações (com os trechos do modelo e com os ajustados às
    # palavras), a lista que vai para o modelo de linguagem (etapa 1) e as relações (etapa 2)
    brutos, ajustados, gabarito, listas, avaliacoes_listas = {}, {}, {}, {}, {}
    for arquivo, n in marcadas.items():
        caminho = os.path.join(piloto.PASTA_XML, arquivo)
        texto, ajustados[arquivo], listas[arquivo] = conceitos_da_nota(n)
        brutos[arquivo] = [tuple(s) for s in n["spans"]]
        gabarito[arquivo] = ner.anotacoes_no_texto(caminho, texto)[0]
        avaliacoes_listas[arquivo] = {"avaliacao_conceitos": categorias.avaliar_conceitos(piloto.ler_nota(caminho),
                                                                                         listas[arquivo])}
    sem_o_piloto = sorted(set(marcadas) - do_piloto)
    grupos_de_notas = {"sem o piloto": sem_o_piloto, "piloto": sorted(set(marcadas) & do_piloto),
                       "sem o piloto, inteiras": [a for a in sem_o_piloto if len(listas[a]) <= CONCEITOS_POR_PEDIDO],
                       "sem o piloto, em pedaços": [a for a in sem_o_piloto if len(listas[a]) > CONCEITOS_POR_PEDIDO]}
    grupos_de_notas.update({f"parte {p['parte']}": sorted(n["arquivo"] for n in p["notas"]) for p in partes_lidas})
    marcacao, etapa_1, etapa_2 = {}, {}, {}
    for nome, notas in grupos_de_notas.items():
        marcacao[nome] = {forma: ner.medidas({a: previstos[a] for a in notas}, {a: gabarito[a] for a in notas})
                          for forma, previstos in (("trechos do modelo", brutos), ("ajustados às palavras", ajustados))}
        etapa_1[nome] = {"notas": len(notas), **categorias.resumir_conceitos([avaliacoes_listas[a] for a in notas])}
        etapa_2[nome] = resumir_etapa_2([ligadas_sem_erro[a] for a in notas if a in ligadas_sem_erro])
    # As notas ligadas inteiras quando o limite de conceitos por pedido era maior e de novo em pedaços, com a
    # mesma lista: as duas formas nas mesmas notas
    antes = {r["nota"]: r for r in todos_os_registros()
             if r["nota"] in marcadas and not r.get("erro") and not em_dia(r, marcadas[r["nota"]])}
    religadas = [a for a in sem_o_piloto if a in antes and a in ligadas_sem_erro]
    if religadas:
        etapa_2["mesmas notas, inteiras"] = resumir_etapa_2([antes[a] for a in religadas])
        etapa_2["mesmas notas, em pedaços"] = resumir_etapa_2([ligadas_sem_erro[a] for a in religadas])

    # O grafo: as triplas de cada nota, juntas, e as entidades com o texto mais comum e as notas em que aparecem
    completo, filtrado = collections.defaultdict(set), collections.defaultdict(set)
    formas, notas_da_entidade = collections.defaultdict(collections.Counter), collections.defaultdict(set)
    for arquivo in sorted(marcadas):
        conceitos = {piloto.normalizar(c["texto"]): c["categoria"] for c in listas[arquivo]}
        for c in listas[arquivo]:
            chave = f"{piloto.normalizar(c['texto'])}|{c['categoria']}"
            formas[chave][c["texto"]] += 1
            notas_da_entidade[chave].add(arquivo)
        if arquivo not in ligadas_sem_erro:
            continue
        triplas_c, triplas_f, soltas = triplas_da_nota(ligadas_sem_erro[arquivo]["relacoes"], conceitos)
        for t in triplas_c:
            completo[t].add(arquivo)
        for t in triplas_f:
            filtrado[t].add(arquivo)
        for chave, texto in soltas.items():
            formas[chave][texto] += 1
            notas_da_entidade[chave].add(arquivo)
    os.makedirs(PASTA, exist_ok=True)
    for nome, triplas in (("completo", completo), ("filtrado", filtrado)):
        with open(os.path.join(PASTA, f"grafo_{nome}.tsv"), "w", encoding="utf-8", newline="\n") as f:
            f.writelines(f"{origem}\t{tipo}\t{destino}\n" for origem, tipo, destino in sorted(triplas))
    vizinhos = {"completo": vizinhos_de(completo), "filtrado": vizinhos_de(filtrado)}
    entidades = {chave: {"texto": formas[chave].most_common(1)[0][0], "categoria": chave.rsplit("|", 1)[1],
                         "ocorrencias": len(notas), "notas": sorted(notas),
                         "vizinhos_completo": len(vizinhos["completo"].get(chave, ())),
                         "vizinhos_filtrado": len(vizinhos["filtrado"].get(chave, ()))}
                 for chave, notas in sorted(notas_da_entidade.items())}
    salvar(entidades, os.path.join(PASTA, "entidades.json"))

    arquivos_ligados = [os.path.join(piloto.PASTA_XML, a) for a in sorted(ligadas_sem_erro)]
    grafos = {"do texto, completo": estatisticas(completo), "do texto, filtrado": estatisticas(filtrado),
              "das anotações (chave do exp. 1)": estatisticas(grafo_das_anotacoes(arquivos_ligados, chave_experimento1)),
              "das anotações (mesma chave)": estatisticas(grafo_das_anotacoes(arquivos_ligados, chave_do_texto))}
    categorias_das_entidades = {
        "todas as entidades": dict(collections.Counter(e["categoria"] for e in entidades.values()).most_common()),
        **{f"no grafo {nome}": dict(collections.Counter(e.rsplit("|", 1)[1] for e in vizinhos[nome]).most_common())
           for nome in vizinhos}}
    referencia_5c = None
    if os.path.exists(ner.SAIDA_JSON):
        with open(ner.SAIDA_JSON, encoding="utf-8") as f:
            d5c = json.load(f)
        referencia_5c = {"marcacao": d5c["teste"]["medidas"], "etapa_1": d5c["teste"]["etapa_1"],
                         "etapa_2": d5c.get("ligar", {}).get("resumo")}
        if referencia_5c["etapa_2"]:
            respostas_5c = [e["resposta"] for e in d5c["ligar"]["execucoes"]]
            referencia_5c["etapa_2"] = {**referencia_5c["etapa_2"], "pedidos_por_nota": 1.0,
                                        "pedidos_no_limite": piloto.fracao(respostas_no_limite(respostas_5c),
                                                                           len(respostas_5c))}
    dados = {"notas": {"marcadas": len(marcadas), "ligadas_sem_erro": len(ligadas_sem_erro),
                       "com_erro": sorted(a for a, r in registros.items() if r.get("erro")),
                       "do_piloto": sorted(do_piloto)},
             "partes": [{"parte": p["parte"], "notas_treino": len(p["notas_treino"]),
                         "notas_validacao": len(p["notas_validacao"]), "notas_marcadas": len(p["notas"]),
                         "epocas": len(p["historico"]),
                         "epoca_escolhida": max(p["historico"],
                                                key=lambda h: h["validacao"]["posicao_e_categoria"]["f1"])["epoca"],
                         "minutos": p["minutos"], "validacao": p["validacao"], "marcadas": p["marcadas"]}
                        for p in partes_lidas],
             "marcacao": marcacao, "etapa_1": etapa_1, "etapa_2": etapa_2, "grafos": grafos,
             "categorias_das_entidades": categorias_das_entidades, "referencia_5c": referencia_5c}
    salvar(dados, SAIDA_JSON)

    pct, principais = piloto.pct, ("sem o piloto", "piloto")
    print("\n== Marcação de conceitos: trechos com a mesma posição e categoria das anotações")
    print(f"   {'':26s}{'treino':>7s}{'validação':>10s}{'marcadas':>9s}{'época':>6s}{'min':>5s}{'F1':>7s}"
          f"{'| ajustados às palavras:':>25s}{'precisão':>9s}{'revocação':>10s}{'F1':>7s}{'(só posição)':>13s}")
    linhas = [(f"parte {p['parte']}", p["notas_treino"], p["notas_validacao"], p["notas_marcadas"],
               p["epoca_escolhida"], round(p["minutos"]), marcacao[f"parte {p['parte']}"]) for p in dados["partes"]]
    linhas += [(f"{nome} ({len(grupos_de_notas[nome])} notas)", "", "", "", "", "", marcacao[nome]) for nome in principais]
    if referencia_5c:
        validacao_5c, teste_5c = len(d5c["protocolo"]["notas_validacao"]), len(d5c["teste"]["notas"])
        epoca_5c = max(d5c["historico"], key=lambda h: h["validacao"]["posicao_e_categoria"]["f1"])["epoca"]
        linhas.append((f"piloto no 5c ({teste_5c} notas)", sum(len(p) for p in partes()) - validacao_5c - teste_5c,
                       validacao_5c, teste_5c, epoca_5c, round(d5c["minutos_de_treino"]),
                       {"trechos do modelo": referencia_5c["marcacao"]}))
    for rotulo_linha, treino, validacao, n_marcadas, epoca, minutos, m in linhas:
        modelo, ajustado = m["trechos do modelo"], m.get("ajustados às palavras")
        print(f"   {rotulo_linha:26s}{treino:>7}{validacao:>10}{n_marcadas:>9}{epoca:>6}{minutos:>5}"
              f"{pct(modelo['posicao_e_categoria']['f1']):>7s}{'':25s}"
              + (f"{pct(ajustado['posicao_e_categoria']['precisao']):>9s}{pct(ajustado['posicao_e_categoria']['revocacao']):>10s}"
                 f"{pct(ajustado['posicao_e_categoria']['f1']):>7s}{pct(ajustado['so_posicao']['f1']):>13s}" if ajustado else ""))

    print("\n== Etapa 1: a lista de conceitos (trechos ajustados às palavras)")
    print(f"   {'notas':26s}{'conceitos/nota':>15s}{'anotados na lista':>19s}{'(exata)':>9s}{'categoria certa':>17s}")
    linhas = [(f"{nome} ({etapa_1[nome]['notas']})", etapa_1[nome]) for nome in principais]
    if referencia_5c:
        linhas.append(("piloto no 5c (20)", referencia_5c["etapa_1"]))
    for rotulo_linha, r in linhas:
        print(f"   {rotulo_linha:26s}{r['conceitos_listados_por_nota'] or 0:15.1f}{pct(r['anotados_na_lista_parcial']):>19s}"
              f"{pct(r['anotados_na_lista_exata']):>9s}{pct(r['categoria_certa']):>17s}")

    print(f"\n== Etapa 2: as relações do {categorias.MODELO} (no limite: pedidos com {piloto.MAX_RELACOES} relações"
          f" na resposta, o limite)")
    print(f"   {'notas':34s}{'pedidos/nota':>13s}{'no limite':>10s}{'relações/nota':>14s}{'anotadas achadas':>18s}"
          f"{'(exata)':>9s}{'negações':>10s}{'novas':>8s}{'s/nota':>8s}{'cortadas':>10s}")

    def linha_da_etapa_2(rotulo_linha, r):
        if not r["notas"]:
            print(f"   {rotulo_linha:34s}    (nenhuma nota ligada ainda)")
            return
        print(f"   {rotulo_linha:34s}{r['pedidos_por_nota']:13.1f}{pct(r['pedidos_no_limite']):>10s}"
              f"{r['relacoes_por_nota']:14.1f}{pct(r['relacoes_anotadas_achadas_parcial']):>18s}"
              f"{pct(r['relacoes_anotadas_achadas_exata']):>9s}{pct(r['negacoes_achadas_parcial']):>10s}"
              f"{pct(r['relacoes_novas']):>8s}{r['segundos_por_nota']:8.0f}{r['respostas_cortadas']:>10}")

    for nome in ("sem o piloto", "sem o piloto, inteiras", "sem o piloto, em pedaços", "piloto"):
        linha_da_etapa_2(f"{nome} ({etapa_2[nome]['notas']})", etapa_2[nome])
    if referencia_5c and referencia_5c["etapa_2"]:
        linha_da_etapa_2("piloto no 5c (20)", referencia_5c["etapa_2"])
    if religadas:
        tamanhos = [len(listas[a]) for a in religadas]
        print(f"   as {len(religadas)} notas sem o piloto com {min(tamanhos)} a {max(tamanhos)} conceitos, ligadas"
              f" inteiras e de novo em pedaços:")
        for forma in ("inteiras", "em pedaços"):
            linha_da_etapa_2(f"   {forma}", etapa_2[f"mesmas notas, {forma}"])
    print(f"\n   regras de categoria (fração das relações){''.join(f'{s:>17s}' for s in categorias.SITUACOES)}")
    for nome in principais:
        for grupo, r in etapa_2[nome]["regras"].items():
            rotulo_linha = f"{nome}, {grupo} ({r['relacoes']})"
            print(f"   {rotulo_linha:43s}{''.join(f'{pct(r[s]):>17s}' for s in categorias.SITUACOES)}")

    print(f"\n== Grafos das {len(ligadas_sem_erro)} notas ligadas")
    print(f"   {'grafo':34s}{'entidades':>10s}{'triplas':>9s}{'tipos':>6s}{'vizinhos':>9s}{'(mediana)':>10s}"
          f"{'um vizinho':>11s}{'maior parte ligada':>19s}{'triplas em 2+ notas':>20s}")
    for nome, g in grafos.items():
        print(f"   {nome:34s}{g['entidades']:10d}{g['triplas']:9d}{len(g['tipos']):6d}{g['vizinhos_media']:9.1f}"
              f"{g['vizinhos_mediana']:10.0f}{pct(g['com_um_vizinho']):>11s}{pct(g['maior_componente']):>19s}"
              f"{pct(g['triplas_em_duas_notas_ou_mais']):>20s}")
    for nome in ("do texto, completo", "do texto, filtrado"):
        print(f"   tipos no grafo {nome}: {grafos[nome]['tipos']}")
    for nome, contagem in categorias_das_entidades.items():
        print(f"   entidades por categoria, {nome}: {contagem}")
    print(f"\nGrafos e entidades em {os.path.relpath(PASTA, RAIZ)}; medidas em {os.path.relpath(SAIDA_JSON, RAIZ)}")


def main():
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    argumentos.add_argument("parte", choices=("treinar", "ligar", "montar"))
    argumentos.add_argument("--host", default="http://127.0.0.1:11434", help="endereço do Ollama (parte ligar)")
    args = argumentos.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if args.parte == "treinar":
        treinar()
    elif args.parte == "ligar":
        ligar(args.host)
    else:
        montar()


if __name__ == "__main__":
    main()
