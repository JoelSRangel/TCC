"""
Experimento 5c (piloto): um modelo treinado para a etapa 1.

No experimento 5b, o modelo de linguagem ligou bem os conceitos quando recebeu
a lista dos anotadores (48% das relações anotadas), mas foi fraco para fazer a
lista sozinho (27%). Aqui, quem faz a lista é um modelo pequeno treinado com as
anotações do SemClinBr para marcar os conceitos no texto (reconhecimento de
entidades): o BioBERTpt (pucpr/biobertpt-all), um BERT para português clínico
e biomédico, do mesmo grupo de pesquisa que criou o SemClinBr. Ele é treinado
nas notas que não estão no piloto (90% para treinar, 10% para escolher a
época) e avaliado nas mesmas 20 notas dos experimentos 5 e 5b. As categorias
são os grupos do experimento 5b, mais "abreviatura" para os conceitos marcados
só como abreviatura, sem tipo; quando um conceito tem tipos de grupos
diferentes, vale o primeiro grupo na ordem do experimento 5b.

A etapa 2 é a mesma da versão com categorias do experimento 5b: o modelo de
linguagem liga os conceitos da lista.

O script tem duas partes, porque o treino usa o PyTorch do ambiente conda do
WSL e a etapa 2 usa o Ollama do Windows, que o WSL não alcança:
  treinar   treina o modelo, faz as listas das 20 notas e mede a etapa 1
  ligar     manda cada lista para o modelo de linguagem e mede as relações

Posições das anotações: os arquivos usam quebras de linha CRLF e as posições
contam os dois caracteres, mas o leitor de XML troca CRLF por LF, o que
desloca as posições. Por isso o texto da nota é lido direto do arquivo. As
anotações cuja posição não bate com o texto são procuradas perto dela; as que
não aparecem ficam de fora (o script conta quantas).

Saídas:
  results/experimento5c_ner.json     (não versionado)
  ~/tcc_modelos/experimento5c_ner    (o modelo treinado, fora do repositório)

Uso:
    # no WSL, no ambiente conda do projeto, com o pacote transformers instalado
    conda activate tcc_graphs
    python experimentos/05c_piloto_com_ner.py treinar
    # no Windows, com o Ollama rodando e o qwen2.5:7b baixado
    python experimentos/05c_piloto_com_ner.py ligar
"""
import argparse
import glob
import html
import importlib.util
import json
import os
import random
import re
import sys
import time
import xml.etree.ElementTree as ET

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAIDA_JSON = os.path.join(RAIZ, "results", "experimento5c_ner.json")
SAIDA_5B = os.path.join(RAIZ, "results", "experimento5b_categorias.json")
PASTA_MODELO = os.path.join(os.path.expanduser("~"), "tcc_modelos", "experimento5c_ner")
MODELO_BASE = "pucpr/biobertpt-all"
SEMENTE = 0
FRACAO_VALIDACAO = 0.1
EPOCAS = 10
PACIENCIA = 2  # épocas seguidas sem melhorar na validação antes de parar
LOTE = 8
TAXA = 5e-5
TOKENS_POR_JANELA = 254  # mais os dois tokens especiais do BERT: 256
SOBREPOSICAO = 64  # notas longas viram janelas que se sobrepõem


def carregar(nome, arquivo):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(RAIZ, "experimentos", arquivo))
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# Grupos, gabarito, etapa 2 e medidas: os mesmos do experimento 5b (e, por ele, do 5)
categorias = carregar("categorias", "05b_piloto_com_categorias.py")
piloto = categorias.piloto

CATEGORIAS = (*categorias.GRUPOS, "abreviatura")
ROTULOS = ("O", *(f"{prefixo}-{c}" for c in CATEGORIAS for prefixo in ("B", "I")))


def texto_cru(arquivo):
    """Texto da nota como está no arquivo, com as quebras de linha CRLF que as posições das anotações contam."""
    bruto = open(arquivo, "rb").read().decode("utf-8")
    texto = re.search(r"<TEXT>(.*?)</TEXT>", bruto, re.S).group(1)
    return texto[9:-3] if texto.startswith("<![CDATA[") else html.unescape(texto)


def sem_espacos(t):
    return re.sub(r"\s+", "", t)


def rotulo(tag):
    """A categoria de uma anotação: o primeiro dos seus grupos, ou "abreviatura" se ela não tem tipo."""
    grupos = categorias.grupos(tag)
    return next((g for g in categorias.GRUPOS if g in grupos), "abreviatura")


def anotacoes_no_texto(arquivo, texto):
    """(início, fim, categoria) de cada anotação no texto cru, e quantas foram achadas perto da posição marcada
    ou ficaram de fora."""
    spans, contagem = [], {"no_lugar": 0, "achadas_perto": 0, "fora": 0}
    for a in ET.parse(arquivo).getroot().iter("annotation"):
        inicio, fim, alvo = int(a.get("start")), int(a.get("end")), html.unescape(a.get("text") or "")
        if sem_espacos(texto[inicio:fim]) == sem_espacos(alvo):
            contagem["no_lugar"] += 1
        else:
            i = texto.find(alvo, max(0, inicio - 30), fim + 30)
            if i < 0:
                contagem["fora"] += 1
                continue
            inicio, fim = i, i + len(alvo)
            contagem["achadas_perto"] += 1
        while inicio < fim and texto[inicio].isspace():  # sem espaços nas pontas, como os trechos que o modelo marca
            inicio += 1
        while fim > inicio and texto[fim - 1].isspace():
            fim -= 1
        spans.append((inicio, fim, rotulo(a.get("tag"))))
    return spans, contagem


def divisao():
    """Notas de treino, de validação e de teste (as 20 do piloto, sorteadas como nos experimentos 5 e 5b)."""
    arquivos = sorted(glob.glob(os.path.join(piloto.PASTA_XML, "*.xml")))
    teste = sorted(random.Random(piloto.SEMENTE).sample(arquivos, 20))
    resto = sorted(set(arquivos) - set(teste))
    validacao = sorted(random.Random(SEMENTE).sample(resto, round(FRACAO_VALIDACAO * len(resto))))
    return sorted(set(resto) - set(validacao)), validacao, teste


def janelas(n_tokens):
    """(início, fim) das janelas de tokens que cobrem a nota."""
    inicio, saida = 0, []
    while True:
        fim = min(inicio + TOKENS_POR_JANELA, n_tokens)
        saida.append((inicio, fim))
        if fim == n_tokens:
            return saida
        inicio = fim - SOBREPOSICAO


def rotular(offsets, spans):
    """Rótulo BIO de cada token; quando duas anotações se sobrepõem (raro), fica a mais longa."""
    rotulos = ["O"] * len(offsets)
    for inicio, fim, categoria in sorted(spans, key=lambda s: s[0] - s[1]):
        dentro = [i for i, (a, b) in enumerate(offsets) if a < fim and b > inicio]
        if dentro and all(rotulos[i] == "O" for i in dentro):
            rotulos[dentro[0]] = f"B-{categoria}"
            for i in dentro[1:]:
                rotulos[i] = f"I-{categoria}"
    return rotulos


def decodificar(rotulos, offsets):
    """(início, fim, categoria) dos trechos marcados pelos rótulos BIO."""
    spans, atual = [], None
    for r, (a, b) in zip(rotulos, offsets):
        if r == "O":
            if atual:
                spans.append(tuple(atual))
            atual = None
            continue
        prefixo, categoria = r.split("-", 1)
        if prefixo == "B" or atual is None or atual[2] != categoria:
            if atual:
                spans.append(tuple(atual))
            atual = [a, b, categoria]
        else:
            atual[1] = b
    if atual:
        spans.append(tuple(atual))
    return spans


def lista_de_conceitos(texto, spans):
    """Conceitos marcados, na ordem do texto e sem repetir (no formato da lista da etapa 1 do experimento 5b)."""
    lista, vistos = [], set()
    for inicio, fim, categoria in sorted(spans):
        conceito = " ".join(texto[inicio:fim].split())
        if piloto.normalizar(conceito) and piloto.normalizar(conceito) not in vistos:
            vistos.add(piloto.normalizar(conceito))
            lista.append({"texto": conceito, "categoria": categoria})
    return lista


def medidas(previstos, gabarito):
    """Precisão, revocação e F1 de trechos com a mesma posição e categoria (e só com a mesma posição)."""
    resultado = {}
    for nome, chave in (("posicao_e_categoria", lambda s: s), ("so_posicao", lambda s: s[:2])):
        p = {(n, chave(s)) for n, spans in previstos.items() for s in spans}
        g = {(n, chave(s)) for n, spans in gabarito.items() for s in spans}
        precisao, revocacao = piloto.fracao(len(p & g), len(p)), piloto.fracao(len(p & g), len(g))
        f1 = 2 * precisao * revocacao / (precisao + revocacao) if precisao and revocacao else 0.0
        resultado[nome] = {"precisao": precisao, "revocacao": revocacao, "f1": f1}
    return resultado


def preparar(tokenizador, arquivo):
    """Texto cru, anotações e tokens de uma nota."""
    texto = texto_cru(arquivo)
    spans, contagem = anotacoes_no_texto(arquivo, texto)
    codigo = tokenizador(texto, add_special_tokens=False, return_offsets_mapping=True)
    return {"arquivo": os.path.basename(arquivo), "texto": texto, "spans": spans, "contagem": contagem,
            "ids": codigo["input_ids"], "offsets": codigo["offset_mapping"]}


def prever(modelo, tokenizador, nota):
    """Trechos marcados pelo modelo; nos tokens que caem em duas janelas, vale a janela em que o token está
    mais longe da borda."""
    import torch

    melhor = [None] * len(nota["ids"])
    with torch.no_grad():
        for a, b in janelas(len(nota["ids"])):
            entrada = torch.tensor([[tokenizador.cls_token_id, *nota["ids"][a:b], tokenizador.sep_token_id]])
            previsto = modelo(input_ids=entrada).logits[0, 1:-1].argmax(-1).tolist()
            for k, i in enumerate(range(a, b)):
                distancia = min(i - a, b - 1 - i)
                if melhor[i] is None or distancia > melhor[i][0]:
                    melhor[i] = (distancia, previsto[k])
    return decodificar([ROTULOS[m[1]] for m in melhor], nota["offsets"])


def avaliar_notas(modelo, tokenizador, notas):
    modelo.eval()
    previstos = {n["arquivo"]: prever(modelo, tokenizador, n) for n in notas}
    return previstos, medidas(previstos, {n["arquivo"]: n["spans"] for n in notas})


def treinar_modelo(tokenizador, treino, validacao, pasta):
    """Treina o modelo nas notas de treino e salva em `pasta` o da época com o maior F1 na validação.
    Devolve o histórico das épocas."""
    import torch
    import transformers

    torch.manual_seed(SEMENTE)
    indice = {r: i for i, r in enumerate(ROTULOS)}
    exemplos = []
    for n in treino:
        rotulos = [indice[r] for r in rotular(n["offsets"], n["spans"])]
        for a, b in janelas(len(n["ids"])):
            exemplos.append(([tokenizador.cls_token_id, *n["ids"][a:b], tokenizador.sep_token_id],
                             [-100, *rotulos[a:b], -100]))

    def lotes(rng):
        """Lotes de exemplos de tamanho parecido (menos enchimento), em ordem aleatória."""
        ordem = list(range(len(exemplos)))
        rng.shuffle(ordem)
        blocos = [sorted(ordem[i:i + 50 * LOTE], key=lambda k: len(exemplos[k][0]))
                  for i in range(0, len(ordem), 50 * LOTE)]
        saida = [bloco[i:i + LOTE] for bloco in blocos for i in range(0, len(bloco), LOTE)]
        rng.shuffle(saida)
        return saida

    def tensores(lote):
        maior = max(len(exemplos[k][0]) for k in lote)
        ids = torch.full((len(lote), maior), tokenizador.pad_token_id)
        mascara = torch.zeros((len(lote), maior), dtype=torch.long)
        alvos = torch.full((len(lote), maior), -100)
        for linha, k in enumerate(lote):
            x, y = exemplos[k]
            ids[linha, :len(x)], mascara[linha, :len(x)], alvos[linha, :len(y)] = torch.tensor(x), 1, torch.tensor(y)
        return ids, mascara, alvos

    modelo = transformers.AutoModelForTokenClassification.from_pretrained(
        MODELO_BASE, num_labels=len(ROTULOS), id2label=dict(enumerate(ROTULOS)), label2id=indice)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=TAXA, weight_decay=0.01)
    passos = EPOCAS * -(-len(exemplos) // LOTE)
    aquecimento = passos // 10  # a taxa sobe nos primeiros 10% dos passos e depois desce até zero
    agenda = torch.optim.lr_scheduler.LambdaLR(
        otimizador, lambda p: min((p + 1) / aquecimento, max(0.0, (passos - p) / (passos - aquecimento))))
    print(f"{len(exemplos)} janelas de treino; até {EPOCAS} épocas de {-(-len(exemplos) // LOTE)} lotes;"
          f" PyTorch {torch.__version__} com {torch.get_num_threads()} threads")

    historico, melhor_f1, sem_melhorar = [], -1.0, 0
    for epoca in range(1, EPOCAS + 1):
        modelo.train()
        perdas, inicio_epoca = [], time.time()
        for numero, lote in enumerate(lotes(random.Random(SEMENTE + epoca)), 1):
            ids, mascara, alvos = tensores(lote)
            perda = modelo(input_ids=ids, attention_mask=mascara, labels=alvos).loss
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otimizador.step()
            agenda.step()
            otimizador.zero_grad()
            perdas.append(perda.item())
            if epoca == 1 and numero == 20:
                print(f"   primeiros 20 lotes em {time.time() - inicio_epoca:.0f} s")
        _, m = avaliar_notas(modelo, tokenizador, validacao)
        f1 = m["posicao_e_categoria"]["f1"]
        historico.append({"epoca": epoca, "perda_media": sum(perdas) / len(perdas), "validacao": m,
                          "minutos": round((time.time() - inicio_epoca) / 60, 1)})
        print(f"   época {epoca}: perda {historico[-1]['perda_media']:.3f}; validação F1 {100 * f1:.1f}%"
              f" (só posição {100 * m['so_posicao']['f1']:.1f}%); {historico[-1]['minutos']} min")
        if f1 > melhor_f1:
            melhor_f1, sem_melhorar = f1, 0
            modelo.save_pretrained(pasta)
            tokenizador.save_pretrained(pasta)
        else:
            sem_melhorar += 1
            if sem_melhorar == PACIENCIA:
                break
    return historico


def treinar():
    import torch
    import transformers

    transformers.logging.set_verbosity_error()
    tokenizador = transformers.AutoTokenizer.from_pretrained(MODELO_BASE)
    arquivos_treino, arquivos_validacao, arquivos_teste = divisao()
    treino = [preparar(tokenizador, a) for a in arquivos_treino]
    validacao = [preparar(tokenizador, a) for a in arquivos_validacao]
    teste = [preparar(tokenizador, a) for a in arquivos_teste]
    contagem = {k: sum(n["contagem"][k] for n in treino + validacao + teste) for k in ("no_lugar", "achadas_perto", "fora")}
    print(f"Notas: {len(treino)} de treino, {len(validacao)} de validação, {len(teste)} de teste (as do piloto)")
    print(f"Anotações: {contagem['no_lugar']} no lugar marcado, {contagem['achadas_perto']} achadas perto,"
          f" {contagem['fora']} de fora")

    comeco = time.time()
    historico = treinar_modelo(tokenizador, treino, validacao, PASTA_MODELO)
    modelo = transformers.AutoModelForTokenClassification.from_pretrained(PASTA_MODELO)
    previstos_validacao, medidas_validacao = avaliar_notas(modelo, tokenizador, validacao)
    previstos_teste, medidas_teste = avaliar_notas(modelo, tokenizador, teste)
    notas_teste = []
    for n in teste:
        lista = lista_de_conceitos(n["texto"], previstos_teste[n["arquivo"]])
        nota = piloto.ler_nota(os.path.join(piloto.PASTA_XML, n["arquivo"]))
        notas_teste.append({"arquivo": n["arquivo"], "conceitos": lista,
                            "avaliacao_conceitos": categorias.avaliar_conceitos(nota, lista)})
    etapa_1 = categorias.resumir_conceitos(notas_teste)

    dados = {"protocolo": {"modelo_base": MODELO_BASE, "semente": SEMENTE, "epocas_maximas": EPOCAS,
                           "paciencia": PACIENCIA, "lote": LOTE, "taxa": TAXA, "tokens_por_janela": TOKENS_POR_JANELA,
                           "sobreposicao": SOBREPOSICAO, "rotulos": ROTULOS, "anotacoes": contagem,
                           "torch": torch.__version__, "transformers": transformers.__version__,
                           "notas_validacao": [n["arquivo"] for n in validacao]},
             "historico": historico, "minutos_de_treino": round((time.time() - comeco) / 60, 1),
             "validacao": medidas_validacao, "teste": {"medidas": medidas_teste, "etapa_1": etapa_1,
                                                       "notas": notas_teste}}
    with open(SAIDA_JSON, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

    pct = piloto.pct
    print("\n== Marcação de conceitos (trecho com a mesma posição e categoria do gabarito)")
    for nome, m in (("validação", medidas_validacao), ("teste (20 notas do piloto)", medidas_teste)):
        print(f"   {nome:28s} precisão {pct(m['posicao_e_categoria']['precisao']).strip()},"
              f" revocação {pct(m['posicao_e_categoria']['revocacao']).strip()},"
              f" F1 {pct(m['posicao_e_categoria']['f1']).strip()} (só posição: F1 {pct(m['so_posicao']['f1']).strip()})")
    print("\n== Etapa 1 nas 20 notas do piloto: lista do modelo treinado x lista do modelo de linguagem (5b)")
    with open(SAIDA_5B, encoding="utf-8") as f:
        llm = json.load(f)["resumo"]["com categorias"]["etapa_1"]
    for nome, r in (("modelo treinado", etapa_1), ("modelo de linguagem", llm)):
        print(f"   {nome:20s} {r['conceitos_listados_por_nota']:5.1f} conceitos por nota;"
              f" {pct(r['anotados_na_lista_parcial']).strip()} dos anotados na lista"
              f" ({pct(r['anotados_na_lista_exata']).strip()} exata); categoria certa {pct(r['categoria_certa']).strip()}")
    print(f"   erros de categoria mais comuns do modelo treinado: {etapa_1['erros_de_categoria_mais_comuns']}")
    print(f"\nResultados em {os.path.relpath(SAIDA_JSON, RAIZ)}; modelo em {PASTA_MODELO}")


def ligar(host):
    with open(SAIDA_JSON, encoding="utf-8") as f:
        dados = json.load(f)
    try:
        versao = piloto.chamar(f"{host}/api/version", timeout=10).get("version")
    except (OSError, ValueError) as erro:
        sys.exit(f"Não foi possível falar com o Ollama em {host} ({erro}). Ele está rodando?")
    modelo = categorias.MODELO
    # Carrega o modelo antes de medir o tempo, com o mesmo tamanho de contexto do piloto
    piloto.chamar(f"{host}/api/generate", {"model": modelo, "prompt": "ok", "stream": False, "keep_alive": "30m",
                                           "options": {"num_ctx": 8192, "num_predict": 1}})
    print(f"Ollama {versao}; etapa 2 com o {modelo}, a partir das listas do modelo treinado")

    execucoes = []
    for numero, item in enumerate(dados["teste"]["notas"], 1):
        nota = piloto.ler_nota(os.path.join(piloto.PASTA_XML, item["arquivo"]))
        lista = [(c["texto"], set() if c["categoria"] == "abreviatura" else {c["categoria"]}) for c in item["conceitos"]]
        comeco, registro = time.time(), {"nota": item["arquivo"]}
        try:
            extraidas, crua, cortada, tokens = piloto.extrair(categorias.prompt_relacoes(nota, lista), modelo, host)
        except (OSError, ValueError) as erro:  # sem conexão, erro do servidor ou resposta que não é JSON
            extraidas, crua, cortada, tokens = [], None, None, None
            registro["erro"] = repr(erro)
        avaliacao = piloto.avaliar(nota, extraidas, "livre")
        registro.update(relacoes=extraidas, resposta=crua, cortada=cortada, tokens=tokens,
                        segundos=round(time.time() - comeco, 1), avaliacao=avaliacao,
                        regras=categorias.avaliar_regras(extraidas, avaliacao["novas"],
                                                         {piloto.normalizar(t): g for t, g in lista}))
        execucoes.append(registro)
        print(f"   [{numero}/{len(dados['teste']['notas'])}] {item['arquivo']}: {len(lista)} conceitos,"
              f" {avaliacao['relacoes_extraidas']} relações, {avaliacao['relacoes_anotadas_achadas_parcial']}/"
              f"{avaliacao['relacoes_anotadas']} anotadas achadas ({registro['segundos']} s)")
    resumo = piloto.resumir(execucoes, "livre")
    resumo["regras"] = categorias.resumir_regras(execucoes)
    dados["ligar"] = {"ollama": versao, "modelo": modelo, "execucoes": execucoes, "resumo": resumo}
    with open(SAIDA_JSON, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

    pct = piloto.pct
    with open(SAIDA_5B, encoding="utf-8") as f:
        anteriores = json.load(f)["resumo"]
    formas = {"livre (5b)": anteriores["livre"], "com categorias (5b)": anteriores["com categorias"],
              "com o modelo treinado": resumo, "guiada com grupos (5b)": anteriores["guiada com grupos"]}
    print("\n== Relações nas 20 notas do piloto, por quem faz a lista de conceitos")
    print(f"   {'forma':24s}{'relações/nota':>14s}{'anotadas achadas':>18s}{'(exata)':>9s}{'negações':>10s}"
          f"{'novas':>8s}{'s/nota':>8s}")
    for nome, r in formas.items():
        print(f"   {nome:24s}{r['relacoes_por_nota']:14.1f}{pct(r['relacoes_anotadas_achadas_parcial']):>18s}"
              f"{pct(r['relacoes_anotadas_achadas_exata']):>9s}{pct(r['negacoes_achadas_parcial']):>10s}"
              f"{pct(r['relacoes_novas']):>8s}{r['segundos_por_nota']:8.0f}")
    print(f"\n   regras de categoria (fração das relações){''.join(f'{s:>17s}' for s in categorias.SITUACOES)}")
    for nome in ("com categorias (5b)", "com o modelo treinado", "guiada com grupos (5b)"):
        for grupo, r in formas[nome]["regras"].items():
            rotulo_linha = f"{nome}, {'batem' if grupo == 'batem' else 'novas'} ({r['relacoes']})"
            print(f"   {rotulo_linha:43s}{''.join(f'{pct(r[s]):>17s}' for s in categorias.SITUACOES)}")
    print(f"\nResultados em {os.path.relpath(SAIDA_JSON, RAIZ)}")


def main():
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    argumentos.add_argument("parte", choices=("treinar", "ligar"))
    argumentos.add_argument("--host", default="http://127.0.0.1:11434", help="endereço do Ollama (parte ligar)")
    args = argumentos.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if args.parte == "treinar":
        treinar()
    else:
        ligar(args.host)


if __name__ == "__main__":
    main()
