"""
Experimento 5e (teste pequeno): um modelo treinado com as anotações decide quais conceitos se ligam.

No experimento 5d, o modelo de linguagem (qwen2.5:7b) ligou sozinho os conceitos de cada nota: achou 43% das
relações anotadas, mas 83% das relações que criou não estão nas anotações. Aqui as anotações viram o "norte"
das ligações: um modelo pequeno, o BioBERTpt (como no 5c), aprende com as relações que os anotadores marcaram
e decide, para cada par de conceitos próximos na nota, se eles se ligam ("associated_with") ou se um nega o
outro ("negation_of"), os dois tipos do SemClinBr. O tipo da ligação no grafo vem das categorias dos dois
conceitos (TIPOS_PELAS_CATEGORIAS). As ligações com valores, doses e tempos ficam de fora do grafo, como
decidido depois do 5d; as medidas saem com todas as ligações e só com as ligações entre conceitos.

As relações anotadas são curtas: 74% ligam dois conceitos vizinhos no texto (sem outro conceito anotado entre
eles) e 97% ficam a até 60 caracteres. Por isso só os pares a até DISTANCIA_MAXIMA caracteres são candidatos.

É um teste pequeno, para ver se o caminho vale a pena antes de treinar com todas as notas: o modelo treina em
200 notas das partes 2 a 5 do 5d, escolhe a época e o limiar em outras 30 e é medido em 50 notas da parte 1,
com as mesmas listas de conceitos que o qwen recebeu no 5d (marcadas pelo modelo de conceitos que não viu a
parte 1). As notas do piloto ficam de fora. Nas 50 notas, compara as formas de ligar:
  qwen              as relações do experimento 5d
  vizinhos          cada conceito ligado ao seguinte no texto, sem passar de uma frase (uma referência simples)
  modelo treinado   os pares que o modelo treinado aceita
  qwen conferido    as relações do qwen que o modelo treinado também aceita
e mede também o modelo treinado com os conceitos dos anotadores, para ver quanto a lista de conceitos pesa.

Saída (não versionada):
  results/experimento5e_teste.json

Uso (no WSL, no ambiente conda do projeto):
    conda activate tcc_graphs
    python experimentos/05e_modelo_de_ligacoes.py
"""
import argparse
import copy
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
SAIDA_JSON = os.path.join(RAIZ, "results", "experimento5e_teste.json")
SEMENTE = 0
CLASSES = ("nenhuma", "associated_with", "negation_of")
DISTANCIA_MAXIMA = 60  # caracteres entre os dois conceitos de um par candidato
FAIXAS = (2, 5, 10, 20, 40, DISTANCIA_MAXIMA)  # faixas de distância que o modelo vê
FIM_DE_FRASE = r"[.;!?]\s|\n"
EPOCAS = 6
PACIENCIA = 2
LOTE = 4  # janelas por lote
TAXA, TAXA_DA_CABECA = 5e-5, 1e-3
LIMIARES = [k / 20 for k in range(1, 20)]
LITERAIS = {"valor", "tempo"}
TIPOS_DE_LITERAIS = {"tem_dose", "tem_valor", "tem_tempo"}
# O tipo da ligação pelas categorias: (tipo, categorias da origem, categorias do destino), na ordem em que são
# testados. Procedimento com doença ou sintoma fica "associado a": pode ser tratamento (cirurgia) ou resultado
# (biópsia).
TIPOS_PELAS_CATEGORIAS = (
    ("localizado_em", {"sinal_sintoma", "doenca", "procedimento", "exame"}, {"anatomia"}),
    ("trata", {"medicamento"}, {"doenca", "sinal_sintoma"}),
    ("tem_resultado", {"exame"}, {"sinal_sintoma", "doenca"}),
    ("tem_dose", {"medicamento", "procedimento"}, {"valor", "tempo"}),
    ("tem_valor", {"exame", "sinal_sintoma"}, {"valor"}),
    ("tem_tempo", {"doenca", "medicamento", "exame", "sinal_sintoma", "procedimento"}, {"tempo"}),
)


def carregar(nome, arquivo):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(RAIZ, "experimentos", arquivo))
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# Partes, listas de conceitos e relações do qwen: as do experimento 5d (e, por ele, do 5c, do 5b e do 5)
grafo = carregar("grafo", "05d_grafo_do_texto.py")
ner, piloto = grafo.ner, grafo.piloto
INDICE = {c: i for i, c in enumerate(ner.CATEGORIAS)}


def anotacoes_e_relacoes(arquivo, texto):
    """Os conceitos anotados, {(início, fim): categoria} no texto cru (com a correção de posição de
    anotacoes_no_texto no 5c), e as relações anotadas, {(conceito, conceito): classe} com o par na ordem do
    texto."""
    raiz = ET.parse(arquivo).getroot()
    por_id, conceitos = {}, {}
    for a in raiz.iter("annotation"):
        inicio, fim, alvo = int(a.get("start")), int(a.get("end")), html.unescape(a.get("text") or "")
        if ner.sem_espacos(texto[inicio:fim]) != ner.sem_espacos(alvo):
            i = texto.find(alvo, max(0, inicio - 30), fim + 30)
            if i < 0:
                continue
            inicio, fim = i, i + len(alvo)
        while inicio < fim and texto[inicio].isspace():
            inicio += 1
        while fim > inicio and texto[fim - 1].isspace():
            fim -= 1
        if fim > inicio:
            por_id[a.get("id")] = (inicio, fim)
            conceitos.setdefault((inicio, fim), ner.rotulo(a.get("tag") or ""))
    relacoes = {}
    for r in raiz.iter("rel"):
        a, b = por_id.get(r.get("annotation1")), por_id.get(r.get("annotation2"))
        if a and b and a != b and r.get("reltype") in CLASSES:
            relacoes[tuple(sorted((a, b)))] = CLASSES.index(r.get("reltype"))
    return conceitos, relacoes


def relacoes_entre_conceitos(arquivo):
    """As relações anotadas como piloto.ler_nota as lê, sem as que tocam um valor ou um tempo."""
    raiz = ET.parse(arquivo).getroot()
    anotacoes = {a.get("id"): (a.get("text") or "", ner.rotulo(a.get("tag") or "")) for a in raiz.findall(".//annotation")}
    return [(anotacoes[r.get("annotation1")][0], r.get("reltype"), anotacoes[r.get("annotation2")][0])
            for r in raiz.findall(".//rel")
            if r.get("annotation1") in anotacoes and r.get("annotation2") in anotacoes
            and not {anotacoes[r.get("annotation1")][1], anotacoes[r.get("annotation2")][1]} & LITERAIS]


def preparar(tokenizador, arquivo, item=None):
    """Texto, tokens, conceitos anotados e relações anotadas de uma nota; com o item marcado no 5d, também os
    conceitos marcados pelo modelo de conceitos (ajustados às palavras, como no 5d)."""
    texto = ner.texto_cru(arquivo)
    anotados, relacoes = anotacoes_e_relacoes(arquivo, texto)
    codigo = tokenizador(texto, add_special_tokens=False, return_offsets_mapping=True)
    nota = {"arquivo": os.path.basename(arquivo), "caminho": arquivo, "texto": texto, "ids": codigo["input_ids"],
            "offsets": codigo["offset_mapping"], "anotados": anotados, "relacoes": relacoes}
    if item:
        _, spans, nota["lista"] = grafo.conceitos_da_nota(item)
        nota["marcados"] = {(a, b): c for a, b, c in spans}
    return nota


def faixa(texto, s1, s2):
    """A faixa de distância entre dois conceitos, e se há um fim de frase entre eles."""
    distancia = max(0, max(s1[0], s2[0]) - min(s1[1], s2[1]))
    k = next(k for k, limite in enumerate(FAIXAS) if distancia <= limite)
    return k + len(FAIXAS) * bool(re.search(FIM_DE_FRASE, texto[min(s1[1], s2[1]):max(s1[0], s2[0])]))


def janelas_da_nota(nota, conceitos, tokenizador):
    """As janelas de tokens da nota (as mesmas do 5c), cada uma com os pares candidatos entre os conceitos que
    cabem inteiros nela. conceitos: {(início, fim): categoria}."""
    saida, offsets = [], nota["offsets"]
    for a, b in ner.janelas(len(nota["ids"])):
        dentro = []
        for (i, f), categoria in sorted(conceitos.items()):
            if i >= offsets[a][0] and f <= offsets[b - 1][1]:
                tokens = [k for k in range(a, b) if offsets[k][0] < f and offsets[k][1] > i]
                if tokens:  # posições no exemplo, depois do token especial do começo
                    dentro.append(((i, f), categoria, tokens[0] - a + 1, tokens[-1] - a + 2))
        pares = []
        for x in range(len(dentro)):
            for y in range(x + 1, len(dentro)):
                (s1, c1, a1, b1), (s2, c2, a2, b2) = dentro[x], dentro[y]
                if s2[0] - s1[1] > DISTANCIA_MAXIMA:
                    break  # os conceitos estão em ordem: os seguintes estão ainda mais longe
                pares.append({"par": (s1, s2), "categorias": (c1, c2), "classe": nota["relacoes"].get((s1, s2), 0),
                              "entrada": (a1, b1, a2, b2, INDICE[c1], INDICE[c2], faixa(nota["texto"], s1, s2))})
        if pares:
            saida.append({"ids": [tokenizador.cls_token_id, *nota["ids"][a:b], tokenizador.sep_token_id],
                          "pares": pares})
    return saida


def criar_modelo(torch, transformers):
    class Ligacoes(torch.nn.Module):
        """O BioBERTpt lê a janela; cada conceito é a média dos seus tokens, e uma camada pequena diz a classe
        do par a partir dos dois conceitos, das categorias e da distância."""

        def __init__(self):
            super().__init__()
            self.bert = transformers.AutoModel.from_pretrained(ner.MODELO_BASE)
            tamanho = self.bert.config.hidden_size
            self.categoria = torch.nn.Embedding(len(ner.CATEGORIAS), 32)
            self.distancia = torch.nn.Embedding(2 * len(FAIXAS), 16)
            self.cabeca = torch.nn.Sequential(torch.nn.Linear(3 * tamanho + 2 * 32 + 16, 256), torch.nn.ReLU(),
                                              torch.nn.Dropout(0.1), torch.nn.Linear(256, len(CLASSES)))

        def forward(self, ids, mascara, pares):
            estados = self.bert(input_ids=ids, attention_mask=mascara).last_hidden_state
            soma = torch.nn.functional.pad(estados.cumsum(1), (0, 0, 1, 0))  # soma[:, p] = soma dos p primeiros

            def media(janela, inicio, fim):
                return (soma[janela, fim] - soma[janela, inicio]) / (fim - inicio).unsqueeze(-1).to(soma.dtype)

            janela, a1, b1, a2, b2, c1, c2, d = pares.T
            e1, e2 = media(janela, a1, b1), media(janela, a2, b2)
            return self.cabeca(torch.cat([e1, e2, e1 * e2, self.categoria(c1), self.categoria(c2), self.distancia(d)], -1))

    return Ligacoes()


def tensores(torch, janelas, pad):
    maior = max(len(j["ids"]) for j in janelas)
    ids = torch.full((len(janelas), maior), pad)
    mascara = torch.zeros((len(janelas), maior), dtype=torch.long)
    pares, classes = [], []
    for linha, j in enumerate(janelas):
        ids[linha, :len(j["ids"])], mascara[linha, :len(j["ids"])] = torch.tensor(j["ids"]), 1
        pares += [(linha, *p["entrada"]) for p in j["pares"]]
        classes += [p["classe"] for p in j["pares"]]
    return ids, mascara, torch.tensor(pares), torch.tensor(classes)


def pontuar(torch, modelo, tokenizador, nota, conceitos):
    """A chance de cada classe para cada par candidato; num par que cai em duas janelas, vale a janela em que a
    ligação é mais provável."""
    melhor = {}
    janelas = janelas_da_nota(nota, conceitos, tokenizador)
    with torch.no_grad():
        for i in range(0, len(janelas), 8):
            lote = janelas[i:i + 8]
            chances = torch.softmax(modelo(*tensores(torch, lote, tokenizador.pad_token_id)[:3]), -1).tolist()
            for p, c in zip([p for j in lote for p in j["pares"]], chances):
                if p["par"] not in melhor or c[0] < melhor[p["par"]][0]:
                    melhor[p["par"]] = c
    return melhor


def aceitas(pontos, limiar):
    """{par: classe} dos pares com chance de ligação de pelo menos `limiar`; a classe é a mais provável entre
    as duas de ligação."""
    return {par: 1 + (c[2] > c[1]) for par, c in pontos.items() if 1 - c[0] >= limiar}


def medidas_dos_pares(previstos, gabarito):
    """Precisão, revocação e F1 dos pares ligados com a classe certa; previstos e gabarito: {nota: {par: classe}}."""
    certos = sum(previstos[n].get(par) == classe for n in gabarito for par, classe in gabarito[n].items())
    p = piloto.fracao(certos, sum(len(v) for v in previstos.values())) or 0.0
    r = piloto.fracao(certos, sum(len(v) for v in gabarito.values())) or 0.0
    return {"precisao": p, "revocacao": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def melhor_limiar(pontos, gabarito):
    return max(((limiar, medidas_dos_pares({n: aceitas(pontos[n], limiar) for n in pontos}, gabarito))
                for limiar in LIMIARES), key=lambda x: x[1]["f1"])


def treinar(torch, transformers, tokenizador, treino, validacao):
    """Treina nas notas de treino e devolve o modelo da época com o maior F1 na validação (com o melhor limiar),
    o limiar e o histórico."""
    torch.manual_seed(SEMENTE)
    exemplos = [j for n in treino for j in janelas_da_nota(n, n["anotados"], tokenizador)]
    modelo = criar_modelo(torch, transformers)
    cabeca = [p for nome, p in modelo.named_parameters() if not nome.startswith("bert.")]
    otimizador = torch.optim.AdamW([{"params": modelo.bert.parameters(), "lr": TAXA},
                                    {"params": cabeca, "lr": TAXA_DA_CABECA}], weight_decay=0.01)
    passos = EPOCAS * -(-len(exemplos) // LOTE)
    aquecimento = max(1, passos // 10)  # as taxas sobem nos primeiros 10% dos passos e depois descem até zero, como no 5c
    agenda = torch.optim.lr_scheduler.LambdaLR(
        otimizador, lambda p: min((p + 1) / aquecimento, max(0.0, (passos - p) / (passos - aquecimento))))
    pares = [p["classe"] for j in exemplos for p in j["pares"]]
    print(f"{len(exemplos)} janelas de treino com {len(pares)} pares candidatos ({sum(c > 0 for c in pares)} ligados);"
          f" até {EPOCAS} épocas; PyTorch {torch.__version__} com {torch.get_num_threads()} threads")
    gabarito = {n["arquivo"]: n["relacoes"] for n in validacao}
    historico, melhor, sem_melhorar = [], None, 0
    for epoca in range(1, EPOCAS + 1):
        modelo.train()
        comeco, perdas = time.time(), []
        ordem = list(range(len(exemplos)))
        random.Random(SEMENTE + epoca).shuffle(ordem)
        for i in range(0, len(ordem), LOTE):
            ids, mascara, entrada, classes = tensores(torch, [exemplos[k] for k in ordem[i:i + LOTE]],
                                                      tokenizador.pad_token_id)
            perda = torch.nn.functional.cross_entropy(modelo(ids, mascara, entrada), classes)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otimizador.step()
            agenda.step()
            otimizador.zero_grad()
            perdas.append(perda.item())
        modelo.eval()
        pontos = {n["arquivo"]: pontuar(torch, modelo, tokenizador, n, n["anotados"]) for n in validacao}
        limiar, m = melhor_limiar(pontos, gabarito)
        historico.append({"epoca": epoca, "perda_media": sum(perdas) / len(perdas), "limiar": limiar, "validacao": m,
                          "minutos": round((time.time() - comeco) / 60, 1)})
        print(f"   época {epoca}: perda {historico[-1]['perda_media']:.3f}; validação F1 {100 * m['f1']:.1f}%"
              f" (precisão {100 * m['precisao']:.1f}%, revocação {100 * m['revocacao']:.1f}%, limiar {limiar});"
              f" {historico[-1]['minutos']} min")
        if melhor is None or m["f1"] > melhor[0]:
            melhor, sem_melhorar = (m["f1"], limiar, copy.deepcopy(modelo.state_dict())), 0
        else:
            sem_melhorar += 1
            if sem_melhorar == PACIENCIA:
                break
    modelo.load_state_dict(melhor[2])
    modelo.eval()
    return modelo, melhor[1], historico


def tipo_da_ligacao(c1, c2, classe):
    """O tipo da ligação pelas categorias dos dois conceitos, e se a ordem deles deve ser trocada."""
    if classe == 2:
        return "nega", c2 == "negacao" and c1 != "negacao"
    for tipo, origens, destinos in TIPOS_PELAS_CATEGORIAS:
        if c1 in origens and c2 in destinos:
            return tipo, False
        if c2 in origens and c1 in destinos:
            return tipo, True
    return "associado_a", False


def como_relacoes(nota, ligados):
    """As relações de uma nota no formato do qwen ({origem, relacao, destino}, mais as categorias), sem repetir.
    ligados: {(conceito, conceito): (categoria, categoria, classe)}."""
    saida, vistas = [], set()
    for (s1, s2), (c1, c2, classe) in sorted(ligados.items()):
        tipo, trocar = tipo_da_ligacao(c1, c2, classe)
        t1, t2 = (" ".join(nota["texto"][a:b].split()) for a, b in (s1, s2))
        if trocar:
            t1, t2, c1, c2 = t2, t1, c2, c1
        chave = (piloto.normalizar(t1), tipo, piloto.normalizar(t2))
        if chave[0] and chave[2] and chave not in vistas:
            vistas.add(chave)
            saida.append({"origem": t1, "relacao": tipo, "destino": t2, "categorias": [c1, c2]})
    return saida


def do_modelo(nota, conceitos, pontos, limiar):
    return como_relacoes(nota, {par: (conceitos[par[0]], conceitos[par[1]], classe)
                                for par, classe in aceitas(pontos, limiar).items()})


def dos_vizinhos(nota):
    """Cada conceito marcado ligado ao seguinte no texto, se estiverem a até DISTANCIA_MAXIMA caracteres e sem um
    fim de frase entre eles; a ligação é uma negação se um dos dois for negação."""
    marcados, ligados = sorted(nota["marcados"].items()), {}
    for (s1, c1), (s2, c2) in zip(marcados, marcados[1:]):
        if 0 <= s2[0] - s1[1] <= DISTANCIA_MAXIMA and not re.search(FIM_DE_FRASE, nota["texto"][s1[1]:s2[0]]):
            ligados[(s1, s2)] = (c1, c2, 2 if "negacao" in (c1, c2) else 1)
    return como_relacoes(nota, ligados)


def do_qwen(nota, registro, pontos=None, limiar=None):
    """As relações do qwen no 5d, com as categorias das pontas (ponta do 5d). Com os pontos do modelo treinado, só
    as que ele também aceita: a maior chance de ligação entre as menções dos dois conceitos na nota."""
    conceitos = {piloto.normalizar(c["texto"]): c["categoria"] for c in nota["lista"]}
    mencoes = {}
    for (a, b) in nota["marcados"]:
        mencoes.setdefault(piloto.normalizar(" ".join(nota["texto"][a:b].split())), []).append((a, b))
    saida = []
    for r in registro["relacoes"]:
        chaves = [grafo.ponta(r[k], conceitos).rsplit("|", 1) for k in ("origem", "destino")]
        if pontos is not None:
            chance = max((1 - pontos[tuple(sorted((m1, m2)))][0]
                          for m1 in mencoes.get(chaves[0][0], []) for m2 in mencoes.get(chaves[1][0], [])
                          if tuple(sorted((m1, m2))) in pontos), default=0.0)
            if chance < limiar:
                continue
        saida.append({**r, "categorias": [chaves[0][1], chaves[1][1]]})
    return saida


def medir(notas, relacoes_por_nota):
    """As medidas do 5d (piloto.resumir) com todas as ligações e só com as ligações entre conceitos (sem as que
    tocam valores ou tempos, nas extraídas e nas anotadas), e quantos "trata" por nota."""
    resultado = {}
    for nome in ("todas", "entre conceitos"):
        execucoes = []
        for n in notas:
            nota, relacoes = n["nota"], relacoes_por_nota[n["arquivo"]]
            if nome == "entre conceitos":
                nota = {**nota, "relacoes": n["relacoes_entre_conceitos"]}
                relacoes = [r for r in relacoes
                            if r["relacao"] not in TIPOS_DE_LITERAIS and not set(r["categorias"]) & LITERAIS]
            execucoes.append({"avaliacao": piloto.avaliar(nota, relacoes, "livre"), "segundos": 0})
        resultado[nome] = piloto.resumir(execucoes, "livre")
    resultado["trata_por_nota"] = piloto.fracao(
        sum(r["relacao"] == "trata" for n in notas for r in relacoes_por_nota[n["arquivo"]]), len(notas))
    return resultado


def main():
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    argumentos.add_argument("--treino", type=int, default=200, help="notas de treino (das partes 2 a 5)")
    argumentos.add_argument("--validacao", type=int, default=30, help="notas para escolher a época e o limiar")
    argumentos.add_argument("--teste", type=int, default=50, help="notas de teste (da parte 1)")
    args = argumentos.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    import torch
    import transformers

    transformers.logging.set_verbosity_error()
    tokenizador = transformers.AutoTokenizer.from_pretrained(ner.MODELO_BASE)
    do_piloto = {os.path.basename(a) for a in ner.divisao()[2]}
    partes = [[a for a in p if os.path.basename(a) not in do_piloto] for p in grafo.partes()]
    resto = random.Random(SEMENTE).sample(sorted(a for p in partes[1:] for a in p), args.treino + args.validacao)
    with open(grafo.arquivo_conceitos(1), encoding="utf-8") as f:
        itens = {n["arquivo"]: n for n in json.load(f)["notas"]}
    registros = grafo.registros_de_relacoes()
    teste_arquivos = sorted(random.Random(SEMENTE).sample(partes[0], args.teste))
    comeco = time.time()
    treino = [preparar(tokenizador, a) for a in sorted(resto[:args.treino])]
    validacao = [preparar(tokenizador, a) for a in sorted(resto[args.treino:])]
    teste = [preparar(tokenizador, a, itens[os.path.basename(a)]) for a in teste_arquivos]
    for n in teste:
        n["nota"], n["relacoes_entre_conceitos"] = piloto.ler_nota(n["caminho"]), relacoes_entre_conceitos(n["caminho"])
    print(f"{len(treino)} notas de treino, {len(validacao)} de validação, {len(teste)} de teste;"
          f" preparadas em {time.time() - comeco:.0f} s")

    modelo, limiar, historico = treinar(torch, transformers, tokenizador, treino, validacao)
    pontos = {n["arquivo"]: pontuar(torch, modelo, tokenizador, n, n["marcados"]) for n in teste}
    pontos_anotados = {n["arquivo"]: pontuar(torch, modelo, tokenizador, n, n["anotados"]) for n in teste}
    formas = {
        "qwen (5d)": {n["arquivo"]: do_qwen(n, registros[n["arquivo"]]) for n in teste},
        "vizinhos": {n["arquivo"]: dos_vizinhos(n) for n in teste},
        "modelo treinado": {n["arquivo"]: do_modelo(n, n["marcados"], pontos[n["arquivo"]], limiar) for n in teste},
        "qwen conferido": {n["arquivo"]: do_qwen(n, registros[n["arquivo"]], pontos[n["arquivo"]], limiar)
                           for n in teste},
        "modelo, conceitos anotados": {n["arquivo"]: do_modelo(n, n["anotados"], pontos_anotados[n["arquivo"]], limiar)
                                       for n in teste}}
    medidas = {nome: medir(teste, relacoes) for nome, relacoes in formas.items()}
    teste_pares = medidas_dos_pares({n["arquivo"]: aceitas(pontos_anotados[n["arquivo"]], limiar) for n in teste},
                                    {n["arquivo"]: n["relacoes"] for n in teste})
    grafo.salvar({"protocolo": {"notas_treino": [n["arquivo"] for n in treino],
                                "notas_validacao": [n["arquivo"] for n in validacao],
                                "notas_teste": [n["arquivo"] for n in teste], "distancia_maxima": DISTANCIA_MAXIMA,
                                "limiar": limiar, "minutos": round((time.time() - comeco) / 60, 1)},
                  "historico": historico, "teste_com_conceitos_anotados": teste_pares, "medidas": medidas,
                  "relacoes": {nome: relacoes for nome, relacoes in formas.items() if nome != "qwen (5d)"}}, SAIDA_JSON)

    pct = piloto.pct
    print(f"\n   modelo treinado, conceitos anotados, pares com a classe certa: precisão {pct(teste_pares['precisao'])},"
          f" revocação {pct(teste_pares['revocacao'])}, F1 {pct(teste_pares['f1'])} (limiar {limiar})")
    for nome in ("todas", "entre conceitos"):
        titulo = "todas as ligações" if nome == "todas" else "só entre conceitos (sem valor, dose e tempo)"
        print(f"\n== {len(teste)} notas de teste, {titulo}")
        print(f"   {'forma':28s}{'ligações/nota':>14s}{'anotadas achadas':>18s}{'(exata)':>9s}"
              f"{'batem c/ anotações':>20s}{'negações':>10s}{'trata/nota':>12s}")
        for forma, m in medidas.items():
            r = m[nome]
            print(f"   {forma:28s}{r['relacoes_por_nota']:14.1f}{pct(r['relacoes_anotadas_achadas_parcial']):>18s}"
                  f"{pct(r['relacoes_anotadas_achadas_exata']):>9s}"
                  f"{pct(None if r['relacoes_novas'] is None else 1 - r['relacoes_novas']):>20s}"
                  f"{pct(r['negacoes_achadas_parcial']):>10s}{m['trata_por_nota']:12.1f}")
    print(f"\nMedidas em {os.path.relpath(SAIDA_JSON, RAIZ)} ({(time.time() - comeco) / 60:.0f} min)")


if __name__ == "__main__":
    main()
