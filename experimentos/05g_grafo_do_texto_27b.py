"""
Experimento 5g: o grafo do texto das 998 notas, ligado pelo qwen3.5:27b.

A bancada (5f) mostrou que o modelo de linguagem era o maior problema da etapa 2. Com o mesmo pedido, o
qwen3.5:27b acha 61% das ligações anotadas entre conceitos, contra 35% do qwen2.5:7b usado no 5d; erra menos e
acha 75% das negações. Aqui as 998 notas são ligadas de novo com ele, com o pedido da forma 4 da bancada (os 5
tipos e duas notas de exemplo, sem os conceitos de valor e de tempo na lista) e as mesmas listas de conceitos do
5d (cada parte marcada pelo modelo de conceitos que não a viu).

Ele é lento (uns 4,5 minutos por nota, quase 15 horas por parte), então as notas vão parte por parte, nas mesmas
5 partes do 5d, e o grafo pode ser montado com as partes que já estão prontas. As 50 notas de teste da bancada,
da parte 1, já foram ligadas pela forma 4, com o mesmo modelo, pedido e lista: os registros delas são trazidos
da bancada em vez de ligar de novo.

O script tem duas partes:
  ligar   (Windows, com o Ollama) liga as notas das partes pedidas; se for interrompida, continua de onde parou
  montar  junta as notas ligadas no grafo e mede o resultado

O grafo é montado como no 5d (entidades "texto normalizado|categoria", um grafo completo e um filtrado pelas
regras de categoria do 5b), mas sem as ligações que tocam valores ou tempos. Montar faz um grafo para cada parte
pronta e um para todas as partes prontas juntas. As medidas contra as anotações são só entre conceitos (como na
bancada) e não contam as 20 notas do piloto, que serviram para ajustar os pedidos, nem as 2 notas de exemplo, cuja
resposta vai no próprio pedido; os grafos têm todas as notas. Para comparar, montar mede também o qwen2.5:7b do
5d nas mesmas notas.

Saídas (não versionadas):
  results/experimento5g/relacoes.jsonl     relações de cada nota, uma linha por nota (com a parte, o arquivo .xml,
                                           as respostas cruas do modelo e quando a nota foi ligada)
  results/experimento5g/parte<k>_notas.csv as notas (arquivos .xml) da parte k e a situação de cada uma: ligada,
                                           trazida da bancada, com erro ou pendente; atualizada a cada nota
  results/experimento5g/parte<k>/          o grafo da parte k, quando ela fica pronta: notas.txt (os arquivos .xml
                                           que entraram), grafo_completo.tsv e grafo_filtrado.tsv (triplas origem,
                                           tipo, destino, sem cabeçalho, no formato que o PyKEEN lê) e entidades.json
  results/experimento5g/prontas/           o mesmo, com todas as partes prontas juntas
  results/experimento5g/LEIA-ME.md         o que é cada arquivo e a situação de cada parte, refeito por montar
  results/experimento5g_resultados.json    medidas

Uso:
    python -X utf8 experimentos/05g_grafo_do_texto_27b.py ligar 1            # só a parte 1
    python -X utf8 experimentos/05g_grafo_do_texto_27b.py ligar --montar     # todas as partes, na ordem, montando
                                                                             # os grafos a cada parte pronta
    python -X utf8 experimentos/05g_grafo_do_texto_27b.py montar
Para pausar a parte ligar (por exemplo, para usar o computador em outra coisa pesada), crie um arquivo vazio
chamado PAUSAR em results/experimento5g: ela para depois da nota em que está. Para continuar, apague o arquivo e
rode a parte ligar de novo.
"""
import argparse
import collections
import csv
import importlib.util
import json
import os
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA = os.path.join(RAIZ, "results", "experimento5g")
SAIDA_RELACOES = os.path.join(PASTA, "relacoes.jsonl")
PASTA_PRONTAS = os.path.join(PASTA, "prontas")
ARQUIVO_PAUSA = os.path.join(PASTA, "PAUSAR")
SAIDA_JSON = os.path.join(RAIZ, "results", "experimento5g_resultados.json")
FORMA = "4"  # a forma da bancada usada aqui


def carregar(nome, arquivo):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(RAIZ, "experimentos", arquivo))
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# O pedido, a extração e as notas marcadas: os da bancada (5f) e, por ela, os do 5d
bancada = carregar("bancada", "05f_bancada_de_extracao.py")
grafo, ligacoes, piloto = bancada.grafo, bancada.ligacoes, bancada.piloto
MODELO, COM_EXEMPLOS = bancada.FORMAS[FORMA]


def registros_de_relacoes():
    """O último registro de cada nota já ligada. Uma linha cortada (se o computador desligou no meio da gravação)
    é ignorada, e a nota é ligada de novo."""
    saida = {}
    if os.path.exists(SAIDA_RELACOES):
        with open(SAIDA_RELACOES, encoding="utf-8") as f:
            for linha in f:
                try:
                    r = json.loads(linha)
                except json.JSONDecodeError:
                    continue
                saida[r["nota"]] = r
    return saida


def gravar(registros):
    with open(SAIDA_RELACOES, "a", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in registros)


def arquivo_da_parte(k):
    return os.path.join(PASTA, f"parte{k}_notas.csv")


def atualizar_lista_da_parte(k, itens):
    """A planilha das notas (arquivos .xml) da parte k, com a situação de cada uma: ligada, trazida da bancada,
    com erro ou pendente. Se ela estiver aberta no Excel, fica para a próxima nota."""
    registros = registros_de_relacoes()
    try:
        with open(arquivo_da_parte(k) + ".tmp", "w", encoding="utf-8-sig", newline="") as f:
            escritor = csv.writer(f, delimiter=";")
            escritor.writerow(["nota", "situação", "relações", "segundos", "ligada em"])
            for item in itens:
                r = registros.get(item["arquivo"]) or {}
                situacao = ("pendente" if not r else "com erro" if r.get("erro") else
                            "trazida da bancada (5f)" if r.get("de") else "ligada")
                escritor.writerow([item["arquivo"], situacao, len(r["relacoes"]) if r else "",
                                   r.get("segundos", ""), r.get("quando", "")])
        os.replace(arquivo_da_parte(k) + ".tmp", arquivo_da_parte(k))
    except OSError as erro:
        print(f"   (não deu para atualizar {os.path.relpath(arquivo_da_parte(k), RAIZ)}: {erro})")


def trazer_as_da_bancada():
    """Os registros das 50 notas de teste que a forma 4 da bancada já ligou, com o mesmo modelo, pedido e lista."""
    feitas = registros_de_relacoes()
    trazidas = [{**r, "parte": 1, "de": f"5f, forma {FORMA}"} for a, r in sorted(bancada.registros(FORMA).items())
                if not r.get("erro") and a not in feitas]
    if trazidas:
        gravar(trazidas)
        print(f"   {len(trazidas)} notas da parte 1 trazidas da bancada (5f, forma {FORMA})")


def parte_pronta(itens):
    registros = registros_de_relacoes()
    return all(i["arquivo"] in registros and not registros[i["arquivo"]].get("erro") for i in itens)


def ligar(partes, host, montar_cada_parte):
    try:
        versao = piloto.chamar(f"{host}/api/version", timeout=10).get("version")
    except (OSError, ValueError) as erro:
        sys.exit(f"Não foi possível falar com o Ollama em {host} ({erro}). Ele está rodando?")
    sem_pensar = bancada.pensa(MODELO, host)
    print(f"Ollama {versao}; {MODELO}{' (sem pensar)' if sem_pensar else ''}, com o pedido da forma {FORMA} da"
          f" bancada")
    os.makedirs(PASTA, exist_ok=True)
    trazer_as_da_bancada()
    feitas = {a for a, r in registros_de_relacoes().items() if not r.get("erro")}
    for k in partes:
        with open(grafo.arquivo_conceitos(k), encoding="utf-8") as f:
            itens = json.load(f)["notas"]
        pendentes = [item for item in itens if item["arquivo"] not in feitas]
        atualizar_lista_da_parte(k, itens)
        print(f"\n== Parte {k}: {len(pendentes)} de {len(itens)} notas para ligar"
              f" (lista em {os.path.relpath(arquivo_da_parte(k), RAIZ)})")
        if not pendentes:
            continue
        bancada.carregar_o_modelo(MODELO, host, sem_pensar)
        comeco_da_parte = time.time()
        for numero, item in enumerate(pendentes, 1):
            if os.path.exists(ARQUIVO_PAUSA):
                print(f"\nPausado. Para continuar, apague {os.path.relpath(ARQUIVO_PAUSA, RAIZ)} e rode a parte ligar"
                      f" de novo: ela continua de onde parou.")
                return
            comeco, registro = time.time(), {"nota": item["arquivo"], "parte": k}
            try:
                registro.update(bancada.ligar_nota(bancada.nota_marcada(item), MODELO, host, sem_pensar, COM_EXEMPLOS))
            except (OSError, ValueError) as erro:  # sem conexão, erro do servidor ou resposta que não é JSON
                registro.update(relacoes=[], erro=repr(erro))
            registro.update(segundos=round(time.time() - comeco, 1), quando=time.strftime("%Y-%m-%d %H:%M:%S"))
            gravar([registro])
            atualizar_lista_da_parte(k, itens)
            restante = (time.time() - comeco_da_parte) / numero * (len(pendentes) - numero)
            print(f"   [{numero}/{len(pendentes)}] {item['arquivo']}: {len(registro['relacoes'])} relações"
                  f" ({registro['segundos']} s; a parte termina por volta de"
                  f" {time.strftime('%d/%m %H:%M', time.localtime(time.time() + restante))})"
                  + (f" ERRO: {registro['erro']}" if registro.get("erro") else ""))
        if montar_cada_parte and parte_pronta(itens):
            print(f"\n== Parte {k} pronta: montando os grafos")
            try:
                montar()
            except Exception as erro:  # um erro ao montar não para a ligação das outras partes
                print(f"   Erro ao montar: {erro!r}")
    print(f"\nRelações em {os.path.relpath(SAIDA_RELACOES, RAIZ)}")


def notas_da_parte(k):
    with open(grafo.arquivo_conceitos(k), encoding="utf-8") as f:
        return json.load(f)["notas"]


def pasta_da_parte(k):
    return os.path.join(PASTA, f"parte{k}")


def sem_literais(tripla):
    return not {tripla[0].rsplit("|", 1)[1], tripla[2].rsplit("|", 1)[1]} & ligacoes.LITERAIS


def montar_grafo(notas, registros, pasta):
    """Os grafos completo e filtrado das notas ({arquivo: nota marcada}), como no 5d, mas sem as ligações que tocam
    valores ou tempos. Grava os dois grafos, as entidades e a lista das notas na pasta e devolve as triplas de cada
    grafo, com as notas em que cada uma aparece."""
    triplas = {"completo": collections.defaultdict(set), "filtrado": collections.defaultdict(set)}
    formas, notas_da_entidade = collections.defaultdict(collections.Counter), collections.defaultdict(set)
    for arquivo, n in sorted(notas.items()):
        conceitos = {piloto.normalizar(c["texto"]): c["categoria"] for c in n["lista"]}
        completo, filtrado, soltas = grafo.triplas_da_nota(registros[arquivo]["relacoes"], conceitos)
        for nome, da_nota in (("completo", completo), ("filtrado", filtrado)):
            for t in filter(sem_literais, da_nota):
                triplas[nome][t].add(arquivo)
        for chave, texto in [(f"{piloto.normalizar(c['texto'])}|{c['categoria']}", c["texto"]) for c in n["lista"]
                             if c["categoria"] not in ligacoes.LITERAIS] + list(soltas.items()):
            formas[chave][texto] += 1
            notas_da_entidade[chave].add(arquivo)
    os.makedirs(pasta, exist_ok=True)
    for nome, t in triplas.items():
        with open(os.path.join(pasta, f"grafo_{nome}.tsv"), "w", encoding="utf-8", newline="\n") as f:
            f.writelines(f"{origem}\t{tipo}\t{destino}\n" for origem, tipo, destino in sorted(t))
    vizinhos = {nome: grafo.vizinhos_de(t) for nome, t in triplas.items()}
    grafo.salvar({chave: {"texto": formas[chave].most_common(1)[0][0], "categoria": chave.rsplit("|", 1)[1],
                          "ocorrencias": len(ns), "notas": sorted(ns),
                          **{f"vizinhos_{nome}": len(vizinhos[nome].get(chave, ())) for nome in vizinhos}}
                  for chave, ns in sorted(notas_da_entidade.items())}, os.path.join(pasta, "entidades.json"))
    with open(os.path.join(pasta, "notas.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines(a + "\n" for a in sorted(notas))
    return triplas


def medir_notas(notas, registros, registros_5d, fora):
    """As medidas do qwen3.5:27b (5g) e do qwen2.5:7b (5d) nas mesmas notas, sem as notas de fora."""
    avaliadas = [{**n, "relacoes_entre_conceitos": ligacoes.relacoes_entre_conceitos(n["caminho"])}
                 for a, n in sorted(notas.items()) if a not in fora]
    return {"notas": len(avaliadas),
            **{nome: ligacoes.medir(avaliadas, {n["arquivo"]: ligacoes.do_qwen(n, regs[n["arquivo"]]) for n in avaliadas})
               for nome, regs in (("qwen3.5:27b (5g)", registros), ("qwen2.5:7b (5d)", registros_5d))}}


def grafos_para_comparar(notas, triplas, registros_5d):
    """O tamanho e as ligações dos grafos do 5g, do grafo completo do 5d e do grafo das anotações, nas mesmas notas
    (no 5d, só entre conceitos, sem os tipos de valor, dose e tempo)."""
    do_5d = collections.defaultdict(set)
    for arquivo, n in notas.items():
        conceitos = {piloto.normalizar(c["texto"]): c["categoria"] for c in n["lista"]}
        for t in grafo.triplas_da_nota(registros_5d[arquivo]["relacoes"], conceitos)[0]:
            if sem_literais(t) and t[1] not in ligacoes.TIPOS_DE_LITERAIS:
                do_5d[t].add(arquivo)
    anotacoes = grafo.grafo_das_anotacoes([n["caminho"] for n in notas.values()], grafo.chave_do_texto)
    return {"5g, completo": grafo.estatisticas(triplas["completo"]),
            "5g, filtrado": grafo.estatisticas(triplas["filtrado"]),
            "5d, completo, entre conceitos": grafo.estatisticas(do_5d),
            "anotações": grafo.estatisticas(anotacoes)}


def escrever_leia_me(situacao, prontas):
    linhas = [
        "# Experimento 5g: o grafo do texto ligado pelo qwen3.5:27b", "",
        f"Refeito por `python -X utf8 experimentos/05g_grafo_do_texto_27b.py montar` em "
        f"{time.strftime('%d/%m/%Y %H:%M')}. O script explica o método no começo do arquivo.", "",
        "As 998 notas do SemClinBr estão nas mesmas 5 partes do experimento 5d. Os conceitos de cada nota vêm do",
        "modelo de conceitos do 5d que não viu a parte dela. As ligações vêm do qwen3.5:27b, com o pedido da forma 4",
        "da bancada (5f): os 5 tipos (trata, tem_resultado, localizado_em, nega, associado_a) e duas notas de exemplo.",
        "", "## Situação das partes", "",
        "| Parte | Notas | Ligadas | Pronta | Grafo da parte |", "|---|---|---|---|---|"]
    linhas += [f"| {k} | {s['notas']} | {s['ligadas']} | {'sim' if k in prontas else 'não'} | "
               f"{f'`parte{k}/`' if k in prontas else '—'} |" for k, s in situacao.items()]
    linhas += [
        "", f"O grafo de todas as partes prontas juntas fica em `prontas/` (parte{'s' if len(prontas) > 1 else ''}"
            f" {', '.join(map(str, prontas))}).",
        "", "## Arquivos", "",
        "- `relacoes.jsonl`: uma linha por nota ligada, com o arquivo .xml (`nota`), a `parte`, as `relacoes`",
        "  (origem, relacao, destino), as respostas cruas do modelo, os `segundos` e `quando` ela foi ligada. As 50",
        "  notas trazidas da bancada têm `de`. Se uma nota aparece duas vezes, vale a última linha.",
        "- `parte<k>_notas.csv`: os arquivos .xml da parte k e a situação de cada um (ligada, trazida da bancada,",
        "  com erro ou pendente). É atualizado a cada nota enquanto a parte é ligada; abre no Excel.",
        "- `parte<k>/` e `prontas/`, um grafo cada:",
        "  - `notas.txt`: os arquivos .xml que entraram no grafo;",
        "  - `grafo_completo.tsv`: todas as ligações entre conceitos, uma tripla por linha (origem, tipo, destino),",
        "    separadas por tabulação, sem cabeçalho;",
        "  - `grafo_filtrado.tsv`: só as ligações que respeitam as regras de categoria do experimento 5b;",
        "  - `entidades.json`: cada entidade (`texto normalizado|categoria`) com o texto mais comum, a categoria,",
        "    as notas em que aparece e quantos vizinhos tem em cada grafo.",
        "- `../experimento5g_resultados.json`: as medidas contra as anotações e o tamanho dos grafos.",
        "", "## Como pausar e continuar", "",
        "- Para pausar a ligação das notas, crie um arquivo vazio chamado `PAUSAR` nesta pasta: ela para depois da",
        "  nota em que está.",
        "- Para continuar, apague o `PAUSAR` e rode",
        "  `python -X utf8 experimentos/05g_grafo_do_texto_27b.py ligar --montar` (no Windows, com o Ollama aberto).",
        "  Ela continua de onde parou e monta os grafos cada vez que uma parte fica pronta.",
        "", "## Para usar no PyKEEN", "", "```python", "from pykeen.triples import TriplesFactory",
        "triplas = TriplesFactory.from_path(\"results/experimento5g/prontas/grafo_completo.tsv\")", "```",
        "", "## Observações", "",
        "- Os grafos não têm as ligações que tocam valores ou tempos (os tipos tem_dose, tem_valor e tem_tempo",
        "  saíram). O `entidades.json` tem todos os conceitos das listas, fora os valores e os tempos, mesmo os que",
        "  não entraram em nenhuma ligação: esses têm 0 vizinhos.",
        "- As medidas não contam as 20 notas do piloto, que serviram para ajustar os pedidos, nem as 2 notas de",
        "  exemplo (9596.xml e 9792.xml), cuja resposta vai no próprio pedido. Os grafos têm todas as notas.", ""]
    with open(os.path.join(PASTA, "LEIA-ME.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(linhas))


def montar():
    registros, registros_5d = registros_de_relacoes(), grafo.registros_de_relacoes()
    fora = {os.path.basename(a) for a in grafo.ner.divisao()[2]} | set(bancada.EXEMPLOS)
    partes, situacao, prontas = {k: notas_da_parte(k) for k in range(1, grafo.PARTES + 1)}, {}, []
    print("== Partes")
    for k, itens in partes.items():
        ligadas = [i["arquivo"] for i in itens if i["arquivo"] in registros and not registros[i["arquivo"]].get("erro")]
        situacao[k] = {"notas": len(itens), "ligadas": len(ligadas),
                       "horas": sum(registros[a].get("segundos", 0) for a in ligadas) / 3600}
        if len(ligadas) == len(itens):
            prontas.append(k)
        print(f"   parte {k}: {len(ligadas)} de {len(itens)} notas ligadas, {situacao[k]['horas']:.1f} horas"
              + (" (pronta)" if k in prontas else ""))
    if not prontas:
        sys.exit("Nenhuma parte pronta ainda: o grafo só é montado com partes inteiras.")
    medidas, grafos, todas = {}, {}, {}
    for k in prontas:
        notas = {i["arquivo"]: bancada.nota_marcada(i) for i in partes[k]}
        todas.update(notas)
        triplas = montar_grafo(notas, registros, pasta_da_parte(k))
        medidas[f"parte {k}"] = medir_notas(notas, registros, registros_5d, fora)
        grafos[f"parte {k}"] = grafos_para_comparar(notas, triplas, registros_5d)
    juntas = f"partes {', '.join(map(str, prontas))} juntas"
    if len(prontas) > 1:
        medidas[juntas] = medir_notas(todas, registros, registros_5d, fora)
    grafos[juntas] = grafos_para_comparar(todas, montar_grafo(todas, registros, PASTA_PRONTAS), registros_5d)
    grafo.salvar({"modelo": MODELO, "forma_da_bancada": FORMA, "partes": situacao, "prontas": prontas,
                  "fora_das_medidas": sorted(fora), "medidas": medidas, "grafos": grafos}, SAIDA_JSON)
    escrever_leia_me(situacao, prontas)

    pct = piloto.pct
    print("\n== Ligações entre conceitos, contra as anotações (sem as notas do piloto e as de exemplo)")
    print(f"   {'notas':28s}{'modelo':20s}{'ligações/nota':>14s}{'encontradas':>13s}{'confirmadas':>13s}"
          f"{'negações':>10s}{'trata/nota':>12s}")
    for grupo, m in medidas.items():
        rotulo = f"{grupo} ({m['notas']})"
        for nome in ("qwen3.5:27b (5g)", "qwen2.5:7b (5d)"):
            r = m[nome]["entre conceitos"]
            print(f"   {rotulo:28s}{nome:20s}"
                  f"{r['relacoes_por_nota']:14.1f}{pct(r['relacoes_anotadas_achadas_parcial']):>13s}"
                  f"{pct(None if r['relacoes_novas'] is None else 1 - r['relacoes_novas']):>13s}"
                  f"{pct(r['negacoes_achadas_parcial']):>10s}{m[nome]['trata_por_nota']:12.1f}")
    print("\n== Grafos (nas mesmas notas)")
    print(f"   {'notas':28s}{'grafo':32s}{'entidades':>10s}{'triplas':>9s}{'vizinhos':>9s}{'um vizinho':>11s}"
          f"{'maior parte ligada':>19s}")
    for grupo, gs in grafos.items():
        for nome, g in gs.items():
            print(f"   {grupo:28s}{nome:32s}{g['entidades']:10d}{g['triplas']:9d}{g['vizinhos_media']:9.1f}"
                  f"{pct(g['com_um_vizinho']):>11s}{pct(g['maior_componente']):>19s}")
    print(f"\nGrafos em {os.path.relpath(PASTA, RAIZ)} (veja o LEIA-ME.md); medidas em"
          f" {os.path.relpath(SAIDA_JSON, RAIZ)}")


def main():
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    argumentos.add_argument("parte", choices=("ligar", "montar"))
    argumentos.add_argument("partes", nargs="*", type=int, choices=range(1, grafo.PARTES + 1),
                            help="as partes das notas para ligar, de 1 a 5 (sem nenhuma, todas, na ordem)")
    argumentos.add_argument("--montar", action="store_true",
                            help="na parte ligar, monta os grafos cada vez que uma parte fica pronta")
    argumentos.add_argument("--host", default="http://127.0.0.1:11434", help="endereço do Ollama (parte ligar)")
    args = argumentos.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if args.parte == "ligar":
        ligar(args.partes or range(1, grafo.PARTES + 1), args.host, args.montar)
    else:
        montar()


if __name__ == "__main__":
    main()
