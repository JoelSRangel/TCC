"""
Experimento 8b: prontuários novos pelo fluxo inteiro (passo 10 do experimento 8).

O experimento 8 mediu o fluxo numa simulação. Aqui ele é usado de verdade, em prontuários que nunca entraram em nenhum
grafo: as 2 notas de data/raw/teste e, se for pedido, qualquer nota em .txt.
  1. o modelo de vetores que foi melhor no filtro do experimento 8 é treinado com todas as anotações (o grafo inicial);
  2. o BioBERTpt do experimento 5d marca os conceitos do prontuário;
  3. o qwen3.5:27b propõe as ligações, com o mesmo pedido do 5g;
  4. cada ligação vai para um grupo: já está no grafo, conceito novo ou avaliável; as avaliáveis com nota acima do corte
     entram e as outras vão para revisão. O corte é a nota mediana das ligações avaliáveis do experimento 8, ou seja,
     o de manter as 50% melhores.

O script tem quatro partes, porque o treino e o BioBERTpt usam o PyTorch do WSL e o qwen3.5:27b usa o Ollama do Windows:
  treinar  (WSL)      treina o modelo final e calcula o corte
  marcar   (WSL)      marca os conceitos de cada prontuário
  ligar    (Windows)  manda cada prontuário ao qwen3.5:27b
  decidir  (WSL)      separa as ligações e grava o que entraria no grafo

Saídas (não versionadas; têm conceitos das notas), em results/experimento8/fluxo/:
  modelo_final.pt, modelo_final_mapas.json   o modelo final
  <prontuário>.json                          conceitos marcados e ligações propostas
  <prontuário>_decisoes.txt                  o que entra, o que vai para revisão e o que tem conceito novo
  resumo.json                                só contagens

Uso:
    # no WSL, no ambiente conda do projeto
    python experimentos/08b_prontuarios_novos.py treinar
    python experimentos/08b_prontuarios_novos.py marcar data/raw/teste/8907.xml data/raw/teste/8908.xml
    # no Windows, com o Ollama aberto
    python -X utf8 experimentos/08b_prontuarios_novos.py ligar
    # no WSL de novo
    python experimentos/08b_prontuarios_novos.py decidir
"""
import argparse
import glob
import importlib.util
import json
import os
import statistics
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA = os.path.join(RAIZ, "results", "experimento8", "fluxo")
TECNICAS = ("TransE", "ComplEx", "RotatE")


def carregar(nome, arquivo):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(RAIZ, "experimentos", arquivo))
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def ler_json(caminho):
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def gravar_json(dados, caminho):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


def prontuarios():
    return sorted(c for c in glob.glob(os.path.join(PASTA, "*.json")) if os.path.basename(c) not in
                  ("resumo.json", "modelo_final_mapas.json"))


def treinar():
    """O modelo final com todas as anotações, e o corte de nota vindo do experimento 8."""
    import torch

    atualizacao = carregar("atualizacao", "08_atualizacao_do_grafo.py")
    sugestao = atualizacao.sugestao
    resultados = ler_json(atualizacao.SAIDA_JSON)["tecnicas"]
    tecnica = max(TECNICAS, key=lambda t: resultados[t]["auc"])
    base, notas_por_tecnica = atualizacao.carregar_resultados(atualizacao.PASTA)
    notas_avaliaveis = [v for a, n in base.items() for l, v in zip(n["links"], notas_por_tecnica[tecnica][a]["links"])
                        if l["grupo"] == "avaliavel"]
    corte = statistics.median(notas_avaliaveis)

    notas = sugestao.ler_notas()
    triplas = set().union(*sugestao.triplas_das_anotacoes(notas).values())
    modelo, fabrica, info = sugestao.treinar_kge(triplas, atualizacao.BASES["anotacoes"]["grafo"], tecnica, 0, sugestao.retreino.MAX_EPOCAS)
    os.makedirs(PASTA, exist_ok=True)
    torch.save(modelo, os.path.join(PASTA, "modelo_final.pt"))
    gravar_json({"tecnica": tecnica, "corte": corte, "ligacoes": len(triplas), "entidades": fabrica.entity_to_id,
                 "relacoes": fabrica.relation_to_id, "triplas": sorted(triplas), "info": info},
                os.path.join(PASTA, "modelo_final_mapas.json"))
    print(f"{tecnica} treinado com {len(triplas)} ligações; corte de nota {corte:.4f}")


def marcar(arquivos):
    """Conceitos de cada prontuário, com o BioBERTpt. Grava também o texto, que a parte ligar precisa."""
    sugestao = carregar("sugestao", "07_sugestao_de_diagnosticos.py")
    ner, piloto = sugestao.ner, sugestao.piloto
    for arquivo in arquivos:
        xml = arquivo.endswith(".xml")
        if xml:
            texto, texto_do_pedido = ner.texto_cru(arquivo), piloto.ler_nota(arquivo)["texto"]
        else:
            with open(arquivo, encoding="utf-8") as f:
                texto = texto_do_pedido = f.read()
        spans = sugestao.marcar_conceitos(texto)
        lista = ner.lista_de_conceitos(texto, spans)
        nome = os.path.splitext(os.path.basename(arquivo))[0]
        gravar_json({"arquivo": os.path.basename(arquivo), "caminho": arquivo, "xml": xml, "texto": texto,
                     "texto_do_pedido": texto_do_pedido, "spans": spans, "lista": lista}, os.path.join(PASTA, f"{nome}.json"))
        print(f"{os.path.basename(arquivo)}: {len(lista)} conceitos marcados")


def ligar(host):
    """As ligações propostas pelo qwen3.5:27b, com o pedido do 5g. Só usa a biblioteca padrão."""
    g5 = carregar("g5", "05g_grafo_do_texto_27b.py")
    sem_pensar = g5.bancada.pensa(g5.MODELO, host)
    g5.bancada.carregar_o_modelo(g5.MODELO, host, sem_pensar)
    for caminho in prontuarios():
        dados = ler_json(caminho)
        if "relacoes" in dados:
            continue
        n = {"arquivo": dados["arquivo"], "texto": dados["texto"], "spans": [tuple(s) for s in dados["spans"]],
             "lista": dados["lista"], "nota": {"texto": dados["texto_do_pedido"]}}
        comeco = time.time()
        registro = g5.bancada.ligar_nota(n, g5.MODELO, host, sem_pensar, g5.COM_EXEMPLOS)
        dados.update(relacoes=registro["relacoes"], cortada=registro["cortada"], segundos=round(time.time() - comeco, 1))
        gravar_json(dados, caminho)
        print(f"{dados['arquivo']}: {len(dados['relacoes'])} ligações em {dados['segundos']} s")


def decidir():
    """O que entra no grafo, o que vai para revisão e o que tem conceito novo, para cada prontuário."""
    import torch

    atualizacao = carregar("atualizacao", "08_atualizacao_do_grafo.py")
    sugestao, piloto, grafo, ligacoes = atualizacao.sugestao, atualizacao.piloto, atualizacao.grafo, atualizacao.ligacoes
    casa, texto_da = atualizacao.casa, atualizacao.texto_da
    mapas = ler_json(os.path.join(PASTA, "modelo_final_mapas.json"))
    modelo = torch.load(os.path.join(PASTA, "modelo_final.pt"), weights_only=False)
    modelo.eval()

    class Fabrica:
        entity_to_id, relation_to_id = mapas["entidades"], mapas["relacoes"]

    nota = atualizacao.nota_kge(modelo, Fabrica)
    ligados, conhecidos = set(), set()
    for o, _, d in mapas["triplas"]:
        a, b = texto_da(o), texto_da(d)
        ligados |= {(a, b), (b, a)}
        conhecidos |= {a, b}

    resumo = {"tecnica": mapas["tecnica"], "corte": mapas["corte"], "prontuarios": {}}
    for caminho in prontuarios():
        dados = ler_json(caminho)
        lista = {piloto.normalizar(c["texto"]): c["categoria"] for c in dados["lista"]}
        triplas = sorted(t for t in grafo.triplas_da_nota(dados["relacoes"], lista)[0] if sugestao.g5.sem_literais(t))
        gabarito = atualizacao.pares_anotados(dados["caminho"]) if dados["xml"] else None
        grupos = {"entra no grafo": [], "vai para revisão": [], "já estava no grafo": [], "tem conceito novo": []}
        batem = dict.fromkeys(grupos, 0)
        for o_chave, relacao, d_chave in triplas:
            o, d = texto_da(o_chave), texto_da(d_chave)
            valor = None
            if (o, d) in ligados:
                grupo = "já estava no grafo"
            elif o in conhecidos and d in conhecidos:
                valor = nota(o, d, atualizacao.tipo_no_grafo(relacao, "anotacoes"))
                grupo = "entra no grafo" if valor >= mapas["corte"] else "vai para revisão"
            else:
                grupo = "tem conceito novo"
            certa = bool(gabarito) and bool(atualizacao.combina(o, d, gabarito))
            batem[grupo] += certa
            grupos[grupo].append(f"  {o} --{relacao}--> {d}" + (f"  (nota {valor:.2f})" if valor is not None else "")
                                 + ("  [bate com as anotações]" if certa else ""))

        linhas = [f"Prontuário: {dados['arquivo']}", f"Conceitos marcados pelo BioBERTpt: {len(dados['lista'])}",
                  f"Ligações propostas pela LLM: {len(triplas)} ({dados['segundos']} s)",
                  f"Modelo: {mapas['tecnica']}, treinado com {mapas['ligacoes']} ligações; corte de nota {mapas['corte']:.2f}"]
        for g, itens in grupos.items():
            linhas += ["", f"== {g.capitalize()}: {len(itens)}", *(itens or ["  (nenhuma)"])]
        nome = os.path.splitext(os.path.basename(caminho))[0]
        with open(os.path.join(PASTA, f"{nome}_decisoes.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(linhas) + "\n")
        resumo["prontuarios"][dados["arquivo"]] = {
            "conceitos": len(dados["lista"]), "ligacoes": len(triplas), "segundos_da_llm": dados["segundos"],
            "anotadas": len(gabarito) if gabarito is not None else None,
            "grupos": {g: {"ligacoes": len(i), "batem": batem[g] if gabarito is not None else None} for g, i in grupos.items()}}
        print(f"{dados['arquivo']}: " + "; ".join(f"{g}: {len(i)}" for g, i in grupos.items())
              + (f" | batem com as anotações: {sum(batem.values())} de {len(triplas)}" if gabarito is not None else ""))
    gravar_json(resumo, os.path.join(PASTA, "resumo.json"))
    print(f"Decisões em {os.path.relpath(PASTA, RAIZ)}")


def main():
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    partes = argumentos.add_subparsers(dest="parte", required=True)
    partes.add_parser("treinar", help="treina o modelo final e calcula o corte de nota (WSL)")
    m = partes.add_parser("marcar", help="marca os conceitos dos prontuários (WSL)")
    m.add_argument("arquivos", nargs="+", help="prontuários em .xml ou .txt")
    l = partes.add_parser("ligar", help="manda os prontuários ao qwen3.5:27b (Windows)")
    l.add_argument("--host", default="http://localhost:11434", help="endereço do Ollama")
    partes.add_parser("decidir", help="decide o que entra no grafo (WSL)")
    args = argumentos.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if args.parte == "treinar":
        treinar()
    elif args.parte == "marcar":
        marcar(args.arquivos)
    elif args.parte == "ligar":
        ligar(args.host)
    else:
        decidir()


if __name__ == "__main__":
    main()
