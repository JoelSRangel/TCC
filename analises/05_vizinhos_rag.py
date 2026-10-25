"""
Proximidade entre vizinhos nos vetores do RotatE usados no protótipo de RAG.

Para uma amostra de entidades, mede a similaridade de cosseno de cada uma
com o seu vizinho mais próximo, e faz a mesma medição em vetores sorteados
ao acaso, com o mesmo número de entidades e a mesma dimensão. Números
citados no cap. 4 (protótipo de RAG): mediana 0,328 nos vetores do RotatE,
0,325 nos vetores aleatórios, e nenhuma entidade com vizinho de
similaridade 0,5 ou mais.

Lê results/rag_embeddings.npz, gerado por treinar_embeddings_rag.py
(branch feature/rag_prototype). Os vetores já vêm normalizados (comprimento
1), então o produto interno é a similaridade de cosseno. O arquivo é lido
sem numpy para o script rodar em qualquer Python 3. As sementes são fixas,
então o resultado se repete a cada execução. Leva cerca de 20 segundos.

Uso (da raiz do repositório):
    python analises/05_vizinhos_rag.py [tamanho_da_amostra]
"""
import array
import ast
import math
import operator
import os
import random
import sys
import zipfile

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQUIVO = os.path.join(RAIZ, "results", "rag_embeddings.npz")
AMOSTRA = int(sys.argv[1]) if len(sys.argv) > 1 else 300
LIMIARES = (0.3, 0.4, 0.5)


def ler_matriz_npz(caminho, nome="embeddings.npy"):
    """Lê uma matriz float32/float64 salva por numpy.savez_compressed."""
    bruto = zipfile.ZipFile(caminho).read(nome)
    versao = bruto[6]
    tam_cabecalho = int.from_bytes(bruto[8:10] if versao == 1 else bruto[8:12], "little")
    inicio = 10 if versao == 1 else 12
    cabecalho = ast.literal_eval(bruto[inicio:inicio + tam_cabecalho].decode("latin1"))
    valores = array.array({"<f4": "f", "<f8": "d"}[cabecalho["descr"]])
    valores.frombytes(bruto[inicio + tam_cabecalho:])
    linhas, colunas = cabecalho["shape"]
    return [valores[i * colunas:(i + 1) * colunas].tolist() for i in range(linhas)]


def vizinho_mais_proximo(vetores, amostra):
    """Maior similaridade de cada entidade da amostra com qualquer outra."""
    maiores = []
    for i in amostra:
        v = vetores[i]
        maiores.append(max(sum(map(operator.mul, v, vetores[j])) for j in range(len(vetores)) if j != i))
    return sorted(maiores)


def resumo(nome, maiores):
    mediana = maiores[len(maiores) // 2]
    print(f"   {nome}: mediana {mediana:.3f} (mínimo {maiores[0]:.3f}, máximo {maiores[-1]:.3f})")
    for limiar in LIMIARES:
        fracao = 100 * sum(m >= limiar for m in maiores) / len(maiores)
        print(f"      com vizinho de similaridade {limiar} ou mais: {fracao:.1f}%")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    vetores = ler_matriz_npz(ARQUIVO)
    n, d = len(vetores), len(vetores[0])
    print(f"Vetores do RotatE: {n} entidades, {d} valores cada  [texto: 7.961 entidades, 128 valores]")
    print(f"Amostra: {AMOSTRA} entidades\n")

    random.seed(0)
    amostra = random.sample(range(n), AMOSTRA)
    print("1. Vetores do RotatE  [texto: mediana 0,328; nenhuma com vizinho de 0,5 ou mais]")
    resumo("RotatE", vizinho_mais_proximo(vetores, amostra))

    random.seed(1)
    aleatorios = []
    for _ in range(n):
        v = [random.gauss(0, 1) for _ in range(d)]
        norma = math.sqrt(sum(x * x for x in v))
        aleatorios.append([x / norma for x in v])
    amostra = random.sample(range(n), AMOSTRA)
    print("\n2. Vetores sorteados ao acaso, mesmo tamanho  [texto: mediana 0,325]")
    resumo("aleatórios", vizinho_mais_proximo(aleatorios, amostra))


if __name__ == "__main__":
    main()
