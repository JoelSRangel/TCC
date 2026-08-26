import json
import os

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from src.graph_stats import GraphStatsAnalyzer

# Rótulos legíveis para o relatório impresso no console.
LABELS = {
    "total_nos": "Total de nós (conceitos únicos)",
    "total_arestas": "Total de arestas",
    "nos_isolados": "Nós isolados (sem nenhuma relação)",
    "num_componentes_conexos": "Número de componentes conexos",
    "tamanho_maior_componente": "Tamanho do maior componente conexo",
    "pct_nos_no_maior_componente": "% dos nós dentro do maior componente",
    "grau_medio": "Grau médio",
    "grau_mediano": "Grau mediano",
    "grau_min": "Grau mínimo",
    "grau_max": "Grau máximo",
    "grau_desvio_padrao": "Desvio padrão do grau",
}


def main():
    print("=== ESTATÍSTICAS ESTRUTURAIS DO GRAFO DE CONHECIMENTO CLÍNICO ===\n")

    analyzer = GraphStatsAnalyzer(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        stats = analyzer.compute_stats()
    finally:
        analyzer.close()

    for chave, valor in stats.items():
        rotulo = LABELS.get(chave, chave)
        print(f"{rotulo}: {valor}")

    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", "estatisticas_grafo.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[SUCESSO] Estatísticas salvas em: {output_path}")


if __name__ == "__main__":
    main()
