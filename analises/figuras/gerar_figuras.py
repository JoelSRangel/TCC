"""
Gera as figuras do texto do TCC que foram refeitas na revisão.

  fig1_contextualizacao  (introdução, Figura de visão geral): texto clínico ->
      extração -> grafo -> vetores, com o exemplo do texto da introdução e
      os dois tipos de relação do SemClinBr (associated_with e negation_of).
  fig2_intuicao          (cap. 2): grafo com Febre, Tosse, Influenza e Fratura
      e o espaço vetorial correspondente.
  fig3_pipeline          (cap. 3): pipeline do trabalho, com o Node2Vec
      (análise exploratória) e os modelos KGE (comparação quantitativa) em
      ramos separados, e o protótipo de RAG.
  fig4_pca_node2vec      (cap. 4): a figura original da análise exploratória
      (results/clusters_semclinbr.png) sem a legenda e sem o título. A legenda
      tinha cores repetidas entre categorias; os pontos não são alterados.

As figuras 1 e 2 são ilustrações: as posições dos pontos no espaço vetorial
foram escolhidas à mão para mostrar a ideia, não vêm de um modelo treinado.

Precisa do matplotlib, então roda no ambiente conda do projeto (no WSL):
    conda activate tcc_graphs
    python analises/figuras/gerar_figuras.py
Os arquivos (.pdf e .png) são salvos nesta mesma pasta.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageChops

PASTA = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(os.path.dirname(PASTA))
LARGURA = 6.3  # polegadas, aproximadamente a largura do texto na página (16 cm)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42})

# Uma cor por categoria, usadas nas figuras 1 e 2.
COR = {
    "sintoma": "#2b6cb0",
    "qualificador": "#6b46c1",
    "negacao": "#4a5568",
    "doenca": "#c53030",
    "procedimento": "#2f855a",
}
CINZA = "#4a5568"


def nova_figura(altura):
    fig = plt.figure(figsize=(LARGURA, altura))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, LARGURA)
    ax.set_ylim(0, altura)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def caixa(ax, x, y, w, h, borda=CINZA, fundo="#f7fafc", tracejada=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                linewidth=0.8, edgecolor=borda, facecolor=fundo,
                                linestyle="--" if tracejada else "-"))


def seta(ax, origem, destino, cor=CINZA, curva=0.0, largura=0.9):
    ax.add_patch(FancyArrowPatch(origem, destino, arrowstyle="-|>", mutation_scale=7,
                                 linewidth=largura, color=cor, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={curva}"))


def no(ax, x, y, rotulo, cor, raio=0.11, abaixo=True, tamanho=6.5):
    ax.add_patch(Circle((x, y), raio, facecolor=cor, edgecolor="white", linewidth=0.8, zorder=3))
    dy = -(raio + 0.06) if abaixo else raio + 0.06
    ax.text(x, y + dy, rotulo, ha="center", va="top" if abaixo else "bottom",
            fontsize=tamanho, color=cor, fontweight="bold", zorder=4)


def aresta(ax, a, b, rotulo, cor=CINZA, deslocamento=(0.0, 0.07), tamanho=6):
    seta(ax, a, b, cor=cor)
    mx, my = (a[0] + b[0]) / 2 + deslocamento[0], (a[1] + b[1]) / 2 + deslocamento[1]
    ax.text(mx, my, rotulo, ha="center", va="center", fontsize=tamanho, color=cor, style="italic",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none"))


def titulo_painel(ax, x, y, w, texto):
    ax.text(x + w / 2, y, texto, ha="center", va="top", fontsize=8, fontweight="bold")


def salvar(fig, nome):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(PASTA, f"{nome}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"   {nome}.pdf e {nome}.png")


def figura_1():
    """Texto -> extração -> grafo -> vetores, com o exemplo da introdução."""
    altura = 3.15
    fig, ax = nova_figura(altura)
    y0, h, gap = 0.1, 2.95, 0.2
    larguras = [1.16, 1.72, 1.42, 1.24]
    xs = [0.08]
    for w in larguras[:-1]:
        xs.append(xs[-1] + w + gap)
    titulos = ["1. Texto\nclínico", "2. Extração\n(PLN)", "3. Grafo de\nconhecimento", "4. Representação\nvetorial"]
    for x, w, t in zip(xs, larguras, titulos):
        caixa(ax, x, y0, w, h)
        ax.text(x + w / 2, y0 + h - 0.08, t, ha="center", va="top", fontsize=7.4, fontweight="bold",
                linespacing=1.2)
    for x, w in zip(xs[:3], larguras[:3]):
        seta(ax, (x + w + 0.03, y0 + h / 2), (x + w + gap - 0.03, y0 + h / 2))
    topo = y0 + h - 0.5  # abaixo dos títulos de duas linhas

    # 1. Texto
    frase = ("Paciente apresenta\ndor lombar intensa,\nsem sinais de\ninfecção, sendo\n"
             "recomendado\ntratamento\nfisioterapêutico.")
    ax.text(xs[0] + larguras[0] / 2, (y0 + topo) / 2, frase, ha="center", va="center", fontsize=7.2,
            style="italic", linespacing=1.5)

    # 2. Entidades (com a categoria na mesma linha) e relações extraídas
    x, w = xs[1] + 0.09, larguras[1]
    entidades = [("DOR LOMBAR", "sintoma"), ("INTENSA", "qualificador"), ("SEM SINAIS DE", "negacao"),
                 ("INFECÇÃO", "doenca"), ("TRATAMENTO\nFISIOTERAPÊUTICO", "procedimento")]
    nomes_cat = {"sintoma": "sintoma", "qualificador": "qualificador", "negacao": "negação",
                 "doenca": "doença", "procedimento": "procedimento"}
    ax.text(x, topo, "Entidades", fontsize=7, fontweight="bold", va="top")
    y = topo - 0.21
    for texto, cat in entidades:
        ax.text(x, y, texto, fontsize=6.3, color=COR[cat], fontweight="bold", va="top", linespacing=1.15)
        ax.text(xs[1] + w - 0.09, y, nomes_cat[cat], fontsize=5.8, color=COR[cat], ha="right", va="top",
                style="italic")
        y -= 0.17 + 0.1 * texto.count("\n")
    y -= 0.08
    ax.text(x, y, "Relações", fontsize=7, fontweight="bold", va="top")
    ax.text(x, y - 0.21, "INTENSA → DOR LOMBAR\n" + "    " + "associated_with", fontsize=5.9, va="top",
            color=CINZA, linespacing=1.3)
    ax.text(x, y - 0.55, "SEM SINAIS DE → INFECÇÃO\n" + "    " + "negation_of", fontsize=5.9, va="top",
            color=CINZA, linespacing=1.3)

    # 3. Grafo: um nó por conceito; o procedimento não tem relação anotada
    x, w = xs[2], larguras[2]
    ra = 0.08
    linhas = {"topo": y0 + 2.0, "meio": y0 + 1.2}
    pos = {"INTENSA": (x + 0.34, linhas["topo"]), "DOR LOMBAR": (x + w - 0.34, linhas["topo"]),
           "SEM SINAIS DE": (x + 0.34, linhas["meio"]), "INFECÇÃO": (x + w - 0.34, linhas["meio"]),
           "TRATAMENTO\nFISIOTERAPÊUTICO": (x + w / 2, y0 + 0.62)}
    cores3 = {"DOR LOMBAR": COR["sintoma"], "INTENSA": COR["qualificador"],
              "SEM SINAIS DE": COR["negacao"], "INFECÇÃO": COR["doenca"],
              "TRATAMENTO\nFISIOTERAPÊUTICO": COR["procedimento"]}
    for origem, destino, rotulo in (("INTENSA", "DOR LOMBAR", "associated_with"),
                                    ("SEM SINAIS DE", "INFECÇÃO", "negation_of")):
        (xa, ya), (xb, yb) = pos[origem], pos[destino]
        seta(ax, (xa + ra + 0.02, ya), (xb - ra - 0.02, yb))
        ax.text((xa + xb) / 2, ya + 0.1, rotulo, ha="center", va="bottom", fontsize=5.6, color=CINZA,
                style="italic")
    for rotulo, (px, py) in pos.items():
        no(ax, px, py, rotulo, cores3[rotulo], raio=ra, tamanho=5.3)

    # 4. Vetores: conceitos ligados ficam próximos (posições ilustrativas)
    x, w = xs[3], larguras[3]
    ax.annotate("", xy=(x + w - 0.1, y0 + 0.25), xytext=(x + 0.13, y0 + 0.25),
                arrowprops=dict(arrowstyle="-|>", color="#a0aec0", linewidth=0.7))
    ax.annotate("", xy=(x + 0.13, topo - 0.05), xytext=(x + 0.13, y0 + 0.25),
                arrowprops=dict(arrowstyle="-|>", color="#a0aec0", linewidth=0.7))
    pontos = [("INTENSA", "qualificador", 0.3, 2.05), ("DOR LOMBAR", "sintoma", 0.4, 1.84),
              ("SEM SINAIS DE", "negacao", 0.3, 1.27), ("INFECÇÃO", "doenca", 0.4, 1.06)]
    for rotulo, cat, px, py in pontos:
        ax.add_patch(Circle((x + px, y0 + py), 0.042, facecolor=COR[cat], edgecolor="none", zorder=3))
        ax.text(x + px + 0.07, y0 + py, rotulo, fontsize=5.2, color=COR[cat], ha="left", va="center",
                fontweight="bold")
    px, py = 0.8, 0.5
    ax.add_patch(Circle((x + px, y0 + py), 0.042, facecolor=COR["procedimento"], edgecolor="none", zorder=3))
    ax.text(x + 0.66, y0 + py + 0.08, "TRATAMENTO\nFISIOTERAPÊUTICO", fontsize=5.2,
            color=COR["procedimento"], ha="center", va="bottom", fontweight="bold", linespacing=1.15)
    salvar(fig, "fig1_contextualizacao")


def figura_2():
    """Grafo pequeno e o espaço vetorial correspondente (ilustração)."""
    altura = 2.5
    fig, ax = nova_figura(altura)
    w, h, y0 = 2.6, 2.3, 0.1
    xg, xv = 0.08, LARGURA - 0.08 - w
    for x, t in ((xg, "Grafo de conhecimento"), (xv, "Espaço vetorial (baixa dimensão)")):
        caixa(ax, x, y0, w, h)
        titulo_painel(ax, x, y0 + h - 0.08, w, t)
    seta(ax, (xg + w + 0.08, y0 + h / 2), (xv - 0.08, y0 + h / 2), largura=1.4)
    ax.text(LARGURA / 2, y0 + h / 2 + 0.16, "Graph\nEmbedding", ha="center", va="bottom",
            fontsize=7.5, fontweight="bold")

    cores = {"Febre": "#2f855a", "Tosse": "#b7791f", "Influenza": "#2b6cb0", "Fratura": "#6b46c1"}
    pos = {"Febre": (xg + 0.55, y0 + 1.55), "Influenza": (xg + 1.65, y0 + 1.55),
           "Tosse": (xg + 1.1, y0 + 0.62), "Fratura": (xg + 2.25, y0 + 0.5)}
    for a, b, d in (("Febre", "Influenza", (0, 0.1)), ("Febre", "Tosse", (-0.28, 0.0)),
                    ("Influenza", "Tosse", (0.3, 0.0))):
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color=CINZA, linewidth=0.9, zorder=1)
        mx, my = (pos[a][0] + pos[b][0]) / 2 + d[0], (pos[a][1] + pos[b][1]) / 2 + d[1]
        ax.text(mx, my, "associada a", fontsize=6, style="italic", color=CINZA, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none"))
    for rotulo, (px, py) in pos.items():
        no(ax, px, py, rotulo, cores[rotulo], raio=0.12, tamanho=7)

    ax.annotate("", xy=(xv + w - 0.15, y0 + 0.3), xytext=(xv + 0.2, y0 + 0.3),
                arrowprops=dict(arrowstyle="-|>", color="#a0aec0", linewidth=0.7))
    ax.annotate("", xy=(xv + 0.2, y0 + h - 0.42), xytext=(xv + 0.2, y0 + 0.3),
                arrowprops=dict(arrowstyle="-|>", color="#a0aec0", linewidth=0.7))
    vetores = {"Febre": (0.75, 1.5), "Influenza": (1.08, 1.62), "Tosse": (0.98, 1.22), "Fratura": (1.9, 0.62)}
    for rotulo, (px, py) in vetores.items():
        ax.add_patch(Circle((xv + px, y0 + py), 0.06, facecolor=cores[rotulo], edgecolor="none", zorder=3))
        ax.text(xv + px + 0.09, y0 + py, rotulo, fontsize=7, color=cores[rotulo], va="center",
                fontweight="bold")
    salvar(fig, "fig2_intuicao")


def figura_3():
    """Pipeline: etapas comuns, os dois ramos de modelos e o protótipo de RAG."""
    altura = 6.15
    fig, ax = nova_figura(altura)

    def etapa(x, y, w, titulo, detalhe, tracejada=False, fundo="#f7fafc"):
        h = 0.5
        caixa(ax, x, y, w, h, fundo=fundo, tracejada=tracejada)
        ax.text(x + w / 2, y + h - 0.12, titulo, ha="center", va="top", fontsize=7.8, fontweight="bold")
        ax.text(x + w / 2, y + 0.1, detalhe, ha="center", va="bottom", fontsize=6.6, color=CINZA)
        return (x, y, w, h)

    cx, cw = 1.0, LARGURA - 2.0
    comuns = [
        ("1. Extração", "leitura das anotações e relações dos 998 arquivos XML do SemClinBr"),
        ("2. Transformação", "normalização do texto e unificação das menções em conceitos"),
        ("3. Construção do grafo", "nós = conceitos; arestas = relações associated_with e negation_of"),
        ("4. Ingestão no Neo4j", "MERGE de nós e arestas; grafo com 17.897 nós e 9.253 arestas"),
    ]
    y = altura - 0.62
    caixas = []
    for titulo, detalhe in comuns:
        caixas.append(etapa(cx, y, cw, titulo, detalhe))
        y -= 0.72
    for (x, yy, w, h), (_, y2, _, h2) in zip(caixas, caixas[1:]):
        seta(ax, (x + w / 2, yy - 0.02), (x + w / 2, y2 + h2 + 0.02))

    bw, xl, xr = 2.95, 0.1, LARGURA - 0.1 - 2.95
    # Títulos dos ramos ao lado das setas, para as setas não cortarem o texto.
    ax.text(xl + bw / 2 + 0.1, y + 0.4, "Análise exploratória", ha="left", fontsize=7.5,
            fontweight="bold", color="#2b6cb0")
    ax.text(xr + bw / 2 - 0.1, y + 0.4, "Comparação quantitativa", ha="right", fontsize=7.5,
            fontweight="bold", color="#c05621")
    esquerda = [etapa(xl, y - 0.25, bw, "5. Projeção no GDS",
                      "em memória, não direcionada e sem o tipo da relação", fundo="#ebf8ff"),
                etapa(xl, y - 0.97, bw, "6. Node2Vec (GDS)", "grafo de um único laudo (73 nós)", fundo="#ebf8ff"),
                etapa(xl, y - 1.69, bw, "7. Avaliação exploratória",
                      "vizinhos por similaridade de cosseno e PCA", fundo="#ebf8ff")]
    direita = [etapa(xr, y - 0.25, bw, "Triplas lidas do Neo4j",
                     "(origem, relação, destino), sem projeção", tracejada=True, fundo="#fffaf0"),
               etapa(xr, y - 0.97, bw, "6. TransE, ComplEx e RotatE",
                     "PyKEEN; 5 execuções por modelo", fundo="#fffaf0"),
               etapa(xr, y - 1.69, bw, "7. Avaliação quantitativa",
                     "predição de links e agrupamento", fundo="#fffaf0")]
    _, y4, _, _ = caixas[-1]
    # Setas retas, do fundo da etapa 4 até o topo de cada ramo.
    seta(ax, (xl + bw / 2, y4 - 0.02), (xl + bw / 2, y - 0.25 + 0.5 + 0.02))
    seta(ax, (xr + bw / 2, y4 - 0.02), (xr + bw / 2, y - 0.25 + 0.5 + 0.02))
    for ramo in (esquerda, direita):
        for (x, yy, w, h), (_, y2, _, h2) in zip(ramo, ramo[1:]):
            seta(ax, (x + w / 2, yy - 0.02), (x + w / 2, y2 + h2 + 0.02))

    rx, ry, rw = 1.0, y - 2.55, LARGURA - 2.0
    etapa(rx, ry, rw, "8. Protótipo de RAG",
          "vetores do RotatE + relações do grafo (Cypher) + modelo de linguagem local", fundo="#f0fff4")
    xq, yq, wq, _ = direita[-1]
    seta(ax, (xq + wq / 2, yq - 0.02), (rx + rw - 0.9, ry + 0.5 + 0.02))
    ax.text(xq + wq / 2 + 0.08, yq - 0.2, "RotatE", fontsize=6.5, color="#c05621", style="italic", va="center")
    salvar(fig, "fig3_pipeline")


def figura_4():
    """Recorta a legenda e o título da figura original, sem alterar os pontos."""
    original = os.path.join(RAIZ, "results", "clusters_semclinbr.png")
    imagem = Image.open(original)
    largura, altura = imagem.size
    # A figura original tem 3600 x 2400 pixels: o título ocupa o topo
    # e a legenda começa por volta de 49% da largura.
    recorte = imagem.crop((0, int(altura * 0.053), int(largura * 0.4955), altura)).convert("RGB")
    # Tira as margens brancas que sobram, deixando uma borda pequena.
    fundo = Image.new("RGB", recorte.size, (255, 255, 255))
    esq, topo, dir_, base = ImageChops.difference(recorte, fundo).getbbox()
    borda = 30
    recorte = recorte.crop((max(esq - borda, 0), max(topo - borda, 0),
                            min(dir_ + borda, recorte.width), min(base + borda, recorte.height)))
    recorte.save(os.path.join(PASTA, "fig4_pca_node2vec.png"), dpi=(300, 300))
    print(f"   fig4_pca_node2vec.png (recorte de {largura}x{altura} para {recorte.size[0]}x{recorte.size[1]})")


def main():
    print("Gerando figuras em", PASTA)
    figura_1()
    figura_2()
    figura_3()
    figura_4()


if __name__ == "__main__":
    main()
