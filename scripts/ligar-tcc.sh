#!/usr/bin/env bash
# Ativa o ambiente de desenvolvimento do TCC (WSL2 + conda + Neo4j via Docker Compose).
#
# Instalação (uma vez só): adicione esta linha ao seu ~/.bashrc do WSL2 (Ubuntu):
#   source /mnt/d/TCC/scripts/ligar-tcc.sh
# Depois é só abrir um novo terminal e digitar `ligar-tcc`.
#
# Se o projeto estiver em outro caminho, ajuste TCC_DIR abaixo.

TCC_DIR="/mnt/d/TCC"

ligar-tcc() {
    echo "[1/3] Ativando ambiente conda 'tcc_graphs'..."
    conda activate tcc_graphs || { echo "Ambiente 'tcc_graphs' não encontrado. Rode: conda env create -f $TCC_DIR/environment.yml"; return 1; }

    echo "[2/3] Subindo o Neo4j via docker compose..."
    (cd "$TCC_DIR" && docker compose up -d)

    echo "[3/3] Pronto."
    echo "Neo4j Browser: http://localhost:7474/browser/  (usuário: neo4j, senha: a que estiver no seu .env)"
    echo "Para checar o status do container: docker ps"
    echo "Para abrir o projeto no VS Code: cd $TCC_DIR && code ."
}

desligar-tcc() {
    echo "Parando o Neo4j..."
    (cd "$TCC_DIR" && docker compose stop)
    conda deactivate
    echo "Neo4j parado e ambiente conda desativado."
}
