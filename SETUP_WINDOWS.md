# Configurando o ambiente no Windows 11 (pós-formatação)

Este projeto foi originalmente desenvolvido no Pop!_OS. Depois de trocar para
Windows 11, o ambiente precisa ser reconstruído — nada do Docker/Conda antigo
sobrevive a uma reinstalação do sistema, e (até esta atualização) o repositório
não versionava nem o `docker-compose.yml`, nem o `environment.yml`, nem a senha
do Neo4j fora do código-fonte. Este arquivo documenta como levantar tudo de novo.

Caminho recomendado: **WSL2 (Ubuntu)**, não PowerShell nativo — fica mais
próximo do fluxo que você já tinha no Pop!_OS, e pacotes como `pykeen`/`torch`
instalam com bem menos atrito no Linux do que no Windows puro.

## 1. WSL2 + Ubuntu

Se você ainda não tem uma distro Ubuntu no WSL2, abra o PowerShell **como
administrador** e rode:

```powershell
wsl --install -d Ubuntu
```

Reinicie se for pedido, e configure usuário/senha do Ubuntu na primeira
abertura. Se já tiver uma distro Ubuntu, pule este passo.

## 2. Integrar o Docker Desktop com o WSL2

Como o Docker Desktop já está instalado: abra **Docker Desktop → Settings →
Resources → WSL Integration**, e ative o toggle da sua distro Ubuntu. Sem
isso, o comando `docker` não vai existir dentro do terminal WSL2.

## 3. Miniconda dentro do WSL2

Abra um terminal Ubuntu (WSL2) e rode:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
# feche e reabra o terminal para o conda ficar disponível
```

## 4. Recriar o ambiente conda `tcc_graphs`

Dentro do WSL2, com o projeto acessível em `/mnt/d/TCC` (o Windows expõe o
drive D: automaticamente em `/mnt/d/` dentro do WSL2):

```bash
cd /mnt/d/TCC
conda env create -f environment.yml
conda activate tcc_graphs
```

As versões exatas dos pacotes usadas antes da formatação foram perdidas (não
havia `environment.yml` versionado). Depois de confirmar que `python main.py`
roda sem erro, rode `conda env export --from-history > environment.yml` e
comite o resultado, para travar as versões que efetivamente funcionaram.

## 5. Configurar a senha do Neo4j (fora do código)

```bash
cp .env.example .env
# edite .env e defina em NEO4J_PASSWORD a senha do seu Neo4j
```

O `.env` já está no `.gitignore` — não é enviado ao GitHub. `config.py` agora
lê a senha de lá em vez de ter o valor hardcoded no código-fonte.

## 6. Subir o Neo4j (Docker Compose)

```bash
docker compose up -d
docker ps        # confirma que o container "neo4j-tcc" está rodando
```

Isso recria o container do zero usando a imagem oficial `neo4j:5.12.0` (a
mesma versão citada no TCC-1) com o plugin GDS (Graph Data Science) já
habilitado via `NEO4J_PLUGINS`. Os dados ficam em um volume Docker nomeado
(`neo4j_data`), então sobrevivem a um `docker compose stop`/`start` normal —
mas **não existiam dados antigos para recuperar**: o container antigo era
local ao Pop!_OS e não foi (nem podia ter sido) migrado para o Windows. O
grafo vai ser reconstruído do zero na primeira execução do `main.py`, já
com a correção de unificação de conceitos.

Acesse http://localhost:7474/browser/ no navegador para confirmar (usuário
`neo4j`, senha do seu `.env`). Para conferir se o GDS está ativo, rode no
Neo4j Browser: `CALL gds.list() YIELD name RETURN name LIMIT 5;`

## 7. Restaurar o corpus SemClinBr

Copie seu backup dos arquivos XML para dentro de `data/raw/` na raiz do
projeto (a pasta é ignorada pelo git de propósito — dados sensíveis não vão
para o GitHub):

```bash
mkdir -p /mnt/d/TCC/data/raw
# copie os .xml do seu backup para essa pasta
```

## 8. Rodar o pipeline

```bash
cd /mnt/d/TCC
python main.py
```

## 9. Atalho para o dia a dia (equivalente ao antigo alias no Pop!_OS)

Adicione ao final do seu `~/.bashrc` (dentro do WSL2):

```bash
source /mnt/d/TCC/scripts/ligar-tcc.sh
```

Abra um novo terminal e use `ligar-tcc` (ativa o conda + sobe o Neo4j) e
`desligar-tcc` (para o Neo4j + desativa o conda) — ver `scripts/ligar-tcc.sh`
para o conteúdo exato.

## VS Code

Instale a extensão **WSL** no VS Code. Rodando `code .` de dentro do terminal
WSL2 (na pasta `/mnt/d/TCC`), o VS Code abre já conectado ao ambiente Linux,
reproduzindo o fluxo que você tinha no Pop!_OS.
