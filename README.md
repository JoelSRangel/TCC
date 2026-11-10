# Representação vetorial em grafos de conhecimento clínicos

Código do TCC "Avaliação de Técnicas de Representação Vetorial em Grafos de Conhecimento Clínicos para Apoio a Sistemas Inteligentes de Saúde".

O projeto monta grafos de conhecimento a partir do SemClinBr, um corpus de prontuários em português anotados por especialistas. Depois compara técnicas de embedding de grafos: TransE, ComplEx, RotatE e, de forma exploratória, Node2Vec. São dois grafos: o das anotações dos especialistas e o que uma LLM local (qwen3.5:27b) extraiu do texto dos prontuários.

## Usos de TransE, RotatE e ComplEx em um grafo de conhecimento

A tabela lista as formas de usar essas técnicas em qualquer grafo de conhecimento e o que este projeto já testou. As regras sem embedding usadas como comparação são a popularidade (quantas ligações cada conceito tem) e os vizinhos em comum.

| # | Uso | O que o modelo faz | Neste projeto |
|---|---|---|---|
| 1 | Completar o grafo (*link prediction*) | Dado (losartana, trata, ?), ordena os conceitos mais prováveis. É o uso para o qual as três técnicas foram criadas | Experimentos 1 e 6: no grafo das anotações, o MRR do RotatE foi 1,47 vez o da popularidade; no grafo da LLM, ficou no nível dela |
| 2 | Julgar uma ligação nova (*triple classification*) | Dá nota a uma ligação e a aceita ou rejeita | Experimento 8: com o TransE, 58,5% de chance de uma ligação certa ter nota maior que uma errada (o acaso é 50%). Sinal fraco, mas acima das regras simples |
| 3 | Achar erros no grafo | As ligações já existentes com nota mais baixa vão para revisão | Experimento 9: no grafo da LLM, nas ligações que o treino não usou, a melhor técnica (TransE) teve AUC de 0,588. Contar em quantos prontuários a ligação aparece deu 0,666. Nas ligações com que treinou, o modelo dá nota máxima a quase todas e não separa nada |
| 4 | Juntar duplicatas | Vetores muito próximos podem ser o mesmo conceito ("IAM" e "infarto agudo do miocárdio") | Experimento 9: a duplicata aparece entre os 10 vetores mais parecidos em 27,0% dos casos (RotatE). Ordenar pelos vizinhos em comum dá 29,6%, e ao acaso seria 0,5% |
| 5 | Ligar a outra base (*entity alignment*) | Liga os conceitos do grafo a uma terminologia ou a outro grafo (CID-10, SNOMED CT) | Não testado |
| 6 | Prever a categoria | Diz pelo vetor se o conceito é doença, remédio, exame... | Experimentos 3 e 9: as 85 categorias não formam grupos nos vetores (silhueta de −0,40 a −0,17), mas um classificador acerta o grupo de 51,2% dos conceitos com os vetores do RotatE. Com a lista dos vizinhos, sem vetores, acerta 51,3%; o piso é 19,2% |
| 7 | Achar conceitos parecidos | Busca os vetores mais próximos | Experimentos 3b e 4: o vizinho mais próximo costuma ser um conceito com as mesmas ligações (40% no RotatE) ou ligado ao próprio conceito (55% no TransE), e não um conceito clinicamente parecido. Na busca do RAG (experimento 4b), os conceitos mais parecidos recuperaram de 4,1% a 8,4%, conforme a técnica, contra 7,0% da melhor busca sem vetores |
| 8 | Buscar contexto para uma LLM (RAG) | Acrescenta à busca as ligações prováveis que faltam no grafo | Experimentos 4 e 4b: com as ligações prováveis do RotatE, a resposta escondida apareceu entre as 5 sugestões em 10,4% das perguntas, contra 7,0% da melhor busca sem vetores (3,5 pontos a mais; intervalo de 95% de 0,9 a 6,1). O TransE deu 8,7%, e o ComplEx, 7,1% |
| 9 | Responder perguntas sobre o grafo | Põe pergunta e resposta no mesmo espaço de vetores (o EmbedKGQA usa ComplEx) | Não testado dessa forma |
| 10 | Consultas de vários passos | Exemplo: "remédios que tratam doenças associadas à dispneia" | Não testado |
| 11 | Recomendar | Sugere diagnóstico, exame ou remédio a partir do que o paciente tem | Experimento 7: com o RotatE, pelo menos uma doença anotada entre as 10 sugestões em 48,1% dos prontuários. Contar as doenças que aparecem junto nos prontuários deu 58,9% |
| 12 | Achar remédios para outras doenças (*drug repurposing*) | Prevê "remédio trata doença" em grafos biomédicos grandes, com genes e proteínas | Fora do alcance: o grafo não tem esse conhecimento |
| 13 | Representar pacientes | Faz a média dos vetores dos conceitos do prontuário, para achar pacientes parecidos ou prever desfecho | Não testado. O SemClinBr não tem desfecho |
| 14 | Dar conhecimento a outro modelo | Os vetores entram em outro modelo, como um modelo de linguagem (o ERNIE fez isso com TransE) | Não testado |
| 15 | Visualizar e agrupar | Mapa em 2D e grupos dos vetores: remédios perto de remédios, doenças perto de doenças | Experimento 3: o K-means achou grupos fracos (silhueta de no máximo 0,24), que não batem com as categorias. A figura 4 do texto (PCA do Node2Vec) é desse tipo |
| 16 | Estudar as relações | O RotatE aprende se uma relação é simétrica, inversa de outra ou composta | Não testado |

Os números vêm dos scripts em `experimentos/`. Os resultados de cada script ficam em `results/`, fora do GitHub, porque contêm texto do corpus.

## Experimentos

Cada script explica no começo do arquivo o que faz e como rodar.

| Script | O que faz |
|---|---|
| `01_retreino_kge.py` | Experimento 1: retreino de TransE, ComplEx e RotatE com um regime adequado |
| `01b_complex.py` | Experimento 1b: por que o ComplEx não aprende neste grafo? |
| `01c_referencia_popularidade.py` | Experimento 1c: os modelos superam uma referência de popularidade? |
| `03_agrupamento.py` | Experimento 3: agrupamento com os vetores dos modelos retreinados |
| `03b_vizinhos_estrutura.py` | Experimento 3b: o que a entidade e seu vizinho mais próximo têm em comum no grafo? |
| `04_avaliacao_rag.py` | Experimento 4: o grafo e os vetores ajudam o modelo de linguagem a responder? |
| `04b_busca_com_as_quatro_tecnicas.py` | Experimento 4b: a busca do experimento 4, com as quatro técnicas |
| `05_piloto_grafo_do_texto.py` | Experimento 5 (piloto): montar o grafo a partir do texto das notas |
| `05b_piloto_com_categorias.py` | Experimento 5b (piloto): a versão com categorias |
| `05c_piloto_com_ner.py` | Experimento 5c (piloto): um modelo treinado para marcar os conceitos |
| `05d_grafo_do_texto.py` | Experimento 5d: o grafo do texto das 998 notas |
| `05d_conferir_relacoes.py` | Experimento 5d: conferência à mão de uma amostra das relações do grafo do texto |
| `05e_modelo_de_ligacoes.py` | Experimento 5e (teste pequeno): um modelo treinado com as anotações decide quais conceitos se ligam |
| `05f_bancada_de_extracao.py` | Experimento 5f: bancada de testes da extração das relações |
| `05g_grafo_do_texto_27b.py` | Experimento 5g: o grafo do texto das 998 notas, ligado pelo qwen3.5:27b |
| `05g_carregar_no_neo4j.py` | Carrega no Neo4j o grafo do experimento 5g, para ver e consultar pelo Neo4j Browser |
| `06_embeddings_grafo_5g.py` | Experimento 6: TransE, ComplEx e RotatE no grafo do texto (5g) |
| `07_sugestao_de_diagnosticos.py` | Experimento 7: sugerir diagnósticos para uma nota nova |
| `08_atualizacao_do_grafo.py` | Experimento 8: atualizar o grafo com prontuários novos |
| `08b_prontuarios_novos.py` | Experimento 8b: prontuários novos pelo fluxo inteiro (passo 10 do experimento 8) |
| `09_outros_usos_dos_vetores.py` | Experimento 9: outros usos dos vetores (prever a categoria, achar duplicatas e apontar ligações suspeitas) |
