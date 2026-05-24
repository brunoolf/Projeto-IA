# Simulação de opinião pública sobre desigualdade no Brasil com LLMs

Projeto final da disciplina **Inteligência Artificial — 1º Semestre 2026 (7G)**, Universidade Presbiteriana Mackenzie, prof. Rogério de Oliveira.

## Autor

**Bruno Ferrão** — RA `10401081`

## Entregáveis

| Item | Caminho |
|---|---|
| Artigo (PDF, layout SBC) | [artigo/artigo.pdf](artigo/artigo.pdf) |
| Notebook executável | [notebook/simulacao.ipynb](notebook/simulacao.ipynb) |
| Dados originais da pesquisa CESOP/IPEC 04839 | [data/raw/](data/raw/) |
| Predições do LLM e métricas | [data/](data/) |
| Gráficos | [figs/](figs/) |
| Vídeo da apresentação (YouTube) | [youtu.be/y5dbSnP9GEg](https://youtu.be/y5dbSnP9GEg) |

## O que é

Peguei a pesquisa do CESOP/IPEC sobre como o brasileiro enxerga desigualdade (2.000 entrevistas presenciais feitas pelo IPEC em julho de 2023) e tentei reproduzir as respostas de uma amostra de 200 pessoas usando um LLM aberto pequeno, o Qwen 2.5 0.5B Instruct. Em seguida treinei um Random Forest com as mesmas variáveis sociodemográficas pra ter um termo de comparação. A ideia era ver até onde um modelo de meio bilhão de parâmetros, rodando em CPU, consegue simular opinião pública brasileira sobre temas sensíveis.

As duas afirmações simuladas, ambas em escala Likert de cinco pontos:

| Código | Afirmação |
|---|---|
| **P6A** | "A abordagem policial é baseada no tipo de cabelo, tipo de vestimenta e cor de pele das pessoas." |
| **P6C** | "Aumentar a representatividade de pessoas negras, mulheres e LGBTQIA+ na política e em cargos de poder contribui para diminuir as desigualdades estruturais." |

Sete variáveis preditoras: sexo, faixa etária, escolaridade, cor/raça, religião, renda familiar e região do país. 200 respondentes × 2 questões × 3 repetições estocásticas = 1.200 inferências.

## Resultados em uma frase

O Random Forest venceu em todas as métricas, mas isso não é o mais interessante. O LLM colapsou 51% das respostas de P6A e 67% das de P6C em "Discorda totalmente", enquanto a população real concentra suas respostas no extremo oposto. Dois modelos com vieses fortes, em direções contrárias.

### Tabela de métricas

| Questão | Modelo | Acurácia | MAE | KL(pred‖real) |
|---|---|---|---|---|
| P6A | LLM Qwen2.5-0.5B | 0,145 | 2,072 | 0,958 |
| P6A | Random Forest (5-CV) | **0,335 ± 0,034** | **1,460** | **0,216** |
| P6C | LLM Qwen2.5-0.5B | 0,130 | 2,373 | 1,301 |
| P6C | Random Forest (5-CV) | **0,440 ± 0,068** | **1,355** | **0,208** |

### Por que isso acontece

Três hipóteses, em ordem de plausibilidade:

1. **Alinhamento defensivo.** O Qwen, como qualquer modelo *instruction-tuned*, é treinado pra ser cauteloso em temas sensíveis. Diante de uma afirmação como "a polícia discrimina por cor", a saída segura é discordar. Isso explicaria o pico em "Discorda totalmente" e a ausência absoluta de "Concorda totalmente" nas duas questões.
2. **Capacidade.** 500 milhões de parâmetros é uma fração dos modelos usados na literatura de *silicon sampling*. A representação interna do brasileiro mediano pode simplesmente não estar lá.
3. **Cobertura de treino.** O corpus do Qwen é majoritariamente em inglês e chinês. O conhecimento detalhado de padrões de opinião brasileira sobre raça e gênero pode ser raso.

O Random Forest, por sua vez, atinge boa acurácia jogando seguro: prevê a moda da amostra (68% e 73% em "Concorda totalmente"). Acerta a direção geral, mas perde a variância da população.

### Variáveis mais importantes

Religião e faixa etária dividem o topo da feature importance do RF, cada uma com 17 a 20% do peso. Escolaridade vem em terceiro (15-16%), depois renda familiar e região. Cor/raça pesa 11%. Sexo é a variável menos informativa, com 7 a 8% — o que surpreende em afirmações sobre raça e gênero, mas é consistente com pesquisas que apontam idade, religião e escolaridade como divisores mais fortes da opinião brasileira do que o gênero do respondente.

## Como rodar

Funciona em CPU, em qualquer notebook recente. Sem GPU, sem chave de API, sem nada proprietário.

```bash
pip install pandas pyreadstat scikit-learn matplotlib seaborn transformers torch
jupyter notebook notebook/simulacao.ipynb
```

Ou abra o `simulacao.ipynb` direto no Google Colab. O notebook lê o arquivo SPSS original em `data/raw/04839.sav`, que está incluído no repositório.

Tempo de execução em CPU Intel típica: 10 a 15 minutos no total. Em Colab gratuito com GPU T4 cai pra cerca de 3 minutos.

## Tecnologias

- **LLM:** Qwen 2.5 0.5B Instruct (Alibaba, Apache 2.0)
- **Baseline:** Random Forest do scikit-learn com 200 árvores, profundidade máxima 8, validação cruzada estratificada em 5 folds
- **Dados:** arquivo SPSS original da pesquisa CESOP/IPEC 04839, incluído no repositório
- **Configuração:** `seed=42`, `temperature=0.8`, `top_p=0.9`

## Estrutura do repositório

```
.
├── README.md                  este arquivo
├── artigo/
│   └── artigo.pdf             artigo em PDF, layout SBC
├── notebook/
│   └── simulacao.ipynb        notebook executável de ponta a ponta
├── data/
│   ├── raw/                   dados originais da pesquisa
│   │   ├── 04839.sav          arquivo SPSS do CESOP/IPEC
│   │   ├── quest_04839.pdf    questionário aplicado
│   │   └── TF_04839.pdf       relatório técnico da pesquisa
│   ├── sample_200.csv         amostra estratificada usada no experimento
│   ├── llm_predictions.csv    1.200 predições do LLM
│   ├── metrics.json           métricas finais
│   └── exemplo_prompt.txt     prompt completo gerado para o LLM
└── figs/                      gráficos gerados pelo notebook
```

## Referências principais

- ARGYLE, L. P. et al. *Out of One, Many: Using Language Models to Simulate Human Samples*. Political Analysis, 31(3):337-351, 2023.
- BISBEE, J. et al. *Synthetic Replacements for Human Survey Data? The Perils of Large Language Models*. Political Analysis, 2024.
- CESOP/IPEC. *Pesquisa 04839 — Percepção dos brasileiros sobre temas relacionados à desigualdade*. Unicamp, 2023.
- *Simulating Public Opinion: Comparing Distributional and Individual-Level Predictions from LLMs and Random Forests*. Entropy, v.27, n.923, 2025.
- QWEN TEAM. *Qwen2.5 Technical Report*. Alibaba, 2024.
