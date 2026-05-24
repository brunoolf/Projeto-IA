# Simulação de Opinião Pública sobre Desigualdade no Brasil com LLMs

> Projeto final da disciplina **Inteligência Artificial — 1º Semestre 2026 (7G)**
> Universidade Presbiteriana Mackenzie | Prof. Rogério de Oliveira

## Aluno

- **Bruno Ferrão** — `bruno_ferrao01@hotmail.com`

## TL;DR

A partir da pesquisa **CESOP/IPEC 04839** *(Percepção dos Brasileiros sobre Temas Relacionados à Desigualdade — IPEC, jul/2023, N=2.000)*, simulamos as respostas de 200 respondentes a duas afirmações em escala Likert de 5 pontos usando o LLM aberto **Qwen 2.5 0.5B Instruct** e comparamos os resultados com um *baseline* supervisionado de **Random Forest**.

Questões simuladas (ambas com 5 alternativas → atendem ao requisito de "ao menos uma com mais de 2 alternativas"):

| ID | Afirmação |
|----|-----------|
| **P6A** | "A abordagem policial é baseada no tipo de cabelo, tipo de vestimenta e cor de pele das pessoas." |
| **P6C** | "Aumentar a representatividade de pessoas negras, mulheres e LGBTQIA+ na política e em cargos de poder contribui para diminuir as desigualdades estruturais." |

Variáveis preditoras (7): `sexo`, `faixa_idade`, `escolaridade`, `raca_cor`, `religiao`, `renda_familiar`, `regiao`.

Configuração: `temperature=0.8`, `top_p=0.9`, `n=200` respondentes amostrados, `repetições=3` → **1.200 inferências totais**.

---

## Entregáveis

| Item | Caminho | Status |
|------|---------|--------|
| Artigo (Markdown — formato SBC opcional) | [`artigo/artigo.md`](artigo/artigo.md) | ✅ |
| Roteiro do vídeo (≤6 min) | [`artigo/roteiro_video.md`](artigo/roteiro_video.md) | ✅ |
| Notebook executável | [`notebook/simulacao.ipynb`](notebook/simulacao.ipynb) | ✅ |
| Script Python (equivalente ao notebook) | [`notebook/run_simulation.py`](notebook/run_simulation.py) | ✅ |
| Amostra de 200 respondentes | [`data/sample_200.csv`](data/sample_200.csv) | ✅ |
| Predições do LLM (1.200 linhas) | [`data/llm_predictions.csv`](data/llm_predictions.csv) | ✅ |
| Métricas (JSON) | [`data/metrics.json`](data/metrics.json) | ✅ |
| Exemplo de prompt gerado | [`data/exemplo_prompt.txt`](data/exemplo_prompt.txt) | ✅ |
| Gráficos | [`figs/`](figs/) | ✅ |
| Link do vídeo no YouTube | _a ser inserido após gravação_ | ⏳ |

---

## Reprodutibilidade

```bash
# Pré-requisitos: Python 3.10+
pip install pandas pyreadstat pypdf scikit-learn matplotlib seaborn requests numpy transformers torch

# Coloque o arquivo Cesop em: <repo>/Projeto_1_2026/04839/04839.sav
python notebook/run_simulation.py
```

Em CPU comum (Intel i5 / i7 de 8ª geração ou superior), o pipeline completo (carregamento do modelo + 1.200 inferências + Random Forest + plots) leva **cerca de 10–15 minutos**.

Compatível com **Google Colab gratuito** (CPU runtime já é suficiente; em GPU T4 cai para ~3 min).

---

## Metodologia (resumo)

1. **Carregamento e limpeza** do arquivo SPSS (`pyreadstat`).
2. **Decodificação** dos códigos numéricos das categorias para *strings* legíveis (ex.: `1 → "Masculino"`, `3 → "Parda"`).
3. **Amostragem estratificada** de 200 respondentes (estratos: sexo × região).
4. **Construção do prompt** com instrução de sistema + perfil + afirmação.
5. **Geração** com `Qwen2.5-0.5B-Instruct`, 3 repetições estocásticas por respondente.
6. **Parsing** robusto: regex extrai o primeiro dígito de 1 a 5; nulos imputados pela moda da amostra.
7. **Avaliação**: acurácia, MAE, divergência KL entre distribuições.
8. **Baseline supervisionado**: Random Forest 200 árvores, profundidade 8, *5-fold StratifiedCV*.
9. **Explicabilidade**: importâncias relativas do RF e matrizes de confusão do LLM.

---

## Resultados (síntese)

### Tabela de métricas

| Questão | Modelo | Acurácia | MAE | KL(pred‖real) |
|---|---|---|---|---|
| **P6A** | LLM Qwen2.5-0.5B | 0,145 | 2,072 | 0,958 |
| **P6A** | Random Forest (5-CV) | **0,335 ± 0,034** | **1,460** | **0,216** |
| **P6C** | LLM Qwen2.5-0.5B | 0,130 | 2,373 | 1,301 |
| **P6C** | Random Forest (5-CV) | **0,440 ± 0,068** | **1,355** | **0,208** |

### Principais achados

1. **Random Forest superou o LLM em todas as métricas** — 2,3× a 3,4× em acurácia, KL de 4 a 6× menor.
2. **Vieses opostos**: o LLM colapsou 50–67% das respostas em "Discorda totalmente"; o RF colapsou 68–73% em "Concorda totalmente". A distribuição real concentra 38–45% em "Concorda totalmente".
3. **Variáveis mais informativas (RF)**: `religião` e `faixa_idade` (cada ~19%); `escolaridade` e `renda_familiar` (~14–16%); `sexo` é a menos informativa (~7–8%).
4. **Tempo de execução**: 1.200 inferências do LLM em **578 s (≈9 min 38 s)** em CPU.

### Figuras

- **Distribuições reais × LLM × RF**: [`figs/01_distribuicoes.png`](figs/01_distribuicoes.png)
- **Matrizes de confusão do LLM**: [`figs/02_confusion_llm.png`](figs/02_confusion_llm.png)
- **Importância das variáveis (RF)**: [`figs/03_feature_importance.png`](figs/03_feature_importance.png)
- **Acurácia LLM × RF**: [`figs/04_accuracy_comparison.png`](figs/04_accuracy_comparison.png)

Discussão completa: ver Seção 5 do [artigo](artigo/artigo.md).

---

## Referências principais

- ARGYLE, L. P. et al. *Out of One, Many: Using Language Models to Simulate Human Samples*. Political Analysis, 2023.
- CESOP/IPEC. *Pesquisa 04839 — Percepção dos Brasileiros sobre Temas Relacionados à Desigualdade*. Unicamp, 2023.
- *Simulating Public Opinion: Comparing Distributional and Individual-Level Predictions from LLMs and Random Forests*. Entropy, v.27, n.923, 2025.
- QWEN TEAM. *Qwen2.5 Technical Report*. Alibaba, 2024.

---

## Licença & Uso

Modelos e dependências utilizados são abertos (Apache 2.0 / MIT). A pesquisa CESOP/IPEC 04839 é disponibilizada gratuitamente pelo Centro de Estudos de Opinião Pública da Unicamp para fins acadêmicos.
