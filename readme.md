# Inteligência Artificial para Classificação de Gêneros Musicais por Análise de Letras

> “As palavras carregam ritmo, emoção e identidade — e nelas, a inteligência artificial pode reconhecer música.”

---

<!-- Caso queira adicionar uma imagem de capa, descomente abaixo -->
<!-- ![Capa do projeto](docs/cover_music_ai.png) -->

**Autor:** Yuri Cordeiro de Almeida  
**MBA em Inteligência Artificial e Big Data – USP/ICMC (2025)**  
**Orientador:** Prof. Dr. Ivandre Paraboni  

---

## 1. Sobre o Projeto

Este repositório reúne o código, resultados e documentação do Trabalho de Conclusão de Curso do MBA em Inteligência Artificial e Big Data da USP/ICMC.  
O estudo investiga **a classificação automática de gêneros musicais a partir das letras**, comparando diferentes representações textuais e modelos de *Processamento de Linguagem Natural (PLN)* e *Aprendizado de Máquina*.

A proposta parte da ideia de que, embora a maioria dos sistemas de recomendação musical utilize dados acústicos, **as letras também expressam padrões semânticos e culturais** que permitem identificar o estilo musical de uma canção.  

---

## 2. Contexto e Motivação

A classificação de gênero musical é um desafio recorrente em *Music Information Retrieval (MIR)*.  
Tradicionalmente, essa tarefa depende de características acústicas como MFCCs e espectrogramas, exigindo alto custo de processamento.  
Neste trabalho, o foco desloca-se para o **texto**: investigar o quanto o conteúdo lírico, isolado do áudio, é capaz de indicar o gênero musical de forma confiável.  

O estudo também busca comparar abordagens clássicas e modernas sob três dimensões:
- Desempenho (métricas quantitativas);
- Interpretabilidade dos modelos;
- Custo computacional e reprodutibilidade.

---

## 3. Corpus

O corpus utilizado contém **2.400 letras em inglês**, divididas igualmente entre oito gêneros principais:

```
rock · pop · hip-hop · r-n-b · country · jazz · electronic · gospel
```

Os dados foram coletados e integrados a partir de três bases públicas:
- [Scrapped Lyrics Dataset – Kaggle](https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres)  
- [Spotify Tracks Dataset – Kaggle](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset)  
- [Melon Playlist Dataset – Universitat Pompeu Fabra](https://doi.org/10.5281/zenodo.2628366)

As letras foram filtradas, padronizadas e balanceadas (300 amostras por gênero).  
Por questões de direitos autorais e tamanho, apenas um **subconjunto representativo** está disponível neste repositório, junto com scripts para reconstrução do corpus completo.

---

## 4. Modelagem e Métodos

Foram comparadas quatro abordagens principais de classificação:

| Grupo | Abordagem | Ferramentas |
|-------|------------|-------------|
| **Baseline** | TF-IDF + Regressão Logística / SVM | `scikit-learn` |
| **Embeddings Fixos** | MPNet, GloVe, FastText | `sentence-transformers`, `transformers` |
| **Transformers** | DistilBERT (fine-tuning supervisionado) | `transformers`, `torch` |
| **Hierárquico** | Rede Hierárquica (HAN) com GRUs bidirecionais | `keras`, `tensorflow` |

### Parâmetros principais

**TF-IDF + Regressão Logística**
```python
TfidfVectorizer(max_features=5000, ngram_range=(1,2))
LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
```

**SVM Linear**
```python
LinearSVC(C=1.0, penalty='l2', loss='squared_hinge', max_iter=1000)
```

**DistilBERT Fine-Tuning**
- Modelo: `distilbert-base-uncased`
- Parâmetros: `max_length=512`, `batch_size=16`, `learning_rate=3e-5`, `epochs=3`

**HAN (Hierarchical Attention Network)**
- Estrutura: GRU bidirecional + pooling
- Embeddings: GloVe 100d e FastText 300d
- Limite de entrada: 15 sentenças × 20 palavras

---

## 5. Resultados

| Modelo | Representação | F1-macro | Acurácia |
|:--|:--|:--:|:--:|
| MPNet + Regressão Logística | Embeddings fixos | **0.50** | **0.51** |
| DistilBERT (fine-tuning) | Transformer contextual | 0.48 | 0.49 |
| TF-IDF + Regressão Logística | Vetorização esparsa | 0.43 | 0.44 |
| HAN + FastText | Estrutura hierárquica | 0.33 | 0.33 |

Os resultados mostram que **embeddings modernos combinados a classificadores lineares** alcançam desempenho comparável a Transformers fine-tunados, mas com menor custo computacional.  
Modelos clássicos, como o TF-IDF, ainda se mantêm competitivos e interpretáveis — úteis como baseline robusto.

---

## 6. Execução

Clone o repositório e instale as dependências:
```bash
git clone https://github.com/cordeirox/tcc-mba-ia-bigdata-usp.git
cd tcc-mba-ia-bigdata-usp
pip install -r requirements.txt
```

Para executar os experimentos:
```bash
python src/train_models.py
```

Os notebooks de cada abordagem estão organizados por etapa:
1. Pré-processamento  
2. TF-IDF + modelos lineares  
3. Embeddings MPNet  
4. Fine-tuning DistilBERT  
5. Modelo hierárquico (HAN)

---

## 7. Principais Conclusões

- **Letras de música são representações válidas de gênero**, especialmente com embeddings semânticos.  
- **Classificadores lineares** continuam relevantes quando combinados a boas representações.  
- **Transformers** entregam resultados próximos, mas exigem mais recursos de hardware.  
- **Modelos hierárquicos**, embora promissores, demandam corpora extensos e tuning refinado.  

Este estudo reforça que, mesmo diante da sofisticação das arquiteturas modernas, a **clareza interpretativa e o custo computacional** ainda são fatores decisivos para aplicações reais em *Music Information Retrieval*.

---

## 8. Referência

DE ALMEIDA, Y. C. *Classificação de Gêneros Musicais a partir das Letras: Estudo comparativo de modelos estatísticos e neurais de aprendizado de máquina em PLN aplicado à música.*  
MBA em Inteligência Artificial e Big Data – USP/ICMC, 2025.

---

## 9. Licença

Este repositório é distribuído sob a **Licença MIT**, permitindo uso, cópia e modificação mediante crédito ao autor.  
Consulte o arquivo [LICENSE](LICENSE) para mais informações.

---

## 10. Contato

📍 São Paulo, Brasil  
📧 [yuricordeiro@usp.br](mailto:yuricordeiro@usp.br)  
🔗 [linkedin.com/in/yuricordeiro](https://linkedin.com/in/yuricordeiro)
