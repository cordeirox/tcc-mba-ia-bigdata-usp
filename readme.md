# Artificial Intelligence for Music Genre Classification through Lyrics Analysis

> "If, for Chomsky (1965), the beauty of language lay in the power of finite rules to generate infinite expressions, throughout this work we understand that dense vectors and modern models reveal another kind of infinity — that of emergent relations in continuous high-dimensional spaces."

---

<!-- Optional cover image -->
<!-- ![Project cover](docs/cover_music_ai.png) -->

**Author:** Yuri Cordeiro de Almeida  
**MBA in Artificial Intelligence and Big Data – University of São Paulo (USP/ICMC, 2025)**  
**Advisor:** Prof. Dr. Ivandre Paraboni  
<p style="font-size:9px; color:gray;">
<i>Final grade: 9.0 / 10
</p>

---

## 1. About the Project

This repository contains the code, results, and documentation of the Capstone Project developed for the MBA in Artificial Intelligence and Big Data at USP/ICMC.  
The study investigates **automatic music genre classification based on lyrics**, comparing different textual representations and models of *Natural Language Processing (NLP)* and *Machine Learning (ML)*.

The central hypothesis is that, although most recommendation systems rely on audio data, **lyrics also encode semantic and cultural patterns** that can reveal the musical style of a song.

---

## 2. Background and Motivation

Music genre classification is a recurring challenge in *Music Information Retrieval (MIR)*.  
Traditionally, this task relies on acoustic features such as MFCCs and spectrograms, which demand high computational cost.  
This research focuses instead on the **text** itself — exploring how far lyrical content, without any audio, can reliably indicate musical genres.

The work also compares classic and modern approaches along three dimensions:
- Performance (quantitative metrics)  
- Model interpretability  
- Computational cost and reproducibility

---

## 3. Dataset

The final corpus includes **2,400 English lyrics**, equally distributed across eight major genres:

```
rock · pop · hip-hop · r-n-b · country · jazz · electronic · gospel
```
Here is the final dataset link: [TCC MBA IA & Big Data – Lyrics and Music Genres Dataset](https://www.kaggle.com/datasets/yuricordeiro/tcc-mba-ia-usp)

The dataset was built by integrating three open sources:
- [Scrapped Lyrics Dataset – Kaggle](https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres)  
- [Spotify Tracks Dataset – Kaggle](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset)  
- [Melon Playlist Dataset – Universitat Pompeu Fabra] - Thanks to Dmitry Bogdanov

Lyrics were filtered, standardized, and balanced (300 samples per genre).  
Due to copyright and size constraints, only a **representative subset** is available in this repository, along with scripts to reconstruct the full corpus.

---

## 4. Modeling and Methods

Four main classification approaches were compared:

| Group | Approach | Tools |
|-------|-----------|-------|
| **Baseline** | TF-IDF + Logistic Regression / SVM | `scikit-learn` |
| **Fixed Embeddings** | MPNet, GloVe, FastText | `sentence-transformers`, `transformers` |
| **Transformers** | DistilBERT (fine-tuned) | `transformers`, `torch` |
| **Hierarchical** | HAN (GRU-based Hierarchical Neural Network) | `keras`, `tensorflow` |

### Key Parameters

**TF-IDF + Logistic Regression**
```python
TfidfVectorizer(max_features=5000, ngram_range=(1,2))
LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
```

**Linear SVM**
```python
LinearSVC(C=1.0, penalty='l2', loss='squared_hinge', max_iter=1000)
```

**DistilBERT Fine-Tuning**
- Model: `distilbert-base-uncased`
- Parameters: `max_length=512`, `batch_size=16`, `learning_rate=3e-5`, `epochs=3`

**HAN (Hierarchical Attention Network)**
- Structure: Bidirectional GRU + pooling  
- Embeddings: GloVe 100d and FastText 300d  
- Input limit: 15 sentences × 20 words

---

## 5. Results

| Model | Representation | F1-macro | Accuracy |
|:--|:--|:--:|:--:|
| MPNet + Logistic Regression | Fixed embeddings | **0.50** | **0.51** |
| DistilBERT (fine-tuned) | Contextual Transformer | 0.48 | 0.49 |
| TF-IDF + Logistic Regression | Sparse vectorization | 0.43 | 0.44 |
| HAN + FastText | Hierarchical structure | 0.33 | 0.33 |

Results show that **modern embeddings combined with simple linear classifiers** can reach performance comparable to fine-tuned Transformers, with lower computational cost.  
Classic baselines like TF-IDF remain competitive and interpretable — useful as a strong reference.

---

## 6. Reproduction

Clone the repository and install dependencies:
```bash
git clone https://github.com/cordeirox/tcc-mba-ia-bigdata-usp.git
cd tcc-mba-ia-bigdata-usp
pip install -r requirements.txt
```

Run experiments:
```bash
python src/train_models.py
```

The notebooks are organized by stage:
1. Data preprocessing  
2. TF-IDF and classical models  
3. MPNet embeddings  
4. DistilBERT fine-tuning  
5. Hierarchical model (HAN)

---

## 7. Main Conclusions

- **Lyrics contain meaningful linguistic signals** that can support genre classification.  
- **Linear models** remain effective when combined with robust embeddings.  
- **Transformers** offer comparable accuracy but at higher computational cost.  
- **Hierarchical models** show potential but require larger datasets and tuning.  

This work reinforces that, even with sophisticated architectures, **interpretability and efficiency** remain key factors for practical applications in *Music Information Retrieval*.

---

## 8. Reference

DE ALMEIDA, Y. C. *Music Genre Classification from Lyrics: Comparative Study of Statistical and Neural Models for NLP Applied to Music.*  
MBA in Artificial Intelligence and Big Data – USP/ICMC, 2025.

---

## 9. License

This repository is distributed under the **MIT License**, allowing reuse, modification, and distribution with credit to the author.  
See the [LICENSE](LICENSE) file for details.

---

## 10. Contact

📍 São Paulo, Brazil  
📧 [yuri.cordeirox@gmail.com](mailto:yuri.cordeirox@gmail.com)  
🔗 [linkedin.com/in/yuricordeiro](https://linkedin.com/in/yuricordeiro)
