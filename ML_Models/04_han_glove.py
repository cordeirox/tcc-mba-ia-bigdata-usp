import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, TimeDistributed, Bidirectional, GRU, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

MAX_SENTENCES = 15
MAX_SENTENCE_LENGTH = 20
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100

df = pd.read_csv("data/Letras-e-generos-full.csv", encoding='latin-1', sep=';', engine='python')
df.columns = df.columns.str.strip()
df['Music Genre'] = df['Music Genre'].str.lower().str.strip()
df = df.drop_duplicates(subset='Lyrics')
df_balanced = df.groupby('Music Genre', group_keys=False).apply(lambda x: x.sample(n=300, random_state=42)).reset_index(drop=True)

texts, labels = [], []
for _, row in df_balanced.iterrows():
    sentences = sent_tokenize(row['Lyrics'])
    tokenized = [word_tokenize(s.lower()) for s in sentences]
    texts.append(tokenized[:MAX_SENTENCES])
    labels.append(row['Music Genre'])

flat_sentences = [" ".join(sent) for song in texts for sent in song]
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token="<UNK>")
tokenizer.fit_on_texts(flat_sentences)

data = np.zeros((len(texts), MAX_SENTENCES, MAX_SENTENCE_LENGTH), dtype='int32')
for i, song in enumerate(texts):
    for j, sentence in enumerate(song):
        if j < MAX_SENTENCES:
            seq = tokenizer.texts_to_sequences([" ".join(sentence)])[0][:MAX_SENTENCE_LENGTH]
            data[i, j, :len(seq)] = seq

le = LabelEncoder()
labels_enc = le.fit_transform(labels)
categorical_labels = to_categorical(labels_enc)

X_train, X_test, y_train, y_test = train_test_split(data, categorical_labels, test_size=0.2, stratify=categorical_labels, random_state=42)

embedding_matrix = np.load("models/han_glove_embedding_matrix.npy")

input_layer = Input(shape=(MAX_SENTENCES, MAX_SENTENCE_LENGTH))
embedding_layer = Embedding(input_dim=MAX_NB_WORDS, output_dim=EMBEDDING_DIM, weights=[embedding_matrix],
                             input_length=MAX_SENTENCE_LENGTH, trainable=False)
x = TimeDistributed(embedding_layer)(input_layer)
x = TimeDistributed(Bidirectional(GRU(50, return_sequences=True)))(x)
x = TimeDistributed(GlobalAveragePooling1D())(x)
x = Bidirectional(GRU(50))(x)
x = Dense(64, activation='relu')(x)
output = Dense(len(le.classes_), activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.1)

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))
