import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("data/Letras-e-generos-full.csv", encoding='latin-1', sep=';', engine='python')
df.columns = df.columns.str.strip()
df['Music Genre'] = df['Music Genre'].str.lower().str.strip()
df = df.drop_duplicates(subset='Lyrics')
df_balanced = df.groupby('Music Genre', group_keys=False).apply(lambda x: x.sample(n=300, random_state=42)).reset_index(drop=True)

X = df_balanced['Lyrics'].tolist()
y = df_balanced['Music Genre'].tolist()

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

model = SentenceTransformer('all-mpnet-base-v2')
X_train_emb = model.encode(X_train, batch_size=32, show_progress_bar=True)
X_test_emb = model.encode(X_test, batch_size=32, show_progress_bar=True)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_emb, y_train)
y_pred = clf.predict(X_test_emb)

print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
