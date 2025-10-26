import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("data/Letras-e-generos-full.csv", encoding='latin-1', sep=';', engine='python')
df.columns = df.columns.str.strip()
df['Music Genre'] = df['Music Genre'].str.lower().str.strip()
df = df.drop_duplicates(subset='Lyrics')
df_balanced = df.groupby('Music Genre', group_keys=False).apply(lambda x: x.sample(n=300, random_state=42)).reset_index(drop=True)

X = df_balanced['Lyrics']
y = df_balanced['Music Genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

print(classification_report(y_test, y_pred, zero_division=0))
