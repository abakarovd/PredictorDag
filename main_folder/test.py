import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

names = [
    "Магомед Магомедов Магомедович",
    "Иван Иванов Иванович",
    "Гаджи Гаджиев Гаджиевич"
]

vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2,3))
X = vectorizer.fit_transform(names)

df_tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print(df_tfidf.round(2))


