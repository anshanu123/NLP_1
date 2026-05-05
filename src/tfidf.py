from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_vectorization(documents):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)
    
    return vectorizer.get_feature_names_out(), X.toarray()
