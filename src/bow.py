from sklearn.feature_extraction.text import CountVectorizer

def bag_of_words(documents):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)
    
    return vectorizer.get_feature_names_out(), X.toarray()
