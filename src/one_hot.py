def one_hot_encoding(documents):
    words = list(set(" ".join(documents).split()))
    word_to_index = {word: i for i, word in enumerate(words)}
    
    one_hot_vectors = []
    
    for doc in documents:
        doc_vector = []
        for word in doc.split():
            vector = [0]*len(words)
            vector[word_to_index[word]] = 1
            doc_vector.append(vector)
        one_hot_vectors.append(doc_vector)
    
    return words, one_hot_vectors
