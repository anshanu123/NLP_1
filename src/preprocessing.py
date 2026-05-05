import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def preprocess_documents(documents):
    processed_docs = []
    
    for text in documents:
        text = text.lower()
        words = text.split()
        words = [word for word in words if word not in stopwords.words('english')]
        processed_docs.append(" ".join(words))
    
    return processed_docs
