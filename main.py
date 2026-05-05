import os

os.makedirs("output", exist_ok=True)
from src.preprocessing import preprocess_documents
from src.one_hot import one_hot_encoding
from src.bow import bag_of_words
from src.tfidf import tfidf_vectorization
import pandas as pd

# Step 1: Load data
with open("data/sample.txt", "r") as file:
    documents = file.readlines()

documents = [doc.strip() for doc in documents]

print("Original Documents:", documents)

# Step 2: Preprocess
processed_docs = preprocess_documents(documents)
print("\nProcessed Documents:", processed_docs)

# Step 3: One-Hot Encoding
vocab, one_hot = one_hot_encoding(processed_docs)
print("\nOne Hot Encoding:", one_hot)

# Step 4: Bag of Words
bow_vocab, bow_matrix = bag_of_words(processed_docs)
print("\nBoW:\n", bow_matrix)

df_bow = pd.DataFrame(bow_matrix, columns=bow_vocab)
df_bow.to_csv("output/bow_output.csv", index=False)

# Step 5: TF-IDF
tfidf_vocab, tfidf_matrix = tfidf_vectorization(processed_docs)
print("\nTF-IDF:\n", tfidf_matrix)

df_tfidf = pd.DataFrame(tfidf_matrix, columns=tfidf_vocab)
df_tfidf.to_csv("output/tfidf_output.csv", index=False)
