import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from gensim.models import Word2Vec, FastText
from sentence_transformers import SentenceTransformer

text1_original = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes. Thank your message to show our words to the doctor, as his next contract checking, to all of us. I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago. I am very appreciated the full support of the professor, for our Springer proceedings publication"""

text1_reconstructed = """Today is our Dragon Boat Festival in our Chinese culture. I hope you enjoy it as my deepest wish. Thank you for showing our message to the doctor regarding his next contract check. I received this message a few days ago from the professor. I am very grateful for the full support of the professor for our Springer proceedings publication."""

text2_original = """During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor? Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think. Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so. Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets"""

text2_reconstructed = """During our final discussion, I told him about the new submission we have been waiting for since last autumn. The updates were confusing because they did not include the full reviewer or editor feedback. Nevertheless, I believe the team, despite some delays and reduced communication in recent days, really tried their best on the paper and cooperation. We should all be grateful for the acceptance and efforts until the Springer link finally arrived last week. Also, please remind me if the doctor still plans to edit the acknowledgments section before sending it again. Overall, let's make sure everyone is safe and celebrate the outcome with strong coffee and future goals."""

texts_original = [text1_original, text2_original]
texts_reconstructed = [text1_reconstructed, text2_reconstructed]

w2v_model = Word2Vec([text.split() for text in texts_original + texts_reconstructed], vector_size=50, window=5, min_count=1)
ft_model = FastText([text.split() for text in texts_original + texts_reconstructed], vector_size=50, window=5, min_count=1)
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_word2vec_embedding(text, model):
    words = text.split()
    vecs = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vecs, axis=0)

def get_fasttext_embedding(text, model):
    words = text.split()
    vecs = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vecs, axis=0)

def get_bert_embedding(text, model):
    return model.encode([text])[0]

def get_custom_embedding(text):
    words = text.split()
    vocab = list(set(" ".join(texts_original + texts_reconstructed).split()))
    vec = np.zeros(len(vocab))
    for i, w in enumerate(vocab):
        vec[i] = words.count(w)
    return vec

w2v_orig_emb = np.array([get_word2vec_embedding(t, w2v_model) for t in texts_original])
w2v_rec_emb = np.array([get_word2vec_embedding(t, w2v_model) for t in texts_reconstructed])

ft_orig_emb = np.array([get_fasttext_embedding(t, ft_model) for t in texts_original])
ft_rec_emb = np.array([get_fasttext_embedding(t, ft_model) for t in texts_reconstructed])

bert_orig_emb = np.array([get_bert_embedding(t, bert_model) for t in texts_original])
bert_rec_emb = np.array([get_bert_embedding(t, bert_model) for t in texts_reconstructed])

custom_orig_emb = np.array([get_custom_embedding(t) for t in texts_original])
custom_rec_emb = np.array([get_custom_embedding(t) for t in texts_reconstructed])

def print_cosine(name, orig, rec):
    for i, (o, r) in enumerate(zip(orig, rec), start=1):
        sim = cosine_similarity([o], [r])[0][0]
        print(f"{name}: Text {i}: {sim:.4f}")

print("Cosine Similarity between original and reconstructed texts:\n")
print_cosine("Word2Vec", w2v_orig_emb, w2v_rec_emb)
print_cosine("FastText", ft_orig_emb, ft_rec_emb)
print_cosine("BERT", bert_orig_emb, bert_rec_emb)
print_cosine("Custom_NLP", custom_orig_emb, custom_rec_emb)

embedding_types = {
    "Word2Vec": (w2v_orig_emb, w2v_rec_emb),
    "FastText": (ft_orig_emb, ft_rec_emb),
    "BERT": (bert_orig_emb, bert_rec_emb),
    "Custom_NLP": (custom_orig_emb, custom_rec_emb)
}

def visualize_embeddings_separately(embedding_types, method="PCA"):
    for name, (orig, rec) in embedding_types.items():
        all_emb = np.vstack([orig, rec])
        if method == "PCA":
            reducer = PCA(n_components=2)
            reduced = reducer.fit_transform(all_emb)
        elif method == "t-SNE":
            n_samples = all_emb.shape[0]
            perplexity = min(30, max(1, n_samples - 1))
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            reduced = reducer.fit_transform(all_emb)
        else:
            raise ValueError("Method must be 'PCA' or 't-SNE'")
        
        plt.figure(figsize=(6,5))
        plt.scatter(reduced[:len(orig), 0], reduced[:len(orig), 1], label="Original", c='blue')
        plt.scatter(reduced[len(orig):, 0], reduced[len(orig):, 1], label="Reconstructed", c='red')
        plt.title(f"{name} Embeddings Visualization ({method})")
        plt.legend()
        plt.show()

visualize_embeddings_separately(embedding_types, method="PCA")
visualize_embeddings_separately(embedding_types, method="t-SNE")
