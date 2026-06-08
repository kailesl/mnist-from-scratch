import numpy as np
class attention:
    def __init__(self,vocab_length,vocab_size):
        self.w=np.zeros((vocab_size))
        self.word=np.zeros((vocab_size,vocab_length))
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    def weight(self,word,vocab_word,i):
        self.w[i]=self.cosine_similarity(word,vocab_word)
        return self.w
at=attention(vocab_length,vocab_size)
for word in vocab:
    for i in range(vocab_size):
        w=at.weight(word,vocab_word,i)
    word=word*w#这里维度是对不上的，我大致想表达的意思就是这些 
