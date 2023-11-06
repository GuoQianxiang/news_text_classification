# 引用所需的库
import nltk
nltk.download()
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

# 准备训练数据
documents = ["I love machine learning", "I hate math", "I enjoy playing video games"]
tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(documents)]

# 定义Doc2Vec模型
model = Doc2Vec(vector_size=100, min_count=1, epochs=10)

# 构建词汇表
model.build_vocab(tagged_data)

# 训练Doc2Vec模型
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# 获取文档向量
vector = model.infer_vector(word_tokenize("I love NLP"))
print(vector)