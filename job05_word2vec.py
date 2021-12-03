import pandas as pd
from gensim.models import Word2Vec  # 현재 버전 4.0.1  문제가 발생할 경우 3.8.3로 다운그레이드를 한다.

review_word = pd.read_csv('./crawling_data/cleaned_review_2015_2021.csv')
review_word.info()
print(review_word.head())

cleaned_token_review = list(review_word['cleaned_sentences'])

cleaned_tokens = []
for sentence in cleaned_token_review:
    token = sentence.split()
    cleaned_tokens.append(token)

# print(cleaned_tokens)
embedding_model = Word2Vec(cleaned_tokens, vector_size=100, window=4, min_count=20, workers=4, epochs=100, sg=1)
# size : 차원수 제한, workers : 코어수, min_count : 출현빈도 20회 이상만 사용, iter : 반복횟수, window : 몇개씩 묶어서 학습할 것인지?
# gensim 4.0.0 size => vector_size, iter => epochs
embedding_model.save('./models/word2VecModel_2015_2021.model')
# print(embedding_model.wv.vocab.keys())  # gensim 4.0.0 => print(list(embedding_model.wv.index_to_key))
# print(len(embedding_model.wv.vocab.keys()))  # gensim 4.0.0 => print(len(list(embedding_model.wv.index_to_key)))
print(list(embedding_model.wv.index_to_key))
print(len(list(embedding_model.wv.index_to_key)))