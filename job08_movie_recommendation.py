import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmwrite, mmread
import pickle
from gensim.models import Word2Vec

df_reviews = pd.read_csv('./crawling_data/cleaned_review_2015_2021.csv')
Tfidf_matrix = mmread('./models/Tfidf_movie_review.mtx').tocsr()
with open('./models/tfidf.pickle', 'rb') as f:
    Tfidf = pickle.load(f)

def getRecommendation(cosine_sim):
    simScore = list(enumerate(cosine_sim[-1]))
    simScore = sorted(simScore, key=lambda x:x[1], reverse=True)
    simScore = simScore[1:11]  # 내림차순으로 나올테니 가장 처음에 나오는것은 자기 자신이다. 그러므로 처음은 제외한다.
    movididx = [i[0] for i in simScore]
    recMovieList = df_reviews.iloc[movididx]
    return recMovieList

# movie_idx = df_reviews[df_reviews['titles'] == '겨울왕국 2 (Frozen 2)'].index[0]  # 해당 영화제목의 위치를 받는다.
# print(movie_idx)
#
# print(df_reviews.iloc[movie_idx, 0])
#
# cosine_sim = linear_kernel(Tfidf_matrix[movie_idx], Tfidf_matrix)
# recommendation = getRecommendation(cosine_sim)
# print(recommendation.iloc[:, 0])

embedding_model = Word2Vec.load('./models/word2VecModel_2015_2021.model')

# ======================================================================================================
# key_word = '스파이더맨'
# sentence = [key_word] * 11
# sim_word = embedding_model.wv.most_similar(key_word, topn=10)  # 유사단어 10개를 받는다.
# words = []
# for word, _ in sim_word:  # 앞에는 단어, 뒤에는 유사도
#     words.append(word)
# print(words)
#
# for i, word in enumerate(words):
#     sentence += [word] * (10 - i)  # 유사한 단어들이 반복된다.
# sentence = ' '.join(sentence)
# print(sentence)
# sentence_vec = Tfidf.transform([sentence])
# cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)
# recommendation = getRecommendation(cosine_sim)
# print(recommendation['titles'])

# ===============================================================================================================
sentence = '어느 날 기이한 존재로부터 지옥행을 선고받은 사람들. 충격과 두려움에 휩싸인 도시에 대혼란의 시대가 도래한다. 신의 심판을 외치며 세를 확장하려는 종교단체와 진실을 파헤치는 자들의 이야기'
print(sentence)
sentence_vec = Tfidf.transform([sentence])
cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)
recommendation = getRecommendation(cosine_sim)
print(recommendation['titles'])