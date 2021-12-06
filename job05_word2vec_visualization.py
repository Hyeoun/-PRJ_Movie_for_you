import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE  # 차원축소
from matplotlib import font_manager, rc  # 한글 사용시
import matplotlib as mpl
# https://gaussian37.github.io/ml-concept-t_sne/

font_path = './malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
mpl.rcParams['axes.unicode_minus']=False
rc('font', family=font_name)

embedding_model = Word2Vec.load('./models/word2VecModel_2015_2021.model')
key_word = '여름'
sim_word = embedding_model.wv.most_similar(key_word, topn=10)  # key_word와 유사단어 topn개수만큼 뽑는다.
print(sim_word)

vectors = []
labels = []
for label, _ in sim_word:
    labels.append(label)
    vectors.append(embedding_model.wv[label])  # 각 단어의 벡터값을 추가한다.
print(vectors[0])
print(len(vectors[0]))

df_vectors = pd.DataFrame(vectors)
print(df_vectors.head())

tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)  # 평면좌표로 축소
# n_components : 몇차원으로 축소할것인가?, random_state : 랜덤값이 나오긴 하는데 똑같이 나온다.
new_value = tsne_model.fit_transform(df_vectors)  # 2차원 벡터를 준다.
df_xy = pd.DataFrame({'words':labels, 'x':new_value[:, 0], 'y':new_value[:, 1]})
print(df_xy.tail())

print(df_xy.shape)

df_xy.loc[df_xy.shape[0]] = (key_word, 0, 0)  # key_word 데이터 추가한다.
print(df_xy.tail(11))

plt.figure(figsize=(8,8))
plt.scatter(0, 0, s=1500, marker='*')
for i in range(len(df_xy.x) - 1):
    a = df_xy.loc[[i, len(df_xy.x) - 1], :]
    plt.plot(a.x, a.y, '-D', linewidth=1)
    plt.annotate(df_xy.words[i], xytext=(1, 1), xy=(df_xy.x[i], df_xy.y[i]), textcoords='offset points', ha='right', va='bottom')
    # textcoords='offset points' : 글자를 조금 띄어놓는다.
plt.show()