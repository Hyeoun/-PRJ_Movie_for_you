import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic # ui를 클래스로 바꿔준다.
from PyQt5.QtCore import QStringListModel
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec
from scipy.io import mmread
import pickle

form_window = uic.loadUiType('./movie_recommendation.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self): # 버튼 누르는 함수 처리해 주는 곳
        super().__init__()
        self.setupUi(self)
        self.df_reviews = pd.read_csv('./crawling_data/cleaned_review_2015_2021.csv')
        self.Tfidf_metrix = mmread('./models/Tfidf_movie_review.mtx').tocsr()
        self.embedding_model = Word2Vec.load('./models/word2VecModel_2015_2021.model')
        with open('./models/tfidf.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.titles = list(self.df_reviews['titles'])
        self.titles.sort()
        for title in self.titles:
            self.cmb_titles.addItem(title)
        model = QStringListModel()
        model.setStringList(self.titles)
        self.cmb_titles.currentIndexChanged.connect(self.cmb_titles_slot)
        self.btn_recommend.clicked.connect(self.btn_recommend_slot)
        completer = QCompleter()
        completer.setModel(model)
        self.le_keyword.setCompleter(completer)

    def cmb_titles_slot(self):
        title = self.cmb_titles.currentText()
        recommendation_title = self.recommend_by_movie_title(title)
        self.lbl_recommend.setText(recommendation_title)

    def getRecommendation(self, cosine_sim):
        simScore = list(enumerate(cosine_sim[-1]))
        simScore = sorted(simScore, key=lambda x: x[1], reverse=True)
        simScore = simScore[1:11]  # 내림차순으로 나올테니 가장 처음에 나오는것은 자기 자신이다. 그러므로 처음은 제외한다.
        movididx = [i[0] for i in simScore]
        recMovieList = self.df_reviews.iloc[movididx]
        return recMovieList.titles

    def btn_recommend_slot(self):
        key_word = self.le_keyword.text()
        if key_word:
            if key_word in self.titles:
                recommendation_title = self.recommend_by_movie_title(key_word)
                self.lbl_recommend.setText(recommendation_title)
            else:
                key_word = key_word.split()
                if len(key_word) > 10:
                    sentence = ' '.join(key_word)
                    recommendation_title = self.recommend_by_sentence(sentence)
                    self.lbl_recommend.setText(recommendation_title)
                else:
                    sentence = [key_word[0]] * 11
                    try:
                        sim_word = self.embedding_model.wv.most_similar(key_word[0], topn=10)  # 유사단어 10개를 받는다.
                    except:
                        self.lbl_recommend.setText('제가 모르는 단어네요!')
                        return
                    words = []
                    for word, _ in sim_word:  # 앞에는 단어, 뒤에는 유사도
                        words.append(word)
                    print(words)

                    for i, word in enumerate(words):
                        sentence += [word] * (10 - i)  # 유사한 단어들이 반복된다.
                    sentence = ' '.join(sentence)
                    recommendation_title = self.recommend_by_sentence(sentence)
                    self.lbl_recommend.setText(recommendation_title)

    def recommend_by_movie_title(self, title):
        movie_idx = self.df_reviews[self.df_reviews['titles'] == title].index[0]  # 해당 영화제목의 위치를 받는다.
        cosine_sim = linear_kernel(self.Tfidf_metrix[movie_idx], self.Tfidf_metrix)
        recommendation_title = self.getRecommendation(cosine_sim)
        recommendation_title = '\n'.join(list(recommendation_title))
        return recommendation_title

    def recommend_by_sentence(self, sentence):
        sentence_vec = self.Tfidf.transform([sentence])
        cosine_sim = linear_kernel(sentence_vec, self.Tfidf_metrix)
        recommendation_title = self.getRecommendation(cosine_sim)
        recommendation_title = '\n'.join(list(recommendation_title))
        return recommendation_title

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())