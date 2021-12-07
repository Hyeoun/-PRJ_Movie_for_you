import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import collections
from konlpy.tag import Okt
from matplotlib import font_manager, rc
import matplotlib as mpl
import numpy as np
from PIL import Image

font_path = './malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
mpl.rcParams['axes.unicode_minus']=False
rc('font', family=font_name)

df = pd.read_csv('./crawling_data/cleaned_review_2015_2021.csv')
print(df.head())

words = df.iloc[1, 1]
words = words.split()
print(words)

word_dict = collections.Counter(words)
word_dict = dict(word_dict)
print(word_dict)

wordcloud_img = WordCloud(background_color='white', max_words=2000, font_path=font_path).generate_from_frequencies(word_dict)

plt.figure(figsize=(12,12))
plt.imshow(wordcloud_img, interpolation='bilinear')
plt.axis('off')
plt.show()

alice_mask = np.array(Image.open('./bin_mask.jpg'))

stopwords = ['영화', '감독', '개봉', '개봉일', '촬영', '관객', '관람', '주인공', '출연', '배우', '평점', '들이다', '푸다', '리뷰', '네이버']
wordcloud_img = WordCloud(background_color='white', max_words=2000, font_path=font_path, collocations=False, stopwords=stopwords, mask=alice_mask).generate(df.cleaned_sentences[3])

plt.figure(figsize=(12,12))
plt.imshow(wordcloud_img, interpolation='bilinear')
plt.axis('off')
plt.show()