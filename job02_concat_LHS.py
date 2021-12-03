import pandas as pd

df = pd.DataFrame()
years = 2017
page = 53
for i in range(1, page + 1):
    df_temp = pd.read_csv('./crawling_data/reviews_{}_{}.csv'.format(years, i))
    df_temp.dropna(inplace=True)
    df_temp.drop_duplicates()
    df_temp.columns = ['title', 'reviews']
    df_temp.to_csv('./crawling_data/reviews_{}_{}.csv'.format(years, i), index=False)
    df = pd.concat([df, df_temp], ignore_index=True)

df.info()
df.to_csv('./crawling_data/reviews_{}.csv'.format(years), index=False)