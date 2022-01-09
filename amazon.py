import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv('datasets/amazon_review.csv')
df.head()

# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.
# Yorumları tarihe göre ağırlıklandırarak değerlendirmemiz isteniyor.

#  Var Olan Average Rating
df["overall"].mean() #4.587

# Time - based weighted
df["day_diff"].describe()

'''
count    4915.000000
mean      437.367040
std       209.439871
min         1.000000
25%       281.000000
50%       431.000000
75%       601.000000
max      1064.000000
Name: day_diff, dtype: float64
'''

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["day_diff"] <= 281, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 281) & (dataframe["day_diff"] <= 431), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 431) & (dataframe["day_diff"] <= 601), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 601), "overall"].mean() * w4 / 100

time_based_weighted_average(df) #4.5955

# Var Olan Average Rating : 4.587
# Güncel Yorumlara Göre Average Rating : 4.5955

# GÖREV 2: Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.

# helpful_no değişkenini oluşturalım.
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df = df[["reviewerName", "overall", "summary", "helpful_yes", "helpful_no", "total_vote", "reviewTime"]]

# score_up_down_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp veriye ekleyelim.

#### score_up_down_diff

def score_up_down_diff(up, down):
    return up - down

df["score_up_down_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

#### score_average_rating

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

#### wilson_lower_bound

def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

# Ürün için ürün detay sayfasında görüntülenecek 20 review

df[["reviewerName","wilson_lower_bound"]].sort_values("wilson_lower_bound", ascending=False).head(20)

'''
                              reviewerName  wilson_lower_bound
2031                  Hyoun Kim "Faluzure"            0.957544
3449                     NLee the Engineer            0.936519
4212                           SkincareCEO            0.912139
317                Amazon Customer "Kelly"            0.818577
4672                               Twister            0.808109
1835                           goconfigure            0.784651
3981            R. Sutton, Jr. "RWSynergy"            0.732136
3807                            R. Heisler            0.700442
4306                         Stellar Eller            0.670334
4596           Tom Henriksen "Doggy Diner"            0.663595
315             Amazon Customer "johncrea"            0.657411
1465                              D. Stein            0.645670
1609                                Eskimo            0.645670
4302                             Stayeraug            0.639772
4072                           sb21 "sb21"            0.609666
1072                        Crysis Complex            0.565518
2583                               J. Wong            0.565518
121                                 A. Lee            0.565518
1142  Daniel Pham(Danpham_X @ yahoo.  com)            0.565518
1753                             G. Becker            0.565518
'''