# 文書をBoWを用いてベクトル化する

from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle
import sys

# __pychache__を消す
sys.dont_write_bytecode = True

current_dir = os.path.dirname(__file__)

# stopwordsを読み出す
stop = pickle.load(open(os.path.join(
                        current_dir, 'pkl_objects', 'stopwords.pkl'), 'rb'))

def tokenizer(text):
    # HTMLタグ（<（文字列）>の形の単語を排除
    text = re.sub('<[^>]*>', '', text)
    # ?, :, ;を排除。ただし排除しないパターンをうまく指定して、:Dや;P、:-)のような顔文字が消えないようにしている。
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # [\W]+で英数字と_以外の任意の文字を消し、lowerで小文字にする。
    # :-) を:)のように変換して、一旦消えた絵文字を後ろにくっつける。
    text = (re.sub(r'[\W]+', ' ', text.lower()) + ''.join(emoticons).replace('-', ''))
    # クレンジング済みテキストを単語に分割し、stopwordsに含まれなければリストに追加
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)