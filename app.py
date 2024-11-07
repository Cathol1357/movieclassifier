from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np

# ローカルからHashingVectorizerをインポート
from vectorizer import vect

app = Flask(__name__)

####### 分類器の準備
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'classifier.pkl'), 'rb'))

# ユーザーのレビューデータベース
db = os.path.join(cur_dir, 'review.sqlite')

# ユーザーの入力文章documentを受け取ってネガポジ判定する
def classify(document):
    label = {0: 'negative', 1: 'positive'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    
    proba = clf.predict_proba(X).max()
    return label[y], proba

# ユーザーのレビューを受けてトレーニングする
def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])

# ユーザーのレビューをデータベースに保存する
def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("INSERT INTO review_db (review, sentiment, date) VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()
    

###### Flask

class ReviewForm(Form):
    moviereview = TextAreaField('',
                                [validators.DataRequired(), 
                                 validators.length(min=15)])
    

@app.route('/')
def index():
    # フォームを読み出す
    form = ReviewForm(request.form)
    
    # form部分にセットしてレンダリング
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    
    # HTTPリクエストと中身の確認
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        y, proba = classify(review)
        
        # ユーザーの入力したレビュー、予測結果、確率の値を渡してレンダリング
        return render_template('results.html', 
                               content=review,
                               prediction=y,
                               probability=round(proba*100, 2))
    
    return render_template('reviewform.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
    # ユーザーの入力を取り出す
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']
    
    # 整数値をpos negの文字列に変換
    inv_label = {'negative' : 0, 'positive' : 1}
    y = inv_label[prediction]
    
    # ユーザーから間違いを指摘されたら、正解の方にデータを修正する
    if feedback == 'Incorrect':
        y = int(not(y))
    # 学習
    train(review, y)
    
    # データベースに追加
    sqlite_entry(db, review, y)
    
    return render_template('thanks.html')

if __name__ == '__main__':
    app.run(debug=True)
    
    
    