from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from flask_bootstrap import Bootstrap
from textblob import TextBlob,Word
import random
import time
from sklearn.linear_model import LogisticRegression
import itertools
#from vectorizer import vect
import numpy as np
from myproject import app,db
from flask import render_template, redirect, request, url_for, flash,abort
from flask_login import login_user,login_required,logout_user
from myproject.models import User,MovieReview,Sentimentclass,SpamMessages,SpamOrHam
from myproject.forms import LoginForm, RegistrationForm
from werkzeug.security import generate_password_hash, check_password_hash
import os
from forms import  AddForm

@app.route('/')
def home():
    return render_template('nlp.html')


@app.route('/welcome')
@login_required
def welcome_user():
    return render_template('welcome_user.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You logged out!')
    return redirect(url_for('home'))


@app.route('/login', methods=['GET', 'POST'])
def login():

    form = LoginForm()
    if form.validate_on_submit():
        # Grab the user from our User Models table
        user = User.query.filter_by(email=form.email.data).first()

       
        if user.check_password(form.password.data) and user is not None:
            #Log in the user

            login_user(user)
            flash('Logged in successfully.')

           
            # flask saves that URL as 'next'.
            next = request.args.get('next')

            # So let's now check if that next exists, otherwise we'll go to
            # the welcome page.
            if next == None or not next[0]=='/':
                next = url_for('welcome_user')

            return redirect(next)
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()

    if form.validate_on_submit():
        user = User(email=form.email.data,
                    username=form.username.data,
                    password=form.password.data)

        db.session.add(user)
        db.session.commit()
        flash('Thanks for registering! Now you can login!')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)




@app.route('/add_review',methods=['GET', 'POST'])
def add_review():
    if request.method == 'POST':

        #if request.method == 'POST':
        movietextcolumn = request.form['movietextcolumn']
        newreview = MovieReview(movietextcolumn)
        db.session.add(newreview)
        db.session.commit()
        #NLP Stuff
        return redirect(url_for('classify'))
    return render_template('base.html')
@app.route('/classify')
def classify():
    lastreview = MovieReview.query.order_by(MovieReview.id.desc()).first()

    blob = TextBlob(str(lastreview))
    received_text3 = blob


    blob_sentiment,blob_subjectivity = blob.sentiment.polarity ,blob.sentiment.subjectivity
#wordreview = ['negative','positive']
#blob_sentiment = int(blob_sentiment1)
#blob_review = wordreview[blob_sentiment[0]]

    if blob_sentiment >= 0.1:
        blob_sentiment = 'positive'
    elif blob_sentiment <= -0.1:
        blob_sentiment = 'negative'
    else:
        blob_sentiment = 'neutral'
    #print (blob_sentiment)

    sentimentcolumn = Sentimentclass(blob_sentiment)
    db.session.add(sentimentcolumn)
    db.session.commit()



    return redirect(url_for('list_review'))

@app.route('/list_review')
def list_review():
    # Grab a list of puppies from database.
    sentiments = Sentimentclass.query.all()
    moviereviews = MovieReview.query.all()

    #review_post = MovieReview.query.get_or_404(MovieReview.id)
    return render_template('result2.html', sentiments=sentiments,moviereviews=moviereviews)


@app.route('/analyse',methods=['POST'])
def analyse():
    start = time.time()

    if request.method == 'POST':
        rawtext = request.form['rawtext']
    #NLP Stuff
        blob = TextBlob(rawtext)
        received_text2 = blob


    blob_sentiment,blob_subjectivity = blob.sentiment.polarity ,blob.sentiment.subjectivity
    #wordreview = ['negative','positive']
    #blob_sentiment = int(blob_sentiment1)
    #blob_review = wordreview[blob_sentiment[0]]
    number_of_tokens = len(list(blob.words))
    # Extracting Main Points
    nouns = list()


    for word, tag in blob.tags:
        if tag == 'NN':
            nouns.append(word.lemmatize())
            len_of_words = len(nouns)
            rand_words = random.sample(nouns,len(nouns))
            final_word = list()
            for item in rand_words:
                word = Word(item).pluralize()
                final_word.append(word)
                summary = final_word
                end = time.time()
                final_time = end-start




    return render_template('nlp.html',received_text = received_text2,
    number_of_tokens=number_of_tokens,blob_sentiment=blob_sentiment,
    blob_subjectivity=blob_subjectivity,summary=summary,final_time=final_time)








@app.route('/spam_detection')
def spam_detection():
    return render_template('spam_detection.html')


@app.route('/sentiments')
def sentiments():
    return render_template('sentiments.html')

@app.route('/spam_message_dir',methods=['GET','POST'])
def spam_message_dir():

    if request.method == 'POST':
        spam_messages_column = request.form['spam_messages_column']
        new_spam = SpamMessages(spam_messages_column)
        db.session.add(new_spam)
        db.session.commit()
    return redirect(url_for('predict'))
    return render_template('base.html')

@app.route('/predict')
def predict():

    df= pd.read_csv('YoutubeSpamMergedData.csv')
    df_data = df[["CONTENT","CLASS"]]
    # Features and Labels
    df_x = df_data['CONTENT']
    df_y = df_data['CLASS']
    # Extract Feature With CountVectorizer
    corpus = df_x     #collection of texts
    cv = CountVectorizer()      #counts the number of words
    X = cv.fit_transform(corpus) # Fit the Data by applying mean and standard deviation (data - mean)/standard deviation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
    #Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test) #R-squared value for the predictions
    #Alternative Usage of Saved Model
    # ytb_model = open("naivebayes_spam_model.pkl","rb")
    # clf = joblib.load(ytb_model)
    spam_messages_column2 = SpamMessages.query.order_by(SpamMessages.id.desc()).first()
    data = [str(spam_messages_column2)]
    vect = cv.transform(data).toarray()
    my_prediction = clf.predict(vect)
#return render_template('result.html',my_prediction = my_prediction)
    if my_prediction == 1:
        my_prediction = 'Spam'
    elif my_prediction == 0:
        my_prediction = 'Not a Spam'


    spamham_column = SpamOrHam(my_prediction)
    db.session.add(spamham_column)
    db.session.commit()
    return redirect(url_for('list_spam'))


@app.route('/list_spam')
def list_spam():
    # Grab a list of puppies from database.
    spam_messages = SpamMessages.query.all()
    spamham = SpamOrHam.query.all()
    return render_template('result.html',spam_messages = spam_messages,spamham = spamham)























if __name__ == '__main__':
    app.run(debug=True)
