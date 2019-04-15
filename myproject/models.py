from myproject import db,login_manager
from werkzeug.security import generate_password_hash,check_password_hash
from flask_login import UserMixin
from sqlalchemy.types import Integer, String
# By inheriting the UserMixin we get access to a lot of built-in attributes
# which we will be able to call in our views!
# is_authenticated()
# is_active()
# is_anonymous()
# get_id()


# The user_loader decorator allows flask-login to load the current user
# and grab their id.
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)

class User(db.Model, UserMixin):

    # Create a table in the db
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key = True)
    email = db.Column(db.String(64), unique=True, index=True)
    username = db.Column(db.String(64), unique=True, index=True)
    password_hash = db.Column(db.String(128))

    def __init__(self, email, username, password):
        self.email = email
        self.username = username
        self.password_hash = generate_password_hash(password)

    def check_password(self,password):
        # https://stackoverflow.com/questions/23432478/flask-generate-password-hash-not-constant-output
        return check_password_hash(self.password_hash,password)

    def __repr__(self):
        return f"UserName: {self.username}"

class MovieReview(db.Model):
    # Setup the relationship to the User table
    __tablename__ = 'moviereviews'
    __table_args__ = {'sqlite_autoincrement': True}

    id = db.Column(db.Integer,primary_key = True)
    movietextcolumn = db.Column(db.Text)
    #moviereviews = db.relationship('Sentimentclass', backref='moviereviews', lazy=True)
    def __init__(self,movietextcolumn):
        self.movietextcolumn = movietextcolumn
        #self.moviereviews = moviereviews
    def __repr__(self):
        return f"{self.movietextcolumn}"
class Sentimentclass(db.Model):
    __tablename__ = 'sentiments'
    id = db.Column(db.Integer,primary_key = True)
    sentimentcolumn = db.Column(db.Text,nullable=True)
    #moviereview_id = db.Column(db.Integer, db.ForeignKey('moviereviews.id'), nullable=False)
    def __init__(self,sentimentcolumn):
        self.sentimentcolumn = sentimentcolumn
        #self.moviereview_id = moviereview_id
    def __repr__(self):
        return f"{self.sentimentcolumn}"

class SpamMessages(db.Model):
    __tablename__ = 'spam_messages'
    id = db.Column(db.Integer,primary_key = True)
    spam_messages_column =  db.Column(db.Text,nullable=True)
    def __init__(self,spam_messages_column):
        self.spam_messages_column = spam_messages_column

    def __repr__(self):
        return f"{self.spam_messages_column}"

class SpamOrHam(db.Model):
    __tablename__ = 'spamham'
    id = db.Column(db.Integer,primary_key = True)
    spamham_column =  db.Column(db.Text,nullable=True)
    def __init__(self,spamham_column):
        self.spamham_column = spamham_column

    def __repr__(self):
        return f"{self.spamham_column}"
