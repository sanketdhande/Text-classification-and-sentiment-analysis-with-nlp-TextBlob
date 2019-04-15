from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField



class AddForm(FlaskForm):

    movietextcolumn = StringField(' REVIEW ')
    submit = SubmitField('submit')
