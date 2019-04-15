"""empty message

Revision ID: 4a3f60995f95
Revises: 
Create Date: 2019-04-14 22:18:57.002550

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '4a3f60995f95'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('moviereviews',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('movietextcolumn', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sqlite_autoincrement=True
    )
    op.create_table('sentiments',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('sentimentcolumn', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('spam_messages',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('spam_messages_column', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('spamham',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('spamham_column', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('email', sa.String(length=64), nullable=True),
    sa.Column('username', sa.String(length=64), nullable=True),
    sa.Column('password_hash', sa.String(length=128), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
    op.drop_table('spamham')
    op.drop_table('spam_messages')
    op.drop_table('sentiments')
    op.drop_table('moviereviews')
    # ### end Alembic commands ###