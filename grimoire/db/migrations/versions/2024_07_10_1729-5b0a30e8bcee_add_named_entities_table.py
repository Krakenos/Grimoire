"""add named entities table

Revision ID: 5b0a30e8bcee
Revises: 7fc7a59c8cf5
Create Date: 2024-07-10 17:29:01.877376

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5b0a30e8bcee'
down_revision: Union[str, None] = '7fc7a59c8cf5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('spacy_named_entities',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('chat_id', sa.Integer(), nullable=False),
    sa.Column('entity_name', sa.String(), nullable=False),
    sa.Column('entity_label', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['chat_id'], ['message.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('spacy_named_entities')
    # ### end Alembic commands ###
