"""add tables for multiuser

Revision ID: 7cbb06f823fc
Revises: 4b9a98eb28b8
Create Date: 2024-03-25 15:54:45.021609

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import orm, text, select, insert
from sqlalchemy.ext.declarative import declarative_base

# revision identifiers, used by Alembic.
revision: str = '7cbb06f823fc'
down_revision: Union[str, None] = '4b9a98eb28b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

Base = declarative_base()


class Chat(Base):
    __tablename__ = 'chat'

    id = sa.Column(sa.Integer, primary_key=True)
    external_id = sa.Column(sa.String, nullable=False)
    user_id = sa.Column(sa.Integer, sa.ForeignKey('user.id'), nullable=False)


class User(Base):
    __tablename__ = 'user'

    id = sa.Column(sa.Integer, primary_key=True)
    external_id = sa.Column(sa.String, nullable=False)


class Message(Base):
    __tablename__ = 'message'

    id = sa.Column(sa.Integer, primary_key=True)
    chat_id = sa.Column(sa.String, nullable=False)


class KnowledgeOld(Base):
    __tablename__ = 'knowledge'

    id = sa.Column(sa.Integer(), nullable=False),
    chat_id = sa.Column(sa.String(), nullable=False)
    entity = sa.Column(sa.String(), nullable=False)
    entity_type = sa.Column(sa.String(), nullable=True)
    entity_label = sa.Column(sa.String(), nullable=True)
    summary = sa.Column(sa.String(), nullable=True)
    token_count = sa.Column(sa.Integer(), nullable=True)
    updated_date = sa.Column(sa.DateTime(), nullable=False)
    update_at = sa.Column(sa.Integer(), nullable=True)
    update_count = sa.Column(sa.Integer(), nullable=True)


class KnowledgeNew(Base):
    __tablename__ = 'knowledge_new'

    id = sa.Column(sa.Integer(), nullable=False),
    chat_id = sa.Column(sa.Integer(), nullable=False)
    entity = sa.Column(sa.String(), nullable=False)
    entity_type = sa.Column(sa.String(), nullable=True)
    entity_label = sa.Column(sa.String(), nullable=True)
    summary = sa.Column(sa.String(), nullable=True)
    token_count = sa.Column(sa.Integer(), nullable=True)
    updated_date = sa.Column(sa.DateTime(), nullable=False)
    update_at = sa.Column(sa.Integer(), nullable=True)
    update_count = sa.Column(sa.Integer(), nullable=True)


class MessageOld(Base):
    __tablename__ = 'message'

    id = sa.Column(sa.Integer(), nullable=False)
    chat_id = sa.Column(sa.String(), nullable=False)
    message_index = sa.Column(sa.Integer(), nullable=False)
    summary = sa.Column(sa.String(), nullable=True)
    message = sa.Column(sa.String(), nullable=False)
    summary_tokens = sa.Column(sa.Integer(), nullable=True)
    message_tokens = sa.Column(sa.Integer(), nullable=True)
    created_date = sa.Column(sa.DateTime(), nullable=False)


class MessageNew(Base):
    __tablename__ = 'message_new'

    id = sa.Column(sa.Integer(), nullable=False)
    chat_id = sa.Column(sa.Integer(), nullable=False)
    message_index = sa.Column(sa.Integer(), nullable=False)
    summary = sa.Column(sa.String(), nullable=True)
    message = sa.Column(sa.String(), nullable=False)
    summary_tokens = sa.Column(sa.Integer(), nullable=True)
    message_tokens = sa.Column(sa.Integer(), nullable=True)
    created_date = sa.Column(sa.DateTime(), nullable=False)


def upgrade() -> None:
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    # Creating User table and adding default user
    op.create_table('user',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('external_id', sa.String(), nullable=False),
                    sa.PrimaryKeyConstraint('id')
                    )

    default_user = User(external_id='DEFAULT_USER')
    session.add(default_user)
    session.commit()

    # Creating Chat table and adding already existing chat ids
    op.create_table('chat',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('external_id', sa.String(), nullable=False),
                    sa.Column('user_id', sa.Integer(), nullable=False),
                    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
                    sa.PrimaryKeyConstraint('id')
                    )
    existing_chats = session.query(Message.chat_id).distinct()
    chats = {chat_id[0]: Chat(external_id=chat_id[0], user_id=default_user.id) for chat_id in existing_chats}
    session.add_all(chats)
    session.commit()

    # Migrating the data and modifying Knowledge table
    op.create_table('knowledge_new',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('chat_id', sa.Integer(), nullable=False),
                    sa.Column('entity', sa.String(), nullable=False),
                    sa.Column('entity_type', sa.String(), nullable=True),
                    sa.Column('entity_label', sa.String(), nullable=True),
                    sa.Column('summary', sa.String(), nullable=True),
                    sa.Column('token_count', sa.Integer(), nullable=True),
                    sa.Column('updated_date', sa.DateTime(), nullable=False),
                    sa.Column('update_at', sa.Integer(), nullable=True),
                    sa.Column('update_count', sa.Integer(), nullable=True),
                    sa.ForeignKeyConstraint(['chat_id'], ['chat.id'], ),
                    sa.PrimaryKeyConstraint('id'))

    query = select(KnowledgeOld)
    knowledge_entries = session.scalars(query)
    new_knowledge_entries = []

    for entry in knowledge_entries:
        new_entry = {
            'id': entry.id,
            'chat_id': chats[entry.chat_id].id,
            'entity': entry.entity,
            'entity_type': entry.entity_type,
            'entity_label': entry.entity_label,
            'summary': entry.summary,
            'token_count': entry.token_count,
            'updated_date': entry.updated_date,
            'update_at': entry.update_at,
            'update_count': entry.update_count,
        }
        new_knowledge_entries.append(new_entry)

    session.execute(insert(KnowledgeNew), new_knowledge_entries)
    session.commit()
    op.drop_table('knowledge')
    op.rename_table('knowledge_new', 'knowledge')

    # Migrating the data and modifying Message table
    op.create_table('message_new',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('chat_id', sa.Integer(), nullable=False),
                    sa.Column('message_index', sa.Integer(), nullable=False),
                    sa.Column('summary', sa.String(), nullable=True),
                    sa.Column('message', sa.String(), nullable=False),
                    sa.Column('summary_tokens', sa.Integer(), nullable=True),
                    sa.Column('message_tokens', sa.Integer(), nullable=True),
                    sa.Column('created_date', sa.DateTime(), nullable=False),
                    sa.ForeignKeyConstraint(['chat_id'], ['chat.id'], ),
                    sa.PrimaryKeyConstraint('id')
                    )
    query = select(MessageOld)
    message_entries = session.scalars(query)
    new_message_entries = []

    for entry in message_entries:
        new_entry = {
            'id': entry.id,
            'chat_id': chats[entry.chat_id].id,
            'message_index': entry.message_index,
            'summary': entry.summary,
            'message': entry.message,
            'summary_tokens': entry.summary_tokens,
            'message_tokens': entry.message_tokens,
            'created_date': entry.created_date
        }
        new_message_entries.append(new_entry)

    session.execute(insert(MessageNew), new_message_entries)
    op.drop_table('message')
    op.rename_table('message_new', 'message')

    # ### end Alembic commands ###


def downgrade() -> None:
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    chats = session.query(Chat).all()

    # Moving old ids into Message table
    op.drop_constraint(None, 'message', type_='foreignkey')
    op.add_column('message', sa.Column('old_chat_id', sa.VARCHAR(), nullable=True))
    for chat in chats:
        bind.execute(text('UPDATE message SET old_chat_id = :external_id WHERE chat_id = :chat_id'),
                     {
                         'chat_id': chat.id,
                         'external_id': chat.external_id
                     })
    op.drop_column('message', 'chat_id')
    op.alter_column('message', 'new_chat_id',
                    new_column_name='chat_id',
                    existing_nullable=True,
                    nullable=False)

    # Moving old ids into Knowledge table
    op.drop_constraint(None, 'knowledge', type_='foreignkey')
    op.add_column('knowledge', sa.Column('old_chat_id', sa.VARCHAR(), nullable=True))
    for chat in chats:
        bind.execute(text('UPDATE knowledge SET old_chat_id = :external_id WHERE chat_id = :chat_id'),
                     {
                         'chat_id': chat.id,
                         'external_id': chat.external_id
                     })
    op.drop_column('knowledge', 'chat_id')
    op.alter_column('knowledge', 'new_chat_id',
                    new_column_name='chat_id',
                    existing_nullable=True,
                    nullable=False)

    # Dropping new tables
    op.drop_table('chat')
    op.drop_table('user')
    # ### end Alembic commands ###
