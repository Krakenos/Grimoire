"""add segmented memories

Revision ID: f6391e0dc0e5
Revises: 7c91039c0bda
Create Date: 2025-01-09 13:24:57.880478

"""

from collections.abc import Sequence

import pgvector
import sqlalchemy as sa
import sqlalchemy_utils
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f6391e0dc0e5"
down_revision: str | None = "7c91039c0bda"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "segmented_memories",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("chat_id", sa.Integer(), nullable=False),
        sa.Column("summary", sqlalchemy_utils.types.encrypted.encrypted_type.StringEncryptedType(), nullable=False),
        sa.Column("vector_embedding", pgvector.sqlalchemy.vector.VECTOR(), nullable=True),
        sa.Column("created_date", sa.DateTime(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["chat_id"],
            ["chat.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "segmented_memories_message",
        sa.Column("segmented_memory_id", sa.Integer(), nullable=False),
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["message.id"],
        ),
        sa.ForeignKeyConstraint(
            ["segmented_memory_id"],
            ["segmented_memories.id"],
        ),
        sa.PrimaryKeyConstraint("segmented_memory_id", "message_id"),
    )
    op.add_column("chat", sa.Column("segmented_memory_interval", sa.Integer(), nullable=False, server_default="5"))
    op.add_column("chat", sa.Column("segmented_memory_messages", sa.Integer(), nullable=False, server_default="10"))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("chat", "segmented_memory_messages")
    op.drop_column("chat", "segmented_memory_interval")
    op.drop_table("segmented_memories_message")
    op.drop_table("segmented_memories")
    # ### end Alembic commands ###
