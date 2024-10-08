"""add frozen and enabled to knowledge

Revision ID: 603509951db4
Revises: b9882f6c79fc
Create Date: 2024-08-16 20:04:04.135024

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "603509951db4"
down_revision: str | None = "b9882f6c79fc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column("knowledge", sa.Column("enabled", sa.Boolean(), nullable=True))
    op.add_column("knowledge", sa.Column("frozen", sa.Boolean(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("knowledge", "frozen")
    op.drop_column("knowledge", "enabled")
    # ### end Alembic commands ###
