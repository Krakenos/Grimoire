"""add setting_override table

Revision ID: 443e4f800e04
Revises: 0b519949f5fc
Create Date: 2026-07-26 17:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlalchemy_utils
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "443e4f800e04"
down_revision: str | None = "0b519949f5fc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "setting_override",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("section", sa.String(), nullable=False),
        sa.Column("value", sqlalchemy_utils.types.encrypted.encrypted_type.StringEncryptedType(), nullable=False),
        sa.Column("updated_date", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_setting_override_section"), "setting_override", ["section"], unique=True)


def downgrade() -> None:
    op.drop_index(op.f("ix_setting_override_section"), table_name="setting_override")
    op.drop_table("setting_override")
