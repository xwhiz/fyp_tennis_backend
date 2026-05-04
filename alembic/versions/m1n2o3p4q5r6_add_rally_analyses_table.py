"""add rally analyses table

Revision ID: m1n2o3p4q5r6
Revises: l7m8n9o0p1q2
Create Date: 2026-05-04 00:00:00.000000+00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "m1n2o3p4q5r6"
down_revision: Union[str, Sequence[str], None] = "l7m8n9o0p1q2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "rally_analyses",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.BigInteger(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("public_payload", sa.JSON(), nullable=False),
        sa.Column("internal_payload", sa.JSON(), nullable=False),
        sa.Column(
            "owner_id",
            sa.String(length=36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    )
    op.create_index("ix_rally_analyses_task_id", "rally_analyses", ["task_id"], unique=True)
    op.create_index("ix_rally_analyses_owner_id", "rally_analyses", ["owner_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_rally_analyses_owner_id", table_name="rally_analyses")
    op.drop_index("ix_rally_analyses_task_id", table_name="rally_analyses")
    op.drop_table("rally_analyses")
