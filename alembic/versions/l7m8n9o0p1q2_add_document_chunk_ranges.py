"""add document chunk page and line ranges

Revision ID: l7m8n9o0p1q2
Revises: k6l7m8n9o0p1
Create Date: 2026-04-27 00:00:00.000000+00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "l7m8n9o0p1q2"
down_revision: Union[str, Sequence[str], None] = "k6l7m8n9o0p1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "document_chunks",
        sa.Column("page_start", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "document_chunks",
        sa.Column("page_end", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "document_chunks",
        sa.Column("line_start", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "document_chunks",
        sa.Column("line_end", sa.Integer(), nullable=False, server_default="1"),
    )
    op.alter_column("document_chunks", "page_start", server_default=None)
    op.alter_column("document_chunks", "page_end", server_default=None)
    op.alter_column("document_chunks", "line_start", server_default=None)
    op.alter_column("document_chunks", "line_end", server_default=None)


def downgrade() -> None:
    op.drop_column("document_chunks", "line_end")
    op.drop_column("document_chunks", "line_start")
    op.drop_column("document_chunks", "page_end")
    op.drop_column("document_chunks", "page_start")
