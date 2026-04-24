"""at_tag on users, friend_relations, opponent_id on background_tasks

Revision ID: i3j4k5l6m7n8
Revises: h2i3j4k5l6m7
Create Date: 2026-04-24 00:00:00.000000+00:00
"""

from __future__ import annotations

import re
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "i3j4k5l6m7n8"
down_revision: Union[str, Sequence[str], None] = "h2i3j4k5l6m7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _base_slug_from_email(email: str) -> str:
    local = (email or "").split("@", 1)[0].lower()
    slug = re.sub(r"[^a-z0-9_]+", "_", local)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        slug = "user"
    return slug[:32]


def upgrade() -> None:
    op.add_column("users", sa.Column("atTag", sa.String(length=64), nullable=True))
    op.create_index(op.f("ix_users_atTag"), "users", ["atTag"], unique=False)

    conn = op.get_bind()
    rows = conn.execute(sa.text('SELECT id, email FROM users ORDER BY "createdAt"')).fetchall()
    used: set[str] = set()
    for row in rows:
        uid, email = row[0], row[1]
        base = _base_slug_from_email(email)
        candidate = base
        n = 2
        while candidate in used:
            candidate = f"{base}_{n}"[:64]
            n += 1
        used.add(candidate)
        conn.execute(
            sa.text('UPDATE users SET "atTag" = :tag WHERE id = :id'),
            {"tag": candidate, "id": uid},
        )

    op.alter_column("users", "atTag", existing_type=sa.String(length=64), nullable=False)
    op.drop_index(op.f("ix_users_atTag"), table_name="users")
    op.create_index(op.f("ix_users_atTag"), "users", ["atTag"], unique=True)

    op.create_table(
        "friend_relations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("friend_id", sa.String(length=36), nullable=False),
        sa.Column("is_accepted", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["friend_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "friend_id", name="uq_friend_relations_user_friend"),
    )
    op.create_index(op.f("ix_friend_relations_friend_id"), "friend_relations", ["friend_id"], unique=False)
    op.create_index(op.f("ix_friend_relations_user_id"), "friend_relations", ["user_id"], unique=False)
    op.create_index(
        "ix_friend_relations_friend_accepted",
        "friend_relations",
        ["friend_id", "is_accepted"],
        unique=False,
    )
    op.create_index(
        "ix_friend_relations_user_accepted",
        "friend_relations",
        ["user_id", "is_accepted"],
        unique=False,
    )

    op.add_column(
        "background_tasks",
        sa.Column("opponent_id", sa.String(length=36), nullable=True),
    )
    op.create_foreign_key(
        "fk_background_tasks_opponent_id_users",
        "background_tasks",
        "users",
        ["opponent_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(op.f("ix_background_tasks_opponent_id"), "background_tasks", ["opponent_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_background_tasks_opponent_id"), table_name="background_tasks")
    op.drop_constraint("fk_background_tasks_opponent_id_users", "background_tasks", type_="foreignkey")
    op.drop_column("background_tasks", "opponent_id")

    op.drop_index("ix_friend_relations_user_accepted", table_name="friend_relations")
    op.drop_index("ix_friend_relations_friend_accepted", table_name="friend_relations")
    op.drop_index(op.f("ix_friend_relations_user_id"), table_name="friend_relations")
    op.drop_index(op.f("ix_friend_relations_friend_id"), table_name="friend_relations")
    op.drop_table("friend_relations")

    op.drop_index(op.f("ix_users_atTag"), table_name="users")
    op.drop_column("users", "atTag")
