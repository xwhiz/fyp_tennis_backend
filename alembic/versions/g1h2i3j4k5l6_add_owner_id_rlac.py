"""Add owner_id for row-level ownership (RLAC)

Revision ID: g1h2i3j4k5l6
Revises: f6a7b8c9d0e1
Create Date: 2026-04-11 00:00:00.000000+00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "g1h2i3j4k5l6"
down_revision: Union[str, Sequence[str], None] = "f6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TASK_TABLES = (
    "background_tasks",
    "ball_tracks",
    "bounces",
    "direction_change_indices",
    "speeds",
    "thumbnails",
    "video_paths",
    "player_positions",
    "player_heatmap_data",
    "homography_matrices",
    "shot_annotations",
)


def upgrade() -> None:
    bind = op.get_bind()

    admin_id = bind.execute(
        sa.text("SELECT id FROM users WHERE role = 'admin' ORDER BY id LIMIT 1"),
    ).scalar_one_or_none()
    if admin_id is None:
        admin_id = bind.execute(sa.text("SELECT id FROM users ORDER BY id LIMIT 1")).scalar_one_or_none()

    for table in TASK_TABLES:
        op.add_column(
            table,
            sa.Column("owner_id", sa.String(length=36), nullable=True),
        )
        op.create_index(op.f(f"ix_{table}_owner_id"), table, ["owner_id"], unique=False)

    op.add_column(
        "model_metrics",
        sa.Column("owner_id", sa.String(length=36), nullable=True),
    )
    op.create_index(op.f("ix_model_metrics_owner_id"), "model_metrics", ["owner_id"], unique=False)

    if admin_id is not None:
        for table in TASK_TABLES:
            bind.execute(
                sa.text(f"UPDATE {table} SET owner_id = :oid WHERE owner_id IS NULL"),
                {"oid": admin_id},
            )
        bind.execute(
            sa.text("UPDATE model_metrics SET owner_id = :oid WHERE owner_id IS NULL"),
            {"oid": admin_id},
        )

    for table in TASK_TABLES:
        null_ct = bind.execute(
            sa.text(f"SELECT COUNT(*) FROM {table} WHERE owner_id IS NULL"),
        ).scalar()
        if null_ct and int(null_ct) > 0:
            raise RuntimeError(
                f"g1h2i3j4k5l6: Table {table} has rows without owner_id; "
                "ensure at least one user exists before migrating.",
            )

    for table in TASK_TABLES:
        op.create_foreign_key(op.f(f"fk_{table}_owner_id_users"), table, "users", ["owner_id"], ["id"], ondelete="CASCADE")

    op.create_foreign_key(
        op.f("fk_model_metrics_owner_id_users"),
        "model_metrics",
        "users",
        ["owner_id"],
        ["id"],
        ondelete="SET NULL",
    )

    for table in TASK_TABLES:
        op.alter_column(table, "owner_id", existing_type=sa.String(length=36), nullable=False)


def downgrade() -> None:
    op.drop_constraint(op.f("fk_model_metrics_owner_id_users"), "model_metrics", type_="foreignkey")
    for table in reversed(TASK_TABLES):
        op.drop_constraint(op.f(f"fk_{table}_owner_id_users"), table, type_="foreignkey")

    for table in reversed(TASK_TABLES):
        op.drop_index(op.f(f"ix_{table}_owner_id"), table_name=table)
        op.drop_column(table, "owner_id")

    op.drop_index(op.f("ix_model_metrics_owner_id"), table_name="model_metrics")
    op.drop_column("model_metrics", "owner_id")
