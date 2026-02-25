"""Add shot_annotations and model_metrics tables

Revision ID: b7f8c9d0e1f2
Revises: 29c34ae8ac45
Create Date: 2025-01-20 00:00:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7f8c9d0e1f2'
down_revision: Union[str, Sequence[str], None] = '29c34ae8ac45'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create shot_annotations table
    op.create_table('shot_annotations',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('task_id', sa.BigInteger(), nullable=False),
    sa.Column('frame_index', sa.BigInteger(), nullable=False),
    sa.Column('player_position_top', sa.JSON(), nullable=True),
    sa.Column('player_position_bottom', sa.JSON(), nullable=True),
    sa.Column('ball_position', sa.JSON(), nullable=True),
    sa.Column('player_image_path', sa.String(), nullable=True),
    sa.Column('predicted_shot_type', sa.String(), nullable=False, server_default='unknown'),
    sa.Column('annotated_shot_type', sa.String(), nullable=False, server_default='unknown'),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    
    # Create model_metrics table
    op.create_table('model_metrics',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('training_status', sa.String(), nullable=False, server_default='not_trained'),
    sa.Column('accuracy', sa.Float(), nullable=True),
    sa.Column('precision', sa.Float(), nullable=True),
    sa.Column('recall', sa.Float(), nullable=True),
    sa.Column('f1_score', sa.Float(), nullable=True),
    sa.Column('total_samples', sa.BigInteger(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('model_metrics')
    op.drop_table('shot_annotations')



