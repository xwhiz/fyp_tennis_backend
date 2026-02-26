"""Add multiple players support and discard flag to shot_annotations

Revision ID: c8d9e0f1a2b3
Revises: b7f8c9d0e1f2
Create Date: 2025-01-21 00:00:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c8d9e0f1a2b3'
down_revision: Union[str, Sequence[str], None] = 'b7f8c9d0e1f2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add discarded column
    op.add_column('shot_annotations',
                  sa.Column('discarded', sa.Boolean(), nullable=False, server_default='false'))
    
    # Add player_image_paths column for storing multiple player images
    op.add_column('shot_annotations',
                  sa.Column('player_image_paths', sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('shot_annotations', 'player_image_paths')
    op.drop_column('shot_annotations', 'discarded')

