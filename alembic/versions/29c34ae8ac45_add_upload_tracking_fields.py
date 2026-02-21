"""Add upload tracking fields to background_tasks

Revision ID: 29c34ae8ac45
Revises: 5821bf69f4fa
Create Date: 2026-02-19 00:00:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '29c34ae8ac45'
down_revision: Union[str, Sequence[str], None] = '5821bf69f4fa'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add total_upload_size column
    op.add_column('background_tasks',
                  sa.Column('total_upload_size', sa.BigInteger(), nullable=False, server_default='0'))
    
    # Add uploaded_size column
    op.add_column('background_tasks',
                  sa.Column('uploaded_size', sa.BigInteger(), nullable=False, server_default='0'))
    
    # Add is_uploaded_fully column
    op.add_column('background_tasks',
                  sa.Column('is_uploaded_fully', sa.Boolean(), nullable=False, server_default='true'))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('background_tasks', 'is_uploaded_fully')
    op.drop_column('background_tasks', 'uploaded_size')
    op.drop_column('background_tasks', 'total_upload_size')

