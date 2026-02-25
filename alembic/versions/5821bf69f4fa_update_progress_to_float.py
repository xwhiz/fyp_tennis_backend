"""Update progress to float and remove total_steps

Revision ID: 5821bf69f4fa
Revises: a3b1c2d3e4f5
Create Date: 2026-02-18 19:58:27.160316+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5821bf69f4fa'
down_revision: Union[str, Sequence[str], None] = 'a3b1c2d3e4f5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Drop total_steps column
    op.drop_column('background_tasks', 'total_steps')
    
    # Alter progress column from Integer to Float
    op.alter_column('background_tasks', 'progress',
                    existing_type=sa.Integer(),
                    type_=sa.Float(),
                    existing_nullable=False,
                    server_default='0.0')


def downgrade() -> None:
    """Downgrade schema."""
    # Alter progress column back to Integer
    op.alter_column('background_tasks', 'progress',
                    existing_type=sa.Float(),
                    type_=sa.Integer(),
                    existing_nullable=False,
                    server_default='0')
    
    # Add total_steps column back
    op.add_column('background_tasks',
                  sa.Column('total_steps', sa.Integer(), nullable=False, server_default='10'))
