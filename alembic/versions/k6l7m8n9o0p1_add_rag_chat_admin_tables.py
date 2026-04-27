"""add rag admin chat tables

Revision ID: k6l7m8n9o0p1
Revises: j4k5l6m7n8o9
Create Date: 2026-04-27 00:00:00.000000+00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

try:
    from pgvector.sqlalchemy import Vector
except Exception:  # pragma: no cover
    Vector = None


revision: str = "k6l7m8n9o0p1"
down_revision: Union[str, Sequence[str], None] = "j4k5l6m7n8o9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _embedding_type(bind):
    if bind.dialect.name == "postgresql" and Vector is not None:
        return Vector(4096)
    return sa.JSON()


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.add_column("users", sa.Column("contextSummary", sa.Text(), nullable=True))

    op.create_table(
        "system_prompts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_by", sa.String(length=36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    )
    op.create_index("ix_system_prompts_created_by", "system_prompts", ["created_by"], unique=False)
    op.create_index("ix_system_prompts_name", "system_prompts", ["name"], unique=True)

    op.create_table(
        "knowledge_documents",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("governing_body", sa.String(length=64), nullable=False),
        sa.Column("competition", sa.String(length=128), nullable=True),
        sa.Column("season_year", sa.Integer(), nullable=True),
        sa.Column("source_file_path", sa.String(length=512), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("mime_type", sa.String(length=128), nullable=False),
        sa.Column("file_size", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("ingestion_status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("ingestion_error", sa.Text(), nullable=True),
        sa.Column("uploaded_by", sa.String(length=36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    )
    op.create_index("ix_knowledge_documents_governing_body", "knowledge_documents", ["governing_body"], unique=False)
    op.create_index("ix_knowledge_documents_competition", "knowledge_documents", ["competition"], unique=False)
    op.create_index("ix_knowledge_documents_season_year", "knowledge_documents", ["season_year"], unique=False)
    op.create_index("ix_knowledge_documents_ingestion_status", "knowledge_documents", ["ingestion_status"], unique=False)
    op.create_index("ix_knowledge_documents_uploaded_by", "knowledge_documents", ["uploaded_by"], unique=False)

    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("task_id", sa.BigInteger(), sa.ForeignKey("background_tasks.id", ondelete="SET NULL"), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="active"),
        sa.Column("last_stream_id", sa.String(length=36), nullable=True),
        sa.Column("summary_text", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    )
    op.create_index("ix_chat_sessions_user_id", "chat_sessions", ["user_id"], unique=False)
    op.create_index("ix_chat_sessions_task_id", "chat_sessions", ["task_id"], unique=False)
    op.create_index("ix_chat_sessions_last_stream_id", "chat_sessions", ["last_stream_id"], unique=False)

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(length=36), sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    )
    op.create_index("ix_chat_messages_session_id", "chat_messages", ["session_id"], unique=False)
    op.create_index("ix_chat_messages_role", "chat_messages", ["role"], unique=False)

    op.create_table(
        "chat_streams",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("session_id", sa.String(length=36), sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("prompt_message_id", sa.BigInteger(), sa.ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("assistant_message_id", sa.BigInteger(), sa.ForeignKey("chat_messages.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    )
    op.create_index("ix_chat_streams_session_id", "chat_streams", ["session_id"], unique=False)
    op.create_index("ix_chat_streams_user_id", "chat_streams", ["user_id"], unique=False)
    op.create_index("ix_chat_streams_prompt_message_id", "chat_streams", ["prompt_message_id"], unique=False)
    op.create_index("ix_chat_streams_status", "chat_streams", ["status"], unique=False)
    op.create_index("ix_chat_streams_assistant_message_id", "chat_streams", ["assistant_message_id"], unique=False)

    op.create_table(
        "chat_attachments",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(length=36), sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("message_id", sa.BigInteger(), sa.ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=False),
        sa.Column("attachment_type", sa.String(length=32), nullable=False),
        sa.Column("file_path", sa.String(length=512), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("mime_type", sa.String(length=128), nullable=False),
        sa.Column("file_size", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    )
    op.create_index("ix_chat_attachments_session_id", "chat_attachments", ["session_id"], unique=False)
    op.create_index("ix_chat_attachments_message_id", "chat_attachments", ["message_id"], unique=False)

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("document_id", sa.BigInteger(), sa.ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("embedding", _embedding_type(bind), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    )
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"], unique=False)

    op.create_table(
        "game_stat_embeddings",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.BigInteger(), sa.ForeignKey("background_tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("player_user_id", sa.String(length=36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("player_scope", sa.String(length=32), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("embedding", _embedding_type(bind), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    )
    op.create_index("ix_game_stat_embeddings_task_id", "game_stat_embeddings", ["task_id"], unique=False)
    op.create_index("ix_game_stat_embeddings_player_user_id", "game_stat_embeddings", ["player_user_id"], unique=False)
    op.create_index("ix_game_stat_embeddings_player_scope", "game_stat_embeddings", ["player_scope"], unique=False)
    op.create_index("ix_game_stat_embeddings_source_type", "game_stat_embeddings", ["source_type"], unique=False)

    op.create_table(
        "user_memory_entries",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chat_session_id", sa.String(length=36), sa.ForeignKey("chat_sessions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("summary_text", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("embedding", _embedding_type(bind), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    )
    op.create_index("ix_user_memory_entries_user_id", "user_memory_entries", ["user_id"], unique=False)
    op.create_index("ix_user_memory_entries_chat_session_id", "user_memory_entries", ["chat_session_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_user_memory_entries_chat_session_id", table_name="user_memory_entries")
    op.drop_index("ix_user_memory_entries_user_id", table_name="user_memory_entries")
    op.drop_table("user_memory_entries")

    op.drop_index("ix_game_stat_embeddings_source_type", table_name="game_stat_embeddings")
    op.drop_index("ix_game_stat_embeddings_player_scope", table_name="game_stat_embeddings")
    op.drop_index("ix_game_stat_embeddings_player_user_id", table_name="game_stat_embeddings")
    op.drop_index("ix_game_stat_embeddings_task_id", table_name="game_stat_embeddings")
    op.drop_table("game_stat_embeddings")

    op.drop_index("ix_document_chunks_document_id", table_name="document_chunks")
    op.drop_table("document_chunks")

    op.drop_index("ix_chat_attachments_message_id", table_name="chat_attachments")
    op.drop_index("ix_chat_attachments_session_id", table_name="chat_attachments")
    op.drop_table("chat_attachments")

    op.drop_index("ix_chat_streams_assistant_message_id", table_name="chat_streams")
    op.drop_index("ix_chat_streams_status", table_name="chat_streams")
    op.drop_index("ix_chat_streams_prompt_message_id", table_name="chat_streams")
    op.drop_index("ix_chat_streams_user_id", table_name="chat_streams")
    op.drop_index("ix_chat_streams_session_id", table_name="chat_streams")
    op.drop_table("chat_streams")

    op.drop_index("ix_chat_messages_role", table_name="chat_messages")
    op.drop_index("ix_chat_messages_session_id", table_name="chat_messages")
    op.drop_table("chat_messages")

    op.drop_index("ix_chat_sessions_last_stream_id", table_name="chat_sessions")
    op.drop_index("ix_chat_sessions_task_id", table_name="chat_sessions")
    op.drop_index("ix_chat_sessions_user_id", table_name="chat_sessions")
    op.drop_table("chat_sessions")

    op.drop_index("ix_knowledge_documents_uploaded_by", table_name="knowledge_documents")
    op.drop_index("ix_knowledge_documents_ingestion_status", table_name="knowledge_documents")
    op.drop_index("ix_knowledge_documents_season_year", table_name="knowledge_documents")
    op.drop_index("ix_knowledge_documents_competition", table_name="knowledge_documents")
    op.drop_index("ix_knowledge_documents_governing_body", table_name="knowledge_documents")
    op.drop_table("knowledge_documents")

    op.drop_index("ix_system_prompts_name", table_name="system_prompts")
    op.drop_index("ix_system_prompts_created_by", table_name="system_prompts")
    op.drop_table("system_prompts")

    op.drop_column("users", "contextSummary")
