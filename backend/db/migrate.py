"""Startup schema migration - adds columns missing from older DBs.

create_all creates new tables but never alters existing ones, so DBs made
before a column was added (e.g. conversations.topic) 500 on upgrade.
ensure_schema runs after create_all in lifespan and backfills missing
columns via ALTER TABLE ADD COLUMN. Idempotent, never raises.
"""

from sqlalchemy import inspect, text

from core.logging import get_logger
from db.models import Base

logger = get_logger(__name__)


def _column_ddl(column, dialect) -> str:
    """Render SQLite ADD COLUMN fragment for a mapped column."""
    type_sql = str(column.type.compile(dialect=dialect))
    parts = [f'"{column.name}"', type_sql]
    if column.server_default is not None:
        arg = column.server_default.arg
        try:
            if hasattr(arg, "compile"):
                default = str(
                    arg.compile(
                        dialect=dialect, compile_kwargs={"literal_binds": True}
                    )
                )
            else:
                default = str(arg)
        except Exception:
            default = ""
        if default:
            parts.append(f"DEFAULT {default}")
    elif column.default is not None:
        arg = getattr(column.default, "arg", None)
        if arg is not None and not callable(arg):
            if isinstance(arg, str):
                escaped = arg.replace("'", "''")
                parts.append(f"DEFAULT '{escaped}'")
            elif isinstance(arg, bool):
                parts.append(f"DEFAULT {1 if arg else 0}")
            else:
                parts.append(f"DEFAULT {arg}")
    if not column.nullable:
        has_default = any(p.startswith("DEFAULT") for p in parts)
        # SQLite rejects ADD COLUMN NOT NULL without a non-null default
        # when rows exist, so keep it nullable rather than fail startup.
        if has_default:
            parts.append("NOT NULL")
    return " ".join(parts)


def ensure_schema(engine) -> list[str]:
    """Add columns present in models but missing on disk. Returns added list."""
    added: list[str] = []
    try:
        insp = inspect(engine)
        existing_tables = set(insp.get_table_names())
    except Exception as e:
        logger.warning("schema migration skipped: inspect failed: %s", e)
        return added
    for table_name, table in Base.metadata.tables.items():
        if table_name not in existing_tables:
            continue  # create_all owns new tables
        try:
            existing = {c["name"] for c in insp.get_columns(table_name)}
        except Exception as e:
            logger.warning("schema migration skipped for table %s: %s", table_name, e)
            continue
        for column in table.columns:
            if column.name in existing:
                continue
            ddl = _column_ddl(column, engine.dialect)
            stmt = f'ALTER TABLE "{table_name}" ADD COLUMN {ddl}'
            try:
                with engine.begin() as conn:
                    conn.execute(text(stmt))
                    if column.index:
                        idx = f"ix_{table_name}_{column.name}"
                        conn.execute(
                            text(
                                f'CREATE INDEX IF NOT EXISTS "{idx}" '
                                f'ON "{table_name}" ("{column.name}")'
                            )
                        )
                added.append(f"{table_name}.{column.name}")
                logger.info("migrated column %s.%s", table_name, column.name)
            except Exception as e:
                logger.warning(
                    "schema migration failed for column %s.%s: %s",
                    table_name,
                    column.name,
                    e,
                )
    if _migrate_topic_to_tag(engine):
        added.append("conversations.tag<topic")
    return added


def _migrate_topic_to_tag(engine) -> bool:
    """Rename conversations.topic to tag, preserving values. Idempotent."""
    try:
        with engine.begin() as conn:
            cols = conn.execute(text("PRAGMA table_info(conversations)")).all()
        names = {c[1] for c in cols}
        if "topic" not in names:
            return False
        with engine.begin() as conn:
            if "tag" not in names:
                conn.execute(text('ALTER TABLE conversations ADD COLUMN "tag" VARCHAR(64)'))
                conn.execute(
                    text('CREATE INDEX IF NOT EXISTS "ix_conversations_tag" '
                         'ON "conversations" ("tag")')
                )
            conn.execute(
                text("UPDATE conversations SET tag = topic "
                     "WHERE topic IS NOT NULL AND tag IS NULL")
            )
            conn.execute(text("ALTER TABLE conversations DROP COLUMN topic"))
        logger.info("migrated conversations.topic to tag")
        return True
    except Exception as e:
        logger.warning("topic to tag migration skipped: %s", e)
        return False
