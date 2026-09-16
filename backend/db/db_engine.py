from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, event
from sqlalchemy.pool import NullPool

from config import config

# SQLite-only engine for local desktop app (NullPool - single-user, no idle handles)
engine = create_engine(
    config.DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
    poolclass=NullPool,
)

# Enable FK cascade + WAL for concurrent reads during SSE streaming
@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise  # Re-raise error so the app framework is notified
    finally:
        db.close()  # Guarantees connection is returned to the pool