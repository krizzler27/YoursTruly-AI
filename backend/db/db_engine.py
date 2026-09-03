from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

from config import config

# SQLite-only engine for local desktop app (NullPool — single-user, no idle handles)
engine = create_engine(
    config.DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
    poolclass=NullPool,
)

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