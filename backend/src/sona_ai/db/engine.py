from sqlalchemy import create_engine, event
from sqlalchemy import text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from sona_ai.core import PROJECT_ROOT


DB_PATH = PROJECT_ROOT / "backend" / "data" / "sona.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


@event.listens_for(engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def init_db():
    from sona_ai.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _migrate_sqlite()


def _migrate_sqlite():
    with engine.begin() as connection:
        columns = {
            row[1]
            for row in connection.execute(text("PRAGMA table_info(recordings)"))
        }
        if "device" not in columns:
            connection.execute(
                text("ALTER TABLE recordings ADD COLUMN device VARCHAR(32) NOT NULL DEFAULT 'auto'")
            )
