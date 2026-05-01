from collections.abc import Iterator
from contextlib import contextmanager

import psycopg

from beacon.config import get_settings


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    with psycopg.connect(get_settings().postgres_dsn) as conn:
        yield conn
