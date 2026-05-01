# Custom Postgres image: PostGIS 3.5 + pgvector, both on arm64.
#
# imresamu/postgis is the only widely-used arm64-native PostGIS image.
# pgvector/pgvector ships pgvector but no PostGIS.
# We multi-stage COPY pgvector's compiled extension into the PostGIS image.

FROM pgvector/pgvector:pg16 AS pgvector_src

FROM imresamu/postgis:16-3.5

# Bring in pgvector — the .so library + the extension control + SQL files.
COPY --from=pgvector_src /usr/lib/postgresql/16/lib/vector.so \
                         /usr/lib/postgresql/16/lib/vector.so
COPY --from=pgvector_src /usr/share/postgresql/16/extension/vector.control \
                         /usr/share/postgresql/16/extension/vector.control
COPY --from=pgvector_src /usr/share/postgresql/16/extension/vector--*.sql \
                         /usr/share/postgresql/16/extension/
