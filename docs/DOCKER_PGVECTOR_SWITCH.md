# Switching the local container to pgvector

`docker-compose.yml` now uses `pgvector/pgvector:pg16` instead of `postgres:16`. Phase 2b needs the
`vector` extension; plain `postgres:16` does not ship it. Nothing in the code uses the extension
yet, so the switch can be made and checked before any vector code arrives.

## What happens to the data volume

The volume (`va_pgdata`, compose project `verified-analyst`, so Docker names it
`verified-analyst_va_pgdata`) is mounted at the same path in both images and holds a Postgres 16
data directory. The pgvector image is the official Postgres 16 image plus the extension files, which
live in the image and not in the volume. So the data should be read as it was, and the extension is
switched on per database later with `CREATE EXTENSION vector`.

**Verified from public sources (2026-11-20):** the pgvector Dockerfile is
`FROM postgres:$PG_MAJOR-$DEBIAN_CODENAME` with `DEBIAN_CODENAME=bookworm`, and it compiles pgvector
on top.

**UNVERIFIED:**
- Which Debian release your current `postgres:16` container runs. The official `postgres:16` tag
  follows the newest Debian, which may differ from bookworm, depending on when you pulled it. If the
  C library version changes between images, the sort order of text can change, and a text index
  built under the old order can then be wrong until it is rebuilt. This project has unique indexes
  on text columns (invoice numbers, stock codes). Check with step 1 and rebuild the indexes with
  step 6; both are cheap here.
- That `pgvector/pgvector` publishes a `pg16-trixie` tag. One third-party page lists
  `pg16-bookworm` and `pg16-trixie` tags, which I have not confirmed on Docker Hub. If step 1 says
  your current container is trixie, use `pg16-trixie` only if `docker pull` finds it.
- That `va_admin` may run `CREATE EXTENSION vector` (it is the container's superuser, so it should).
- That the existing container accepts `docker exec ... psql -U va_admin` without a password. The
  official image trusts local socket connections, which is how steps 1, 2 and 5 work. If it asks
  for a password, set `$env:PGPASSWORD` in that window first.

Everything in the database is rebuildable (`va load`, `va docs ingest`), so the backup in step 2 is
a convenience, not the only copy.

## Steps (Windows PowerShell)

Run in the repository folder. `POSTGRES_PASSWORD` must be set in the window because compose
refuses to start without it; it is only used when a volume is first created, so setting it again
does not change the password stored in the existing volume.

```powershell
cd C:\Users\ramna\OneDrive\Desktop\Projects\verified-analyst

# 1. Which Debian does the running container use, and what collation does the database have?
docker exec verified-analyst-postgres-1 sh -c "grep PRETTY_NAME /etc/os-release"
docker exec verified-analyst-postgres-1 psql -U va_admin -d verified_analyst -c "SELECT version();" -c "SHOW lc_collate;"

# 2. Back up (custom format, written inside the container, then copied out; do not use > for binary)
docker exec verified-analyst-postgres-1 pg_dump -U va_admin -Fc -d verified_analyst -f /tmp/va_backup.dump
docker cp verified-analyst-postgres-1:/tmp/va_backup.dump .\va_backup.dump

# 3. Remove the container but KEEP the volume (never add -v: that deletes the data)
docker compose down

# 4. Apply the patch so docker-compose.yml names the new image, then pull and start
docker compose pull
$env:POSTGRES_PASSWORD   # must print something; if empty, set it first
docker compose up -d --wait

# 5. Check: server version, the extension is available, and the data is still there
docker compose exec postgres psql -U va_admin -d verified_analyst -c "SELECT version();" -c "SELECT name, default_version FROM pg_available_extensions WHERE name = 'vector';" -c "SELECT count(*) FROM analytics.sales_lines;"

# 6. Rebuild every index, in case the C library changed under the text indexes
docker compose exec postgres psql -U va_admin -d verified_analyst -c "REINDEX DATABASE verified_analyst;"

# 7. Run the checks again
python -m pytest -q -o addopts=""
va profile-db
```

If step 5 shows no `vector` row, the old image is still running: `docker compose ps` and
`docker compose images` show the image in use.

## Going back

Set `image:` back to `postgres:16` and run `docker compose up -d --wait`. The volume is untouched by
either image apart from normal writes. If anything looks wrong, `docker compose down` and restore
from the backup into a fresh volume instead of editing data by hand:
`docker compose down -v` (deletes the volume, only after the backup file exists), `docker compose up -d --wait`,
`docker cp .\va_backup.dump verified-analyst-postgres-1:/tmp/va_backup.dump`, then
`docker exec verified-analyst-postgres-1 pg_restore -U va_admin -d verified_analyst /tmp/va_backup.dump`.

## Not changed here

`.github/workflows/ci.yml` still uses the `postgres:16` service image. It moves to the pgvector
image in the commit that adds the first vector test.
