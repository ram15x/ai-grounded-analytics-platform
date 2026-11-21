"""SQL guard: validate, rewrite and execute untrusted SQL safely.

Layers (each is tested on its own):
  1. Parse with sqlglot. Exactly one statement, SELECT / WITH...SELECT / set operations.
  2. Walk the whole tree: no DML/DDL/utility nodes, no INTO, no locking, no parameters,
     only allow-listed relations (plus the query's own CTEs), no table functions,
     no deny-listed functions.
  3. Execute the SQL regenerated from the validated tree, never the original text, so
     a parser differential between sqlglot and Postgres cannot smuggle anything in.
  4. Wrap in an outer LIMIT; run in a READ ONLY transaction with a statement timeout on a
     SELECT-only role that can see the `analytics` schema only (see warehouse/load.py).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import psycopg
import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from verified_analyst.config import connect_kwargs

logging.getLogger("sqlglot").setLevel(logging.ERROR)  # keep "falling back to Command" quiet

DIALECT = "postgres"
MAX_SQL_CHARS = 20_000
DEFAULT_ROW_LIMIT = 1000
DEFAULT_TIMEOUT_MS = 10_000

ALLOWED_RELATIONS = frozenset(
    {"analytics.sales_lines", "analytics.products", "analytics.customers", "analytics.invoices"}
)


def allowed_for(layer) -> frozenset[str]:
    """The relations a layer's queries may read. The retail layer keeps the original set; any
    other dataset may read its own table only, never another dataset's or the retail views."""
    rel = layer.source.relation
    return ALLOWED_RELATIONS if rel in ALLOWED_RELATIONS else frozenset({rel})


_FORBIDDEN_NODE_NAMES = [
    "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "AlterColumn",
    "Command", "Copy", "Set", "SetItem", "Use", "Grant", "Revoke", "Transaction", "Commit",
    "Rollback", "Into", "Lock", "Pragma", "Analyze", "Describe", "Show", "TruncateTable",
    "Attach", "Detach", "LoadData", "Cache", "Uncache", "Kill", "Declare", "Refresh",
    "Placeholder", "Parameter", "SessionParameter", "CurrentUser", "SessionUser",
    "CurrentSchema", "CurrentCatalog", "CurrentVersion", "Export", "Install", "Comment", "Rename",
]  # fmt: skip
FORBIDDEN_NODES: tuple[type, ...] = tuple(
    getattr(exp, n) for n in _FORBIDDEN_NODE_NAMES if hasattr(exp, n)
)

DENIED_FUNCTIONS = frozenset(
    {
        "set_config", "current_setting", "nextval", "setval", "currval", "lastval",
        "version", "generate_series", "generate_subscripts", "query_to_xml",
        "query_to_xml_and_xmlschema", "query_to_xmlschema", "table_to_xml",
        "table_to_xml_and_xmlschema", "table_to_xmlschema", "cursor_to_xml",
        "database_to_xml", "schema_to_xml", "inet_server_addr", "inet_server_port",
        "inet_client_addr", "inet_client_port", "current_database", "current_catalog",
        "glob", "getenv", "system", "sleep", "dblink", "unnest_file",
    }
)  # fmt: skip
DENIED_FUNCTION_PREFIXES = (
    "pg_", "lo_", "dblink", "read_", "parquet_", "duckdb_", "pragma_", "sniff_", "load_",
    "txid_", "xmin", "ts_debug", "brin_", "gin_", "bt_", "hash_",
)  # fmt: skip


class GuardError(ValueError):
    """Raised when SQL is rejected. `code` is a stable machine-readable reason."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class QueryTimeout(RuntimeError):
    pass


@dataclass
class GuardedSQL:
    original: str
    validated: str  # regenerated from the AST; this is what gets executed
    relations: list[str] = field(default_factory=list)


@dataclass
class QueryResult:
    query_sql: str
    executed_sql: str
    columns: list[str]
    rows: list[tuple]
    truncated: bool
    elapsed_ms: float


def _function_names(tree: exp.Expression) -> list[str]:
    names: list[str] = []
    for node in tree.walk():
        if isinstance(node, exp.Anonymous):
            names.append(str(node.name).lower())
        elif isinstance(node, exp.Func):
            names.append(node.sql_name().lower())
    return names


def _check_functions(tree: exp.Expression) -> None:
    for name in _function_names(tree):
        bare = name.split(".")[-1]
        if bare in DENIED_FUNCTIONS or bare.startswith(DENIED_FUNCTION_PREFIXES):
            raise GuardError("function_denied", f"function {bare!r} is not allowed")
    # schema-qualified calls (pg_catalog.pg_sleep) parse as Dot/Anonymous with a db part
    for node in tree.find_all(exp.Dot):
        if isinstance(node.expression, exp.Anonymous | exp.Func):
            bare = str(node.expression.name).lower()
            if bare in DENIED_FUNCTIONS or bare.startswith(DENIED_FUNCTION_PREFIXES):
                raise GuardError("function_denied", f"function {bare!r} is not allowed")


def _check_relations(tree: exp.Expression, allowed: frozenset[str]) -> list[str]:
    cte_names = {c.alias.lower() for c in tree.find_all(exp.CTE) if c.alias}
    seen: list[str] = []
    for t in tree.find_all(exp.Table):
        if not isinstance(t.this, exp.Identifier):
            raise GuardError("table_function", "table functions are not allowed in FROM")
        if t.args.get("catalog"):
            raise GuardError("relation_denied", "catalog-qualified names are not allowed")
        name, schema = t.name.lower(), (t.db or "").lower()
        if not schema and name in cte_names:
            continue
        full = f"{schema or 'analytics'}.{name}"
        if full not in allowed:
            raise GuardError("relation_denied", f"relation {full!r} is not allowed")
        seen.append(full)
    return seen


def validate_sql(
    sql: str,
    allowed_relations: frozenset[str] = ALLOWED_RELATIONS,
    max_chars: int = MAX_SQL_CHARS,
) -> GuardedSQL:
    if not isinstance(sql, str):
        raise GuardError("not_text", "SQL must be a string")
    if "\x00" in sql:
        raise GuardError("null_byte", "SQL contains a NUL byte")
    if len(sql) > max_chars:
        raise GuardError("too_long", f"SQL longer than {max_chars} characters")
    if not sql.strip():
        raise GuardError("empty", "empty SQL")
    try:
        statements = [s for s in sqlglot.parse(sql, dialect=DIALECT) if s is not None]
    except SqlglotError as exc:
        raise GuardError("parse_error", str(exc).splitlines()[0][:200]) from exc
    if not statements:
        raise GuardError("empty", "no statement found")
    if len(statements) > 1:
        raise GuardError(
            "multiple_statements", f"{len(statements)} statements; exactly one allowed"
        )
    root = statements[0]
    while isinstance(root, exp.Subquery):
        root = root.this
    if not isinstance(root, exp.Select | exp.SetOperation):
        raise GuardError("not_select", f"only SELECT is allowed, got {type(root).__name__}")
    for node in root.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise GuardError("forbidden_construct", f"{type(node).__name__} is not allowed")
        if isinstance(node, exp.Select):
            if not node.expressions:
                raise GuardError("parse_error", "SELECT has an empty select list")
            if node.args.get("into"):
                raise GuardError("forbidden_construct", "SELECT INTO is not allowed")
            if node.args.get("locks"):
                raise GuardError("forbidden_construct", "row locking is not allowed")
    _check_functions(root)
    relations = _check_relations(root, allowed_relations)
    return GuardedSQL(
        original=sql, validated=root.sql(dialect=DIALECT, comments=False), relations=relations
    )


def wrap_limit(validated_sql: str, row_limit: int) -> str:
    if (
        not isinstance(row_limit, int)
        or isinstance(row_limit, bool)
        or not 1 <= row_limit <= 100_000
    ):
        raise GuardError("bad_limit", "row_limit must be an integer between 1 and 100000")
    return f"SELECT * FROM ({validated_sql}) AS _va_q LIMIT {row_limit + 1}"


def execute_guarded(
    readonly_url: str,
    sql: str,
    row_limit: int = DEFAULT_ROW_LIMIT,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    allowed_relations: frozenset[str] = ALLOWED_RELATIONS,
) -> QueryResult:
    guarded = validate_sql(sql, allowed_relations)
    executable = wrap_limit(guarded.validated, row_limit)
    if (
        not isinstance(timeout_ms, int)
        or isinstance(timeout_ms, bool)
        or not 1 <= timeout_ms <= 60_000
    ):
        raise GuardError("bad_timeout", "timeout_ms must be an integer between 1 and 60000")
    start = time.perf_counter()
    connect = connect_kwargs(readonly_url)
    try:
        with psycopg.connect(readonly_url, **connect) as conn:
            conn.read_only = True
            with conn.transaction():
                conn.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)}")
                conn.execute("SET LOCAL lock_timeout = 2000")
                cur = conn.execute(executable)
                columns = [d.name for d in cur.description or []]
                rows = cur.fetchall()
    except psycopg.errors.QueryCanceled as exc:
        raise QueryTimeout(f"query exceeded {timeout_ms} ms") from exc
    elapsed = (time.perf_counter() - start) * 1000
    truncated = len(rows) > row_limit
    return QueryResult(
        query_sql=sql,
        executed_sql=executable,
        columns=columns,
        rows=[tuple(r) for r in rows[:row_limit]],
        truncated=truncated,
        elapsed_ms=elapsed,
    )
