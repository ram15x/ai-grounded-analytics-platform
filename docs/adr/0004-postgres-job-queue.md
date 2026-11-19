# ADR-0004: Postgres-backed job queue

Status: Accepted (design), revisit if throughput demands

Context: Free hosting makes each extra service (Redis, a broker) a cost and an idle-behaviour risk. Render's free Key Value store loses data on restart.

Decision: Jobs live in a Postgres table claimed with `FOR UPDATE SKIP LOCKED`. Workers poll.

Consequences: One fewer service and transactional enqueue with business data. Lower throughput and no native scheduling; scheduling comes from n8n.
