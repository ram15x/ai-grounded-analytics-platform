# ADR-0001: Number verification is deterministic code, not an LLM judge

Status: Accepted (design)

Context: The product promise is that every displayed number is checked. An LLM judge can repeat the generator's mistakes and cannot be audited per number.

Decision: The verifier is plain Python. It extracts numbers by rules, ties them to stored query results, recomputes derived values with `Decimal`, and checks metric definitions from a YAML semantic layer. LLMs may propose claims and provenance; they never decide a verdict.

Consequences: Verdicts are reproducible and unit-testable, and the verifier can be mutation-tested. It cannot judge prose meaning or catch "right number, wrong question" (ARCHITECTURE 4.4). Extraction rules need ongoing maintenance.
