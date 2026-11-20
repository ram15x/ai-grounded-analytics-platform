# Real public documents: plan (nothing downloaded)

Status: plan only. **No document has been fetched, and none will be without the owner saying so.**
The owner decided that no private documents are added and that a small set of real public
documents is planned for later.

## Why

The synthetic corpus is written by the project author and tests the author's own assumptions. A few
real public documents are a check against that: wording nobody here controlled, real tables and
real cross-references. They do not make a benchmark.

## Candidate sources (unverified)

- UK government consumer-returns and consumer-rights guidance, and VAT guidance, published on
  GOV.UK. Most GOV.UK text is published under the Open Government Licence v3.0 **unless a page
  says otherwise**. That is the owner's working assumption and **has not been verified per page**:
  pages can carry third-party material, logos and images that the licence does not cover.
- Nothing else is proposed. No URLs are listed here on purpose; they are chosen and checked with the
  owner before any fetch.

## Rules before a file is added

1. The owner approves the list of pages. I ask first; I do not fetch.
2. For each page, someone looks at the page's own licence statement and records it. A page with no
   clear licence, or with a licence outside the list in `tests/test_public_manifest.py`, is not used.
3. Only the licensed text is stored (no images, logos or third-party quotations), as `.md` or
   `.txt`, exactly as copied apart from line endings.
4. Each file gets an entry in `corpus/public/MANIFEST.yaml`: source URL, licence, licence URL,
   retrieval date, SHA-256 of the stored file, and the attribution sentence the licence asks for.
5. The attribution sentence also appears in the file itself or in the repository's notices.
6. Public documents are kept out of `corpus/synthetic`, ingested separately, and never mixed into
   the synthetic dev or held-out scores. Their own questions are written after the pages are chosen
   and labelled the same way (document plus anchor string).
7. Web pages change. The retrieval date and hash say which version was used; a re-fetch is a new
   entry, not an edit.

## What the repository checks today

`tests/test_public_manifest.py` enforces rules 3 to 5 mechanically (fields present, licence in the
allow-list, hash matches, no file without an entry, no entry without a file). With no documents it
checks only the shape of the empty manifest.

## Not decided

- Which pages. Which licence variants beyond OGL v3.0 are acceptable, if any.
- Who fetches: the owner pastes the text, or approves a specific fetch by me.
