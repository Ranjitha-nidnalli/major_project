# Record confirmation protocol

Applies to every fact record before the bot may serve it (Tier A). This is a
human task: nothing here is automated, and no script may set a record to
CONFIRMED.

## What "confirmed" means

A reviewer has checked that the record **matches the cited page of the source
document**: the same product, dose, unit, water volume, crop stage and region.
It does **not** mean the reviewer judged the advice agronomically correct.
Say exactly this in the paper.

## Who reviews

- **Reviewer 1:** project owner.
- **Reviewer 2:** a second person (friend at UAS Mandya), working alone,
  without seeing Reviewer 1's marks.
- **Spot check (wanted, not required to serve):** one UAS faculty member or KVK
  officer reviews a random sample of 10 to 15 records. Record their name,
  role and date.

## Steps per record (use `review_sheet.csv` columns)

1. Open the source at `printed_page` (the page printed on the book, not the PDF viewer page number).
2. Compare `text_kn`, `per_litre`, `per_acre`, `water_per_acre` and any unit with the source line, character by character for numbers and units.
3. If a pesticide is named, look it up in CIB&RC for this crop and fill `REVIEW: pesticide currently registered?` with `yes`, `no` or `n.a.`. Use `registration_helper.py` to find the entry; the human fills the answer.
4. Set `REVIEW: status`:
   - `CONFIRMED`: everything matches the source, and the pesticide is registered or `n.a.`.
   - `WRONG`: something differs. Put the corrected text or dose in `REVIEW: correct text or dose if WRONG` and the page where the right value is.
   - `UNSURE`: the source is unclear, the page is missing, or the product or unit is ambiguous. The record is not served.
5. Write a note for every `WRONG` and `UNSURE`.
6. Do not edit `record_hash`. After any change to a record, the hash is recomputed and the record must be re-confirmed.

## Status values and what the bot does

| Status | Served by the bot? |
| --- | --- |
| CONFIRMED by reviewer 1 only | Yes during development; counted as "single-reviewer" in reports |
| CONFIRMED by both reviewers | Yes; counted as "double-confirmed" |
| WRONG or UNSURE (either reviewer) | No |
| Registered = no | No, even if CONFIRMED |
| Hash mismatch | No |

The paper reports the number of records in each row of this table.

## Disagreements

If the two reviewers differ, they open the source together and agree on one
status. Keep both original marks in the sheet (two review columns, or two
files) so agreement can still be measured. Never overwrite the first mark.

## Agreement

After both reviews, before reconciling, compute on all records both reviewers marked:

- Percent agreement = records with the same status / all records.
- Cohen's kappa on the three statuses (CONFIRMED, WRONG, UNSURE).

Report both, with the number of records. If there are only a few non-CONFIRMED
marks, kappa is unstable; say so and rely on percent agreement and a list of
every disagreement.

## Refresh

- Each record carries its source edition year. A new edition triggers a re-review of the affected crop.
- Re-check CIB&RC registration for every pesticide record at least once a year and before any evaluation run.
- A record found wrong in use is switched off first, then a regression test is added, then it is re-reviewed.
