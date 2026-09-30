# Expert review sheets (THE PLAN, TODO.md)

Three sheets for the agricultural expert. Everything in them is a HUMAN task
under CLAUDE.md: nothing here is filled in, suggested, or checked by an AI,
and nothing an AI writes may be presented as expert-verified.

Open the CSVs in Excel or Google Sheets (they are UTF-8, so Kannada shows correctly).

## 1. `corpus_fact_sheet.csv`: check the knowledge base (days 3–6, ~2–3 h)

One row per piece of text the chatbot can answer from, exactly as the
chatbot sees it (43 rows). Pest, disease, fertilizer and weed rows come first;
those carry the chemical doses and matter most.

For each row:
- **facts_correct**: are the chemical names, doses, units, and timings right?
- **correction**: if not, write the exact corrected text. The chatbot will
  use your wording as-is; nobody will "fix it up" afterwards.
- **valid_for_karnataka**: some text is from Tamil Nadu sources
  (`mentions_tamil_nadu = yes`). Is the advice valid for Karnataka farmers?
- **keep_in_corpus**: `remove` takes the row out of the knowledge base.
- **source**: what you checked against (e.g. a university crop recommendations book and page).
  This is what the paper will cite.

Regenerate the sheet with `cd backend && python eval/expert/build_corpus_fact_sheet.py`.

## 2. `heldout_questions_TEMPLATE.csv`: write new test questions (days 3–6)

Copy it to `heldout_questions.csv` and fill in about **40 questions**, written
the way a farmer would ask them, in Kannada.

- **At least 15 must be questions the system should refuse**: things the
  knowledge base does not cover, such as other crops, pests or diseases not in
  it, prices, government schemes, weather, or questions needing a field visit.
  Include a few that look close to something covered (e.g. a pest that is
  *not* in the knowledge base), since those are the dangerous ones.
- For the rest, give the answer a correct chatbot should give, from the
  knowledge base after your corrections.
- Spread questions across pest, disease, fertilizer, weed and general topics.
- **Do not show these questions to the AI assistant, and do not test them in the
  chatbot**, before the day-7 run. They must stay unseen to be a fair test.

## 3. Rating sheet (days 8–9, ~2 h)

Created after the day-7 run with
`cd backend && python eval/expert/export_rating_sheet.py eval/runs/live/<run>.jsonl`.
For each question it shows what the farmer received and, if the system refused, what
the model had written before it was stopped. Rate correctness, whether it should have
refused, whether any dose is right, whether it could cause harm, and Kannada clarity.
No automatic scores are shown, so they don't influence the rating.
