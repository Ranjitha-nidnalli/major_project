# Why ROUGE-L scores Kannada references 0.000 (K-014)

**Cause.** `rouge_score` tokenizes with `re.sub(r"[^a-z0-9]+", " ", text.lower())`
(`rouge_score.tokenize`). Every Kannada character is outside `a-z0-9`, so it is
replaced by a space and dropped. A reference with no ASCII letter or digit becomes an
empty token list, and ROUGE-L of anything against it, even against itself, is 0.000. A
reference that has a few digits or Latin words (a number, "Co 86032") is reduced to just
those tokens, so a Kannada answer is scored only on its digits. That can read as a perfect
1.000 for a sentence whose Kannada part is wrong.

This is a property of the default tokenizer, not of the model or the data. It explains the
earlier finding that 6 of 16 references scored 0.000 against themselves: those are the
references with no ASCII alphanumerics.

**Decision.** ROUGE-L is not used as quality or safety evidence for Kannada anywhere
(CLAUDE.md rule). Passing a custom tokenizer would fix the zeros but not the real problem:
Kannada is agglutinative, so word overlap is a poor measure, and no overlap metric detects
a 10x dose error. The evaluation runner (`eval/tier_eval.py`) reports counted outcomes
instead.

`tests/invariants/test_rouge_kannada.py` pins the cause so the claim in the paper can be
reproduced.
