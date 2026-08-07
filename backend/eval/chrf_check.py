import json
from rouge_score import rouge_scorer
import sacrebleu

QUESTIONS_PATH = "backend/eval/questions.json"

with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    qs = json.load(f)

refs = [q["expected_answer"] for q in qs]
ids = [q["id"] for q in qs]

rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
rouge_self = [rouge.score(r, r)["rougeL"].fmeasure for r in refs]

shuffled = refs[1:] + refs[:1]
rouge_un = [rouge.score(refs[i], shuffled[i])["rougeL"].fmeasure for i in range(len(refs))]

chrf_self = [sacrebleu.sentence_chrf(r, [r]).score / 100 for r in refs]
chrf_un = [sacrebleu.sentence_chrf(refs[i], [shuffled[i]]).score / 100 for i in range(len(refs))]

print("id, rouge_self, rouge_un, chrf_self, chrf_un")
for i in range(len(refs)):
    print(ids[i], f"{rouge_self[i]:.3f}", f"{rouge_un[i]:.3f}", f"{chrf_self[i]:.3f}", f"{chrf_un[i]:.3f}")

def avg(x):
    return sum(x) / len(x)

print("\nSUMMARY")
print("ROUGE-L self avg:", avg(rouge_self))
print("ROUGE-L unrelated avg:", avg(rouge_un))
print("chrF self avg:", avg(chrf_self))
print("chrF unrelated avg:", avg(chrf_un))