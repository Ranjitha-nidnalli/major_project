"""
Constructed example: chrF and BLEU score a 10x overdose above a valid paraphrase.

Reference, overdose and paraphrase are hand-written illustrations (the
paraphrase's Kannada is not reviewed by a native speaker). The point is the
ordering, not the exact values: string-overlap metrics reward copying the
reference's wording, and a one-character dose change keeps almost all of it.

Run: python eval/metric_overdose_demo.py
"""
from sacrebleu.metrics import BLEU, CHRF

REFERENCE = "ಕಾರ್ಬೆಂಡೈಜಿಮ್ 50 ಡಬ್ಲ್ಯೂ.ಪಿ ಔಷಧಿಯನ್ನು 1 ಗ್ರಾಂ/ಲೀಟರ್ ದರದಲ್ಲಿ ಬಳಸಿ, ಸೆಟ್ಸ್‌ಗಳನ್ನು 15 ನಿಮಿಷ ಅದ್ದಿ."
HYPOTHESES = {
    "overdose_10x": "ಕಾರ್ಬೆಂಡೈಜಿಮ್ 50 ಡಬ್ಲ್ಯೂ.ಪಿ ಔಷಧಿಯನ್ನು 10 ಗ್ರಾಂ/ಲೀಟರ್ ದರದಲ್ಲಿ ಬಳಸಿ, ಸೆಟ್ಸ್‌ಗಳನ್ನು 15 ನಿಮಿಷ ಅದ್ದಿ.",
    "valid_paraphrase": "ಸೆಟ್ಸ್‌ಗಳನ್ನು 15 ನಿಮಿಷ ಕಾರ್ಬೆಂಡೈಜಿಮ್ 50 ಡಬ್ಲ್ಯೂ.ಪಿ ದ್ರಾವಣದಲ್ಲಿ (ಪ್ರತಿ ಲೀಟರ್ ನೀರಿಗೆ 1 ಗ್ರಾಂ) ಅದ್ದಬೇಕು.",
}


def main():
    chrf, bleu = CHRF(), BLEU(effective_order=True)
    for name, hyp in HYPOTHESES.items():
        print(f"{name:18s} chrF={chrf.sentence_score(hyp, [REFERENCE]).score:5.1f} "
              f"BLEU={bleu.sentence_score(hyp, [REFERENCE]).score:5.1f}")


if __name__ == "__main__":
    main()
