import sacrebleu

# 1. The exact, correct answer you expect (from your DB)
# Note: It must be inside a list of lists for sacrebleu
reference = [["ಕೆಂಪು ಕೊಳೆ ರೋಗವನ್ನು ಗ್ಲೋಮೆರೆಲ್ಲಾ ಟುಕುಮಾನೆನ್ಸಿಸ್ ಎಂಬ ಶಿಲೀಂಧ್ರವು ಉಂಟುಮಾಡುತ್ತದೆ."]]

# 2. The actual output your LLM generated
candidate = ["ಕೆಂಪು ಕೊಳೆ ರೋಗವನ್ನು ಗ್ಲೋಮೆರೆಲ್ಲಾ ಟುಕುಮಾನೆನ್ಸಿಸ್ ಉಂಟುಮಾಡುತ್ತದೆ."]

# 3. Calculate the score
bleu = sacrebleu.corpus_bleu(candidate, reference)

print(f"BLEU Score: {bleu.score:.2f} / 100")