from thefuzz import fuzz

def align_words_old(user_words, ref_words):
    aligned = []
    ref_idx = 0
    for uw in user_words:
        best_match = None
        best_score = 0
        for j in range(ref_idx, min(ref_idx + 4, len(ref_words))):
            score = fuzz.ratio(uw["text"], ref_words[j]["text"])
            if score > best_score:
                best_score = score
                best_match = j
        if best_match is not None and best_score >= 50:
            aligned.append((uw, ref_words[best_match]))
            ref_idx = best_match + 1
        else:
            aligned.append((uw, None))
    return aligned

def align_words_new(user_words, ref_words):
    aligned = []
    ref_idx = 0
    for uw in user_words:
        best_match = None
        best_score = 0
        for j in range(ref_idx, min(ref_idx + 15, len(ref_words))):
            score = fuzz.ratio(uw["text"], ref_words[j]["text"])
            if score > best_score:
                best_score = score
                best_match = j
        if best_match is not None and best_score >= 50:
            aligned.append((uw, ref_words[best_match]))
            ref_idx = best_match + 1
        else:
            aligned.append((uw, None))
    return aligned

user_words = [{"text": "بسم"}, {"text": "الله"}, {"text": "الرحمن"}, {"text": "الرحيم"}, {"text": "الحمد"}, {"text": "لله"}, {"text": "رب"}, {"text": "العالمين"}]
ref_words = [{"text": "بسم"}, {"text": "الله"}, {"text": "الرحمن"}, {"text": "الرحيم"}, {"text": "a"}, {"text": "b"}, {"text": "c"}, {"text": "d"}, {"text": "e"}, {"text": "الحمد"}, {"text": "لله"}, {"text": "رب"}, {"text": "العالمين"}]

print("Old:")
for u, r in align_words_old(user_words, ref_words):
    print(f"{u['text']} -> {r['text'] if r else None}")

print("\nNew:")
for u, r in align_words_new(user_words, ref_words):
    print(f"{u['text']} -> {r['text'] if r else None}")
