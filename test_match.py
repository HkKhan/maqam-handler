from thefuzz import fuzz

def align_words_dp(user_words, ref_words):
    n = len(user_words)
    m = len(ref_words)
    
    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    # Backtrack table to reconstruct alignment
    # 0 = Match/Sub, 1 = Insert (skip ref), 2 = Delete (skip user)
    ptr = [[0] * (m + 1) for _ in range(n + 1)]
    
    # Gap penalty
    GAP_PENALTY = -20
    
    # Initialize DP table
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + GAP_PENALTY
        ptr[i][0] = 2
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + GAP_PENALTY
        ptr[0][j] = 1
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = fuzz.ratio(user_words[i-1]["text"], ref_words[j-1]["text"])
            
            # If score is too low, we penalize it so it prefers a gap
            if score < 60:
                match = dp[i-1][j-1] - 30
            else:
                match = dp[i-1][j-1] + score
                
            delete = dp[i-1][j] + GAP_PENALTY
            insert = dp[i][j-1] + GAP_PENALTY
            
            best = max(match, delete, insert)
            dp[i][j] = best
            
            if best == match:
                ptr[i][j] = 0
            elif best == delete:
                ptr[i][j] = 2
            else:
                ptr[i][j] = 1
                
    # Backtrack
    i, j = n, m
    aligned = []
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ptr[i][j] == 0:
            score = fuzz.ratio(user_words[i-1]["text"], ref_words[j-1]["text"])
            if score >= 60:
                aligned.append((user_words[i-1], ref_words[j-1]))
            else:
                aligned.append((user_words[i-1], None))
            i -= 1
            j -= 1
        elif i > 0 and ptr[i][j] == 2:
            aligned.append((user_words[i-1], None))
            i -= 1
        else:
            j -= 1
            
    return aligned[::-1]

ref_text = "بسم الله الرحمن الرحيم الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين اياك نعبد و اياك نستعين اهدنا الصراط المستقيم صراط الذين انعمت عليهم غير المغضوب عليهم و لا الضالين"
ref_words = [{"text": w} for w in ref_text.split()]
user_text = "اعوذ بالله من الشيطان الرجيم بسم الله الرحمن الرحيم الحمد لله رب العالمين"
user_words = [{"text": w} for w in user_text.split()]

for u, r in align_words_dp(user_words, ref_words):
    print(f"{u['text']:>10} -> {r['text'] if r else 'None':>10}")
