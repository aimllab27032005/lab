def minimax(total, turn, alpha, beta):
    if total == 20: return 0
    if total > 20: return -1 if turn else 1

    if turn:
        best = -float('inf')
        for i in range(1, 4):
            best = max(best, minimax(total + i, False, alpha, beta))
            alpha = max(alpha, best)
            if beta <= alpha: break
    else:
        best = float('inf')
        for i in range(1, 4):
            best = min(best, minimax(total + i, True, alpha, beta))
            beta = min(beta, best)
            if beta <= alpha: break
    return best

total = 0

while True:
    human_move = int(input("Enter your move (1, 2, or 3): "))
    if human_move not in [1, 2, 3]: continue
    total += human_move
    print(f"Total: {total}")
    if total >= 20:
      print("You win!")
      break

    print("AI thinking...")
    ai_move = max(range(1, 4), key=lambda i: minimax(total + i, False, -float('inf'), float('inf')))
    total += ai_move
    print(f"AI adds {ai_move}. Total: {total}")
    if total >= 20:
      print("AI wins!")
      break
