class Tokenizer:
    def __init__(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = [line.strip() for line in f.readlines()]
        
        # Mapping token <-> id
        self.token2id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2token = {idx: token for idx, token in enumerate(self.vocab)}
        
        self.UNK_ID = self.token2id.get('[UNK]', 0)

    def tokenize(self, text):
        tokens = []
        for word in text.lower().split():  # split spasi
            i = 0
            while i < len(word):
                for j in range(len(word), i, -1):
                    piece = word[i:j]
                    if i > 0:
                        piece = "##" + piece
                    if piece in self.token2id:
                        tokens.append(piece)
                        i = j
                        break
                else:
                    tokens.append('[UNK]')
                    i += 1
        token_ids = [self.token2id.get(t, self.UNK_ID) for t in tokens]
        return token_ids

    def detokenize(self, token_ids):
        words = []
        word = ""
        for tid in token_ids:
            token = self.id2token.get(tid, '[UNK]')
            if token.startswith("##"):
                word += token[2:]
            else:
                if word:
                    words.append(word)
                word = token
        if word:
            words.append(word)
        return " ".join(words)
    
    def vocab_size(self):
        return len(self.vocab)