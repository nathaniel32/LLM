#config = GPTConfig(block_size=256, vocab_size=50257, n_layer=6, n_head=12, n_embd=384, dropout=0.0, bias=True)
#config = GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=True)
#config = GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1536, dropout=0.0, bias=True)

model_configs = {
    'small': dict(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768,  dropout=0.0, bias=True),
    'medium': dict(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024, dropout=0.0, bias=True),
    'large': dict(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280, dropout=0.0, bias=True),
    'xl': dict(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600, dropout=0.0, bias=True),
}

dataset_configs = {
    'shakespeare': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
    'simple_text': 'https://raw.githubusercontent.com/uwgraphics/VEP2_TCP_SimpleText/refs/heads/main/N3/N37535.txt',
    'wiki_indo': 'https://id.wikipedia.org/w/index.php?title=Indonesia&action=raw',
    'wiki_indo_json': 'https://id.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&titles=Indonesia&explaintext=1'
}