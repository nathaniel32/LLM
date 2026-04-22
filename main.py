import torch
from torch.nn import functional as F
from contextlib import nullcontext
from model import GPT, ModelConfig
import tiktoken
from benchmark import Benchmark
from env import AttnType, PosType

class Main:
    def __init__(self, use_cache=False, stream=False):
        seed = 1337
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = 'float32'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        self.ctx = nullcontext() if self.device == 'cpu' else torch.amp.autocast(device_type=self.device, dtype=ptdtype)
        self.use_cache = use_cache
        self.stream = stream
        self.benchmark = Benchmark(device=self.device)

    @staticmethod
    def from_pretrained(model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        config_args['dropout'] = 0.0

        print(config_args)
        
        # create a from-scratch initialized minGPT model
        config = ModelConfig(**config_args)
        model = GPT(config, attn_type="mha", pos_type="pos_emb")
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def from_out(self, model_type, attn_type:AttnType, pos_type:PosType):
        import os
        
        out_dir = os.path.join("out", model_type, attn_type.value, pos_type.value)
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        gptconf = ModelConfig(**checkpoint['model_args'])
        attn_type = checkpoint['attn_type']
        pos_type = checkpoint['pos_type']
        model = GPT(gptconf, attn_type, pos_type)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        print({'attn_type': attn_type, 'pos_type': pos_type})
        return model
    
    @torch.no_grad()
    def generate(self, idx, model:GPT, enc:tiktoken.Encoding, max_new_tokens, temperature=1.0, top_k=None, stop_token=False):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        eot_token = enc.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]

        if self.use_cache:
            model.reset_cache()

        self.benchmark.start()

        for index in range(max_new_tokens):
            if self.use_cache and index > 0:
                idx_cond = idx[:, [-1]]
            else:
                #idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
                idx_cond = idx

            logits, _ = model(idx_cond, use_cache=self.use_cache)
            
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            #idx_next = torch.argmax(probs, dim=-1, keepdim=True)

            self.benchmark.step()
            
            if idx_next.item() == eot_token and stop_token:
                break
            
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

            if self.stream:
                text = enc.decode(idx[0].tolist())
                print('\033[u\033[J' + text, end='', flush=True)

            if model.config.block_size < idx.size(-1):
                print("Block full!")
                break

        self.benchmark.stop()

        return idx
    
    def warmup(self, model):
        print("Warming up GPU...")

        dummy_input = torch.randint(0, 50257, (1, 20), device=self.device)

        model.eval()

        with torch.no_grad():
            with self.ctx:
                for _ in range(5):
                    _ = model(dummy_input, use_cache=self.use_cache)

        if self.device == 'cuda':
            torch.cuda.synchronize()

        print("Warm-up done.\n")

    def run(self, max_new_tokens, model_type, attn_type:AttnType, pos_type:PosType, start, temperature=0.8, top_k=200, pretrained=None):
        model = self.from_out(model_type, attn_type, pos_type) if pretrained is None else self.from_pretrained(pretrained)
        model.eval()
        model.to(self.device)

        #self.warmup(model)

        enc = tiktoken.get_encoding("gpt2")
        start_ids = enc.encode(start, allowed_special={"<|endoftext|>"})
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])

        with torch.no_grad():
            with self.ctx:
                y = self.generate(x, model, enc, max_new_tokens, temperature=temperature, top_k=top_k)
                
        text = enc.decode(y[0].tolist())

        label = "use_cache=True" if self.use_cache else "use_cache=False"
        print(f'\n[{label}] - [{attn_type}] - [{model_type}]')
        print('Total Token:', len(y[0]))
        print('-'*100)

        return text, y

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--no-cache", action="store_false", dest="use_cache")
parser.add_argument("--print-out", action="store_true", dest="print_out")
parser.add_argument("--max_new_tokens", type=int, default=100000)
parser.add_argument("--model_type", type=str, default="small")
parser.add_argument("--attn_type", type=str, default="mha")
parser.add_argument("--pos_type", type=str, default="wpe")
parser.add_argument("--start", type=str, default="the colors of the German flag are")
parser.add_argument("--compare", action="store_true")
parser.add_argument("--pretrained", type=str)
args = parser.parse_args()
print(vars(args))

main = Main(use_cache=args.use_cache)
text, y = main.run(args.max_new_tokens, args.model_type, AttnType[args.attn_type], PosType[args.pos_type], args.start, pretrained=args.pretrained)

if args.print_out:
    print(text)

if args.compare:
    main_1 = Main(use_cache=not args.use_cache)
    text_1, y_1 = main_1.run(args.max_new_tokens, args.model_type, AttnType[args.attn_type], PosType[args.pos_type], args.start, pretrained=args.pretrained)
    
    if torch.equal(y, y_1):
        print("== OK ==")
    else:
        print("== NO ==")