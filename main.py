import torch
from torch.nn import functional as F
from contextlib import nullcontext
from model import Transformer
import tiktoken
from benchmark import Benchmark
from config import Configs, AttnType, PosType, NormType, ModelType, args, args_configs
from utils import set_seed, ModelContext

class Main:
    def __init__(self, configs:Configs, use_cache=False, stream=False):
        set_seed()

        self.model_context = ModelContext(configs=configs)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[configs.train_type.value.dtype]
        self.ctx = nullcontext() if self.device == 'cpu' else torch.amp.autocast(device_type=self.device, dtype=ptdtype)
        self.use_cache = use_cache
        self.stream = stream
        self.benchmark = Benchmark(device=self.device)

    @torch.no_grad()
    def generate(self, idx, enc:tiktoken.Encoding, max_new_tokens, temperature=1.0, top_k=None, stop_token=False):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        eot_token = enc.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]

        if self.use_cache:
            self.model_context.model.reset_cache()

        self.benchmark.start()

        for index in range(max_new_tokens):
            if self.use_cache and index > 0:
                idx_cond = idx[:, [-1]]
            else:
                idx_cond = idx

            logits, _ = self.model_context.model(idx_cond, use_cache=self.use_cache)
            
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

            if self.model_context.configs.model_type.value.block_size < idx.size(-1):
                print("Block full!")
                break

        self.benchmark.stop()

        return idx
    
    def warmup(self):
        print("Warming up GPU...")

        dummy_input = torch.randint(0, 50257, (1, 20), device=self.device)

        self.model_context.model.eval()

        with torch.no_grad():
            with self.ctx:
                for _ in range(5):
                    _ = self.model_context.model(dummy_input, use_cache=self.use_cache)

        if self.device == 'cuda':
            torch.cuda.synchronize()

        print("Warm-up done.\n")

    def run(self, max_new_tokens, start, temperature=0.8, top_k=200, pretrained=False):
        self.model_context.from_pretrained() if pretrained else self.model_context.get_model(resume=True, device=self.device, filename='best_checkpoint.pt')
        
        self.model_context.model.eval()
        self.model_context.model.to(self.device)

        self.model_context.configs.in_training = False

        label = "use_cache=True" if self.use_cache else "use_cache=False"
        print(f'\n[{label}]')
        print(self.model_context.configs.info())

        #self.warmup(model)

        enc = tiktoken.get_encoding("gpt2")
        start_ids = enc.encode(start, allowed_special={"<|endoftext|>"})
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])

        with torch.no_grad():
            with self.ctx:
                y = self.generate(x, enc, max_new_tokens, temperature=temperature, top_k=top_k)
                
        text = enc.decode(y[0].tolist())

        print('Total Token:', len(y[0]))
        print('-'*100)

        return text, y

main = Main(configs=args_configs, use_cache=args.use_cache)
text, y = main.run(args.max_new_tokens, args.start, pretrained=args.pretrained)

if args.print_out:
    print(text)

if args.compare:
    main_1 = Main(configs=args_configs, use_cache=not args.use_cache)
    text_1, y_1 = main_1.run(args.max_new_tokens, args.start, pretrained=args.pretrained)
    
    if torch.equal(y, y_1):
        print("== OK ==")
    else:
        print("== NO ==")