import torch
import utils
import config
import model as nn_model
from tokenizer import Tokenizer

def main():
    torch.manual_seed(1337)
    tokenizer = Tokenizer('data/vocab_en.txt')
    train_data, val_data  = utils.prepare_data()

    train_data = torch.tensor(tokenizer.tokenize(train_data), dtype=torch.long)
    val_data = torch.tensor(tokenizer.tokenize(val_data), dtype=torch.long)

    print(val_data)
    model = nn_model.GPTLanguageModel(tokenizer.vocab_size()).to(config.device)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # train
    for iter in range(config.max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            train_losses = utils.estimate_loss(model, train_data)
            val_losses = utils.estimate_loss(model, val_data)
            print(f"step {iter}: train loss {train_losses:.4f}, val loss {val_losses:.4f}")

        # sample a batch of data
        xb, yb = utils.get_batch(train_data)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    #print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
    open('more.txt', 'w').write(tokenizer.detokenize(model.generate(context, max_new_tokens=10000)[0].tolist()))

main()