import torch
import utils
import config
import model as nn_model

def main():
    torch.manual_seed(1337)
    
    train_data, val_data, vocab_size, decode,  = utils.prepare_data()

    model = nn_model.GPTLanguageModel(vocab_size).to(config.device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

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
    open('more.txt', 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))

main()