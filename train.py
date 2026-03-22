import csv
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tqdm import tqdm
from modules import build_vocab, get_batch, GPT


def loss_fn(model: nn.Module, X: mx.array, y: mx.array) -> mx.array:
    logits = model(X)
    loss = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        y.reshape(-1),
    )
    return loss.mean()


def main():
    mx.random.seed(42)

    # Data preparation
    with open("dataset.txt", "r", encoding="utf-8") as f:
        data = f.read()

    vocab_size, encode, _ = build_vocab(data)

    data_len = len(data)
    train_data = data[: int(data_len * 0.9)]
    val_data = data[int(data_len * 0.9) :]

    train_ids = mx.array(encode(train_data))
    val_ids = mx.array(encode(val_data))

    # Train setup
    seq_len = 256
    model = GPT(vocab_size, seq_len, d_model=384, num_heads=6, n_layers=6)
    mx.eval(model.parameters())

    total_steps = 10000
    warmup_steps = int(total_steps * 0.1)

    max_lr = 3e-4
    initial_lr = max_lr / 25
    final_lr = max_lr / 10

    lr_schedule = optim.join_schedules(
        [
            optim.linear_schedule(initial_lr, max_lr, warmup_steps),
            optim.cosine_decay(max_lr, total_steps - warmup_steps, end=final_lr),
        ],
        [warmup_steps],
    )

    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.1)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training
    batch_size = 32
    val_every = 200
    val_iters = 50

    total_train_loss = 0.0
    best_val_loss = float("inf")

    with open("history.csv", "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["step", "train_loss", "val_loss", "ppl"])

        pbar = tqdm(range(total_steps), desc="Training")

        for step in pbar:
            X, y = get_batch(train_ids, batch_size, seq_len)

            loss, grads = loss_and_grad(model, X, y)
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(model, grads)

            mx.eval(model.parameters(), optimizer.state)
            total_train_loss += loss.item()

            if step > 0 and step % val_every == 0:
                model.eval()
                total_val_loss = 0.0

                for _ in range(val_iters):
                    X_val, y_val = get_batch(val_ids, batch_size, seq_len)
                    val_loss = loss_fn(model, X_val, y_val)
                    mx.eval(val_loss)
                    total_val_loss += val_loss.item()

                avg_val_loss = total_val_loss / val_iters
                model.train()

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    model.save_weights("best_model.npz")

                avg_train_loss = total_train_loss / val_every
                total_train_loss = 0.0

                ppl = math.exp(min(avg_val_loss, 20))
                current_lr = float(lr_schedule(step))

                csv_writer.writerow(
                    [
                        step,
                        f"{avg_train_loss:.4f}",
                        f"{avg_val_loss:.4f}",
                        f"{ppl:.2f}",
                    ]
                )
                csv_file.flush()

                pbar.set_postfix(
                    {
                        "train": f"{avg_train_loss:.4f}",
                        "val": f"{avg_val_loss:.4f}",
                        "ppl": f"{ppl:.2f}",
                        "lr": f"{current_lr:.2e}",
                    }
                )

    print(f"\nBest val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
