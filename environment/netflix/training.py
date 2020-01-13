import logging
import os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init


def initialize(model):
    def weight_init(m):
        if isinstance(m, nn.Linear):
            print("Initialize linear with Xavier".format(m))
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.Embedding):
            print("Initialize embeddings with Uniform".format(m))
            init.uniform_(m.weight.data, a=-0.1, b=-0.1)
        else:
            print("Don't know how to initialize {}".format(type(m).__name__))
    model.apply(weight_init)


def pick_loss(config):
    if config["training"]["loss"] == "mse":
        return nn.MSELoss()
    else:
        raise ValueError()


def train(model, config, train_loader, val_loader, batches_in_step=100, batches_in_epoch=1000, writer=None, checkpoint_dir=None, loss_epsilon=1e-6, suffix=""):
    criterion = pick_loss(config)
    print("Initial loss: {:.5f}".format(validate(model, config, val_loader)))

    optimizer = model.create_optimizer()

    start = time.time()

    history = []
    num_epochs = config["training"]["num_epochs"]
    loss_smoothing_epochs = config["training"]["loss_smoothing_epochs"]
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            if (i + 1) % batches_in_step == 0:
                print("#", end="")

            if (i + 1) % batches_in_epoch == 0:
                batch_size = len(y)
                stats = {
                    "epoch": epoch + 1,
                    "num_epochs": num_epochs,
                    "batch": i + 1,
                    "num_batches": int(len(train_loader.dataset) / batch_size) + 1,
                    "elapsed": int(time.time() - start),
                    "val_loss": validate(model, config, val_loader),
                    "learning_rate": [pg["lr"] for pg in optimizer.param_groups]
                }

                print(' Epoch: [{epoch}/{num_epochs}], Step: [{batch}/{num_batches}], Elapsed: {elapsed:d}s, Loss: {val_loss:.5f}, LR: {learning_rate}'.format(**stats))

                if len(history) >= loss_smoothing_epochs and np.mean([s["val_loss"] for s in history[-loss_smoothing_epochs:]]) < stats["val_loss"] + loss_epsilon:
                    # Restore model from checkpoint
                    if checkpoint_dir is not None:
                        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.params")))
                        logging.info("Restored model from checkpoint")

                    # Decrease learning rate
                    for group in optimizer.param_groups:
                        group["lr"] *= config["training"]["learning_rate_decay"]
                    logging.info("Decreased learning rates to {}".format([pg["lr"] for pg in optimizer.param_groups]))

                    if any(map(lambda pg: pg["lr"] < config["training"]["min_learning_rate"], optimizer.param_groups)):
                        history.append(stats)
                        logging.info("Early stopping because minimum learning rate achieved")
                        return history
                else:
                    if writer is not None:
                        writer.add_scalar("ValidationLoss_" + suffix, stats["val_loss"], global_step=epoch * stats["num_batches"] + i)
                    if checkpoint_dir is not None:
                        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.params"))

                history.append(stats)
        print()

    return history


def validate(model, config, loader):
    criterion = pick_loss(config)

    n = 0
    loss = 0
    for i, (x, y) in enumerate(loader):
        loss += criterion(model(x), y).sum().tolist()
        n += len(y)

    return loss / n
