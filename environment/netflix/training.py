import logging
import os
import time

import mxnet as mx
import numpy as np

from environment.netflix.initializer import EmbeddingInitializer


def find_feature_config(name, features_config):
    for feature_config in features_config:
        if feature_config["name"] == name:
            return feature_config


def initialize(model, config, context):
    features_config = config["features"]
    for name, block in model._children.items():
        feature_config = find_feature_config(name, features_config)
        if feature_config is not None and feature_config["encoding"] == "embedding":
            logging.info("Initialize feature '{}' uniformly".format(name))
            initializer = EmbeddingInitializer(scale=0.1)
        else:
            logging.info("Initialize block '{}' with Xavier".format(name))
            initializer = mx.init.Xavier()

        block.collect_params().initialize(initializer, ctx=context)

    model.hybridize()


def pick_loss(config):
    if config["training"]["loss"] == "mse":
        return mx.gluon.loss.L2Loss()
    else:
        raise ValueError()


def train_with_optimizer(model, config, optimizer, train_loader, val_loader, context, batches_in_step=100, batches_in_epoch=1000, writer=None, checkpoint_dir=None, loss_epsilon=1e-6, suffix=""):
    criterion = pick_loss(config)
    print("Initial loss: {:.5f}".format(validate(model, config, val_loader, context)))

    start = time.time()

    history = []
    num_epochs = config["training"]["num_epochs"]
    loss_smoothing_epochs = config["training"]["loss_smoothing_epochs"]
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            x = batch[0].as_in_context(context)
            y = batch[1].as_in_context(context)
            batch_size = len(y)

            with mx.autograd.record():
                outputs = model(x)
                loss = criterion(outputs, y)
            loss.backward()
            optimizer.step(batch_size)

            if (i + 1) % batches_in_step == 0:
                print("#", end="")

            if (i + 1) % batches_in_epoch == 0:
                stats = {
                    "epoch": epoch + 1,
                    "num_epochs": num_epochs,
                    "batch": i + 1,
                    "num_batches": int(len(train_loader.dataset) / batch_size) + 1,
                    "elapsed": int(time.time() - start),
                    "val_loss": validate(model, config, val_loader, context),
                    "learning_rate": optimizer.learning_rate
                }

                print(' Epoch: [{epoch}/{num_epochs}], Step: [{batch}/{num_batches}], Elapsed: {elapsed:d}s, Loss: {val_loss:.5f}, LR: {learning_rate:.6f}'.format(**stats))

                if len(history) >= loss_smoothing_epochs and np.mean([s["val_loss"] for s in history[-loss_smoothing_epochs:]]) < stats["val_loss"] + loss_epsilon:
                    # Restore model from checkpoint
                    if checkpoint_dir is not None:
                        model.load_parameters(os.path.join(checkpoint_dir, "model.params"), ctx=context)
                        logging.info("Restored model from checkpoint")

                    # Decrease learning rate
                    optimizer.set_learning_rate(optimizer.learning_rate * config["training"]["learning_rate_decay"])
                    logging.info("Decreased learning rate to {}".format(optimizer.learning_rate))

                    if optimizer.learning_rate < config["training"]["min_learning_rate"]:
                        history.append(stats)
                        logging.info("Early stopping because minimum learning rate achieved")
                        return history
                else:
                    if writer is not None:
                        writer.add_scalar("ValidationLoss_" + suffix, stats["val_loss"], global_step=epoch * stats["num_batches"] + i)
                    if checkpoint_dir is not None:
                        model.save_parameters(os.path.join(checkpoint_dir, "model.params"))

                history.append(stats)
        print()

    return history


def validate(model, config, loader, context):
    criterion = pick_loss(config)

    n = 0
    loss = 0
    for i, batch in enumerate(loader):
        x = batch[0].as_in_context(context)
        y = batch[1].as_in_context(context)

        loss += mx.nd.sum(criterion(model(x), y)).asscalar()
        n += len(y)

    return loss / n
