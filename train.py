import argparse
import yaml
import os
import time
import torch
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from utils.dataloader import get_dataloader_and_vocab
from utils.trainer import Trainer
from utils.helper import (
    get_model_class,
    get_optimizer_class,
    get_lr_scheduler,
    save_config,
    save_vocab,
)
from torch.utils.tensorboard import SummaryWriter


def train(config):

    timestr = time.strftime("%Y%m%d-%H%M%S")

    base = config.get("base", "")
    epochs = config.get("epochs", 5)
    lr = config["learning_rate"]
    # path = config['model_dir']
    
    if base == "test":
        path = f"{config['model_dir']}-TEST-epochs-{epochs}-{lr}-{timestr}"
    elif base == "prod":
        path = f"{config['model_dir']}-PROD-epochs-{epochs}-{lr}-{timestr}"
                    
    writer = SummaryWriter(path)

    if not os.path.exists(path):
        os.makedirs(path)
    
    train_dataloader, vocab = get_dataloader_and_vocab(
        model_name=config["model_name"],
        ds_name=config["dataset"],
        ds_type="train",
        data_dir=config["data_dir"],
        batch_size=config["train_batch_size"],
        shuffle=config["shuffle"],
        vocab=None,
        tokenizer_type=config['tokenizer_type'], 
        base=base
    )
    
    val_dataloader, _ = get_dataloader_and_vocab(
        model_name=config["model_name"],
        ds_name=config["dataset"],
        ds_type="valid",
        data_dir=config["data_dir"],
        batch_size=config["val_batch_size"],
        shuffle=config["shuffle"],
        vocab=vocab,
        tokenizer_type=config['tokenizer_type'],
        base=base
    )
    
    cnt = 0
    for x, y in val_dataloader:
        cnt += 1
        
    print("train_loader has ", cnt)
    
    if isinstance(vocab, dict): 
        vocab_size = len(vocab)
    else:
        vocab_size = len(vocab.get_stoi())
    print(f"Vocabulary size: {vocab_size}")

    model_class = get_model_class(config["model_name"])
    if config['tokenizer_type'] == 'english':
        model = model_class(vocab_size=vocab_size)
    else:
        model = model_class(vocab=vocab)
        
    print("model is, ", type(model), model)
    
    criterion = nn.CrossEntropyLoss()

    optimizer_class = get_optimizer_class(config["optimizer"])
    optimizer = optimizer_class(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = get_lr_scheduler(optimizer, config["epochs"], verbose=True)

    device = config['device']
    if device == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    trainer = Trainer(
        model=model,
        epochs=config["epochs"],
        train_dataloader=train_dataloader,
        train_steps=config["train_steps"],
        val_dataloader=val_dataloader,
        val_steps=config["val_steps"],
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_frequency=config["checkpoint_frequency"],
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=path,
        model_name=config["model_name"],
        writer=writer
    )

    trainer.train()
    print("Training finished.")

    trainer.save_model()
    trainer.save_model_dict()
    trainer.save_loss()
    save_vocab(vocab, path)
    save_config(config, path)
    print("Model artifacts saved to folder:", path)
    writer.close()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    args = parser.parse_args()
        
    
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    train(config)