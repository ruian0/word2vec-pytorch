import os
import numpy as np
import json
import torch
from time import time

class Trainer:
    """Main class for model training"""
    
    def __init__(
        self,
        model,
        epochs,
        train_dataloader,
        train_steps,
        val_dataloader,
        val_steps,
        checkpoint_frequency,
        criterion,
        optimizer,
        lr_scheduler,
        device,
        model_dir,
        model_name,
        writer
    ):  
        self.model = model
        self.epochs = epochs
        self.step_cnt = 0
        self.train_dataloader = train_dataloader
        self.train_steps = train_steps
        self.val_dataloader = val_dataloader
        self.val_steps = val_steps
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_frequency = checkpoint_frequency
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_dir = model_dir
        self.model_name = model_name
        self.writer = writer

        self.loss = {"train": [], "val": []}
        self.model.to(self.device)

    def train(self):
        for epoch in range(self.epochs):
            t0 = time()
            self._train_epoch(epoch)
            self._validate_epoch(epoch)
            t1 = time()
            print(
                "Epoch: {}/{}, {:.2f} sec, Train Loss={:.5f}, Val Loss={:.5f}".format(
                    epoch + 1,
                    self.epochs,
                    t1 - t0,
                    self.loss["train"][-1],
                    self.loss["val"][-1],
                )
            )

            self.lr_scheduler.step()

            if self.checkpoint_frequency:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch_itr):
        self.model.train()
        self.model.embeddings.requires_grad_(False)
        
        running_loss = []

        for i, batch_data in enumerate(self.train_dataloader, 1):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            # if epoch_itr > 2 and loss.item() > 10:
                
            #     import pandas as pd
                
            #     print("......", loss.item())
            #     print("inputs are....")
            #     print(inputs.shape)
            #     print("labels are ....")
            #     print(labels.shape)
                
            #     x_df = pd.DataFrame(input.clone())
            #     x_df.to_csv('tmp.csv')
                
            self.optimizer.step()

            running_loss.append(loss.item())
            self.writer.add_scalar("Loss/step(train)", loss, i + epoch_itr * self.step_cnt)
            
            if i == self.train_steps:
                break
            if i > self.step_cnt:
                self.step_cnt = i
        self.writer.add_scalar("steps", self.step_cnt)

        epoch_loss = np.mean(running_loss)
        self.writer.add_scalar("Loss/epoch(train)", epoch_loss, epoch_itr)

        self.loss["train"].append(epoch_loss)

    def _validate_epoch(self, epoch_itr):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, batch_data in enumerate(self.val_dataloader, 1):
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                self.writer.add_scalar("Loss/step(eval)", loss, i + epoch_itr * self.step_cnt)

                running_loss.append(loss.item())

                if i == self.val_steps:
                    break

        epoch_loss = np.mean(running_loss)
        self.writer.add_scalar("Loss/epoch(eval)", epoch_loss, epoch_itr)
        self.loss["val"].append(epoch_loss)

    def _save_checkpoint(self, epoch):
        """Save model checkpoint to `self.model_dir` directory"""
        epoch_num = epoch + 1
        if epoch_num % self.checkpoint_frequency == 0:
            model_path = "checkpoint_{}.pt".format(str(epoch_num).zfill(3))
            model_path = os.path.join(self.model_dir, model_path)
            torch.save(self.model, model_path)

    def save_model(self):
        """Save final model to `self.model_dir` directory"""
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model, model_path)

    def save_model_dict(self):
        model_path = os.path.join(self.model_dir, "model.pth")
        torch.save(self.model.state_dict(), model_path)

    def save_loss(self):
        """Save train/val loss as json file to `self.model_dir` directory"""
        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)