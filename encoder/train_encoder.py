from collections import OrderedDict
from pathlib import Path
import torch
import sys
import csv
import os
import pandas as pd
# import wandb
from encoder_dataset import UBCDataset
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from config import config
from torch.utils.data import DataLoader
from dotenv import dotenv_values

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.
    
    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include either ".pth" or ".pt" as the file extension.
        
    Example usage:
        save_model(model=model_0,
        targer_dir="models", 
        model_name="model_1")
    """

    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"),  "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to : {model_save_path}")
    torch.save(obj=model,
               f=model_save_path)

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(tqdm(dataloader)):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)
        if isinstance(y_pred, OrderedDict):
                y_pred = y_pred["out"]

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        # y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        # train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    
    # train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    pred_labels = []
    true_labels = []
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(tqdm(dataloader)):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)
            if isinstance(test_pred_logits, OrderedDict):
                test_pred_logits = test_pred_logits["out"]

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            # test_pred_labels = test_pred_logits.argmax(dim=1)
            # test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            # pred_labels = pred_labels + test_pred_labels.cpu().tolist()
            # true_labels = true_labels + y.cpu().tolist()

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    # test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]} 
    For example if training for epochs=2: 
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               # "train_acc": [],
                "test_loss": [],
                # "test_acc": []
                }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        # Print out what's happening
        print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
         #       f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
         #       f"test_acc: {test_acc:.4f} | "
        )
        if config['WANDB']:
            wandb.log({"train_loss": train_loss,
                        "test_loss": test_loss})
    
        # Update results dictionary
        results["train_loss"].append(train_loss)
        # results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        # results["test_acc"].append(test_acc)

        if (int(epoch) % 10) == 0:
            checkpoint = {'state_dict': model.state_dict(),'optimizer' :optimizer.state_dict()}
            torch.save(checkpoint, os.path.join("/home/erila018/git/Ovarian-Cancer-Competition/encoder", f"{config['MODEL_NAME']}_checkpoint.pth"))
            # save_model(model.state_dict(), "../models/saved", f"{config['MODEL_NAME']}_checkpoint.pth")
    # Return the filled results at the end of the epochs
    return results

from datetime import datetime

def init_experiment(wandb,
                    project_name: str="UBC_encoder",
                    experiment_name: str="",
                    extra: str="",
                    config: dict={}):
    """Initializes an experiment with correct naming conventions.

    You need to be logged in before running this function (see wandb.login()).

    After the initialization, track experiment metrics by using the wandb.log() function.

    Args:
        wandb: The imported wandb object from `import wandb`. 
        project_name: The project name (default=TDDE19).
        experiment_name: The experiment name (required).
        extra: Extra information about the experiment.
        config: A dictionary including all hyperparameters and metadata in the experiment. 
    
    Raises:
        NoExperimentNameError: The experiment_name is required.
    """
    assert experiment_name != "", ('The experiment_name is required.')
    
    timestamp = datetime.now().strftime("%m-%d") # returns current date in MM-DD format
    if extra:
        name = f"{timestamp}/{experiment_name}/{extra}"
    else: 
        name = f"{timestamp}/{experiment_name}"
    wandb.init(
        # Set the project where this run will be logged
        project=project_name, 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=name, 
        # Track hyperparameters and run metadata
        config=config)



if __name__ == "__main__":
    if config['WANDB']:
        env_config = dotenv_values("../.env")
        wandb.login(key=env_config["WANDB_API_KEY"])
        init_experiment(wandb, project_name="UBC_encoder", experiment_name=config['MODEL_NAME'], config=config)
    traindf = pd.read_csv(os.path.join(config['INPUT_DIR'], 'train_tiles_updated.csv'))
    
    train_data, val_data = train_test_split(traindf, test_size=config['SPLIT_RATIO'], random_state=config['SEED'])
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)

    train_dataset = UBCDataset(train_data, transforms=config['DATA_TRANSFORMS']["train"])
    train_dataloader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, pin_memory=True)

    val_dataset = UBCDataset(val_data, transforms=config['DATA_TRANSFORMS']["valid"]) 
    val_dataloader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], 
                                num_workers=config['NUM_WORKERS'], shuffle=False, pin_memory=True)
    
    model = config['MODEL']
  
    print(f"Running on device: {next(model.parameters()).device}\n")

    optimizer = torch.optim.Adam(params=model.parameters())
    loss_fn = torch.nn.MSELoss() # Trying MSE, seems more applicable to logits

    if config['WANDB']:
        wandb.watch(model, loss_fn, log="all") # automatically collect the model’s gradients and the model’s topology. TODO: Test if this slows down training

    results = train(model=model,
                   train_dataloader=train_dataloader,
                   test_dataloader=val_dataloader,
                   optimizer=optimizer,
                   loss_fn=loss_fn,
                   epochs=config['EPOCHS'],
                   device=config['DEVICE'])
    
    # Create or open the CSV file in write mode
    with open('/home/erila018/git/Ovarian-Cancer-Competition/encoder/results.csv', mode='w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write the header row
        csv_writer.writerow(["Epoch", "Train Loss", "Test Loss"])

        # Write the data rows
        for i in range(len(results["train_loss"])):
            csv_writer.writerow([i + 1, results["train_loss"][i], results["test_loss"][i]])

        print(f"Loss results saved")

    save_model(model, "../models/saved", f"{config['MODEL_NAME']}.pth")