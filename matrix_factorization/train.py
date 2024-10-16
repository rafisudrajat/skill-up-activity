from tempfile import TemporaryDirectory
import torch
import time
import torch.nn as nn
from model import MF
import os
from torch.utils.data import DataLoader
import logging

def train_model(net:MF,dataloaders:dict[str,DataLoader],dataset_sizes:dict[str,int],device:torch.device,optimizer:torch.optim.Optimizer,scheduler,num_epochs:int=25)->MF:
    """
    Train the Matrix Factorization (MF) model.

    Parameters:
    -----------
    net : MF
        The Matrix Factorization model to be trained.
    dataloaders : dict[str, DataLoader]
        A dictionary containing the DataLoader objects for training and validation datasets.
        Expected keys are 'train' and 'val'.
    dataset_sizes : dict[str, int]
        A dictionary containing the sizes of the training and validation datasets.
        Expected keys are 'train' and 'val'.
    device : torch.device
        The device to run the training on (e.g., 'cpu' or 'cuda').
    optimizer : Optimizer
        The optimizer to use for training.
    scheduler : _LRScheduler
        The learning rate scheduler to use for training.
    num_epochs : int, optional
        The number of epochs to train the model. Default is 25.

    Returns:
    --------
    MF
        The trained Matrix Factorization model with the best validation loss.
    """
    # Setup logger
    logging.basicConfig(filename='training.log', level=logging.INFO)
    logger = logging.getLogger()

    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        criterion = nn.MSELoss(reduction='sum')

        torch.save(net.state_dict(),best_model_params_path)
        best_loss = float('inf')
        for epoch in range(num_epochs):
            logger.info(f'Epoch {epoch}/{num_epochs - 1}')
            logger.info('-' * 10)
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    net.train()  # Set model to training mode
                else:
                    net.eval()   # Set model to evaluate mode
                
                running_loss = 0.0

                # Iterate over data.
                for user_batch, item_batch, rating_batch in dataloaders[phase]:
                    user_batch = user_batch.to(device)
                    item_batch = item_batch.to(device)
                    rating_batch = rating_batch.to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase=='train'):
                        outputs = net(user_batch,item_batch)
                        # apply MSE as loss function
                        mse_loss = criterion(outputs, rating_batch)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            mse_loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += mse_loss.item()
                
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                logger.info(f'{phase} Loss: {epoch_loss:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(net.state_dict(), best_model_params_path)
            logger.info('\n')
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best loss: {best_loss:4f}')
        logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        logger.info(f'Best loss: {best_loss:4f}')

        # load best model weights
        net.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    
    return net
