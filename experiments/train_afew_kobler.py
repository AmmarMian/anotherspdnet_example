# Training on AFEW dataset

import argparse

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from torch.utils.data import random_split, DataLoader
from geoopt.optim import RiemannianAdam

import tqdm
import matplotlib.pyplot as plt

# Activate LaTex rendering for matplotlib
import matplotlib
# matplotlib.rcParams['text.usetex'] = True

# Styling matplotlib
plt.style.use('dark_background')
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.alpha'] = 0.3
matplotlib.rcParams['grid.linestyle'] = 'dotted'
matplotlib.rcParams['axes.facecolor'] = (0.1, 0.1, 0.1, 0.9)
matplotlib.rcParams['font.family'] = 'serif'
# matplotlib.rcParams['font.serif'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 11

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from src.data import AFEWSPDnetDataset
from src.utils import MatplotlibTrainingVisualizer, setup_logging, format_params
# from src.models import SPDNet
from spdnet_kobler.models import SPDNetKobler

logger = setup_logging()

# Tensorboard
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Training on AFEW dataset with spdnet")
    parser.add_argument("--hd", type=str, default="[200, 100, 50]",
                        help="Hidden dimensions of spdnet")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Learning rate")
    parser.add_argument("--eps", type=float, default=1e-2,
                        help="Epsilon for SPDNet")
    parser.add_argument("--softmax", type=bool, default=False,
                        help="Use softmax activation function")
    parser.add_argument("--batchnorm", type=str, default="False",
                        help="Use batch normalization")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")
    parser.add_argument("--dtype", type=str, default="float32",
                        help="Data type (float32 or float64)")
    parser.add_argument("--train_percentage", type=float, default=0.7,
                        help="Percentage of training data")
    parser.add_argument("--seed", type=int, default=5555,
                        help="Random seed")
    parser.add_argument("--shuffle_loader", action="store_true", default=False,
                        help="Whether to shuffle data loaders at each epoch.")
    parser.add_argument("--dataset_path", type=str,
                        default="data/AFEW_spdnet/spdface_400_inter_histeq/",
                        help="Path to AFEW dataset")
    parser.add_argument("--storage_path", type=str, default="results/test",
                        help="Path to save results")

    args = parser.parse_args()

    logger.info("Training on AFEW dataset with spdnet")
    args.hd = eval(args.hd)
    args.device = torch.device(args.device)
    args.batchnorm = eval(args.batchnorm)
    if args.dtype == "float32":
        args.dtype = torch.float32
    elif args.dtype == "float64":
            args.dtype = torch.float64
    logger.info(format_params(vars(args)))


    # Load data
    dataset = AFEWSPDnetDataset(
                directory_path = args.dataset_path,
                preload = True,
                shuffle = False,
                subset = "train",
                rng = torch.Generator().manual_seed(args.seed),
                device = args.device,
                verbose = 1)
    train_size = int(args.train_percentage * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                generator=torch.Generator().manual_seed(args.seed))


    # Create models
    device = torch.device(args.device)
    n_out = dataset.n_classes
    spdnet = SPDNetKobler(input_dim = dataset.d, hidden_layers_size = args.hd,
                    output_dim = dataset.n_classes, eps = args.eps,
                    softmax = args.softmax,
                    precision = args.dtype, batchnorm = args.batchnorm).to(device)
    model_name = str(spdnet)
                    

    # Create optimizers
    optimizer = RiemannianAdam(spdnet.parameters(), lr=args.lr)

    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                  patience=5, verbose=True)


    # Create loss function
    loss_fn = nn.CrossEntropyLoss()

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                            shuffle=args.shuffle_loader)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                            shuffle=args.shuffle_loader)

    # ========================================================================
    # Tensorboard
    # ========================================================================
    logger.info('Creating Tensorboard writer...')
    
    # Create the summary writer in the main directory so we have to check if
    # the directory name starts with 'group_' or not
    base_dir = os.path.basename(os.path.normpath(args.storage_path))
    if base_dir.startswith('group_'):
        writer = SummaryWriter(os.path.join(args.storage_path, '..', 'logs'))
    else:
        writer = SummaryWriter(os.path.join(args.storage_path, 'logs'))

    # Train models
    print("Training models...")

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    with MatplotlibTrainingVisualizer(
            list_metrics = [
                {"name": "Loss", "scale": "log"},
                {"name": "Accuracy"}],
            list_models = [
                {'name': "SPDnet (train)", 'plot_options': {'color': 'white', 'linestyle': '--'}},
                {'name': "SPDnet (val)", 'plot_options': {'color': 'yellow', 'linestyle': '-.'}}
                ],
            layout = (1, 2),
            figure_options = {'figsize': (12, 6)}) as visualizer:
        for epoch in range(args.epochs):

            print(f"Epoch {epoch+1}/{args.epochs}")
            print("-" * 10)

            # Training
            print("Training...")
            spdnet.train()
            _train_loss = 0
            _train_acc = 0
            for X, y in tqdm.tqdm(train_loader):

                X = X.to(args.device).type(args.dtype)
                y = y.to(args.device)


                optimizer.zero_grad()

                y_pred = spdnet(X)
                loss= loss_fn(y_pred, y)
                acc = (y_pred.argmax(dim=1) == y).float().mean() * 100
                _train_loss += loss.item()
                _train_acc += acc

                loss.backward()


                optimizer.step()


            train_loss.append(_train_loss/len(train_loader))
            train_acc.append(_train_acc/len(train_loader))

            # Validation
            spdnet.eval()
            with torch.no_grad():
                print("Validating...")
                _val_loss = 0
                _val_acc = 0
                for X, y in tqdm.tqdm(val_loader):
                    X = X.to(args.device).type(args.dtype)
                    y = y.to(args.device)


                    y_pred = spdnet(X)
                    loss= loss_fn(y_pred, y)
                    acc = (y_pred.argmax(dim=1) == y).float().mean() * 100
                    _val_loss += loss.item()
                    _val_acc += acc

                val_loss.append(_val_loss/len(val_loader))
                val_acc.append(_val_acc/len(val_loader))


            # Scheduler
            scheduler.step(val_loss[-1])

            # Tensorboard
            writer.add_scalars('Loss/train',
                               {model_name:
                                 train_loss[-1]},
                                epoch)
            writer.add_scalars('Accuracy/train',
                               {model_name: train_acc[-1]},
                                epoch)
            writer.add_scalars('Loss/val',
                            {model_name: val_loss[-1]},
                             epoch)
            writer.add_scalars('Accuracy/val',
                            {model_name: val_acc[-1]},
                             epoch)
            writer.add_scalars('Learning rate',
                            {model_name: optimizer.param_groups[0]['lr']},
                            epoch)

            # Update matplotlib visualizer
            visualizer.update(
                [
                    [train_loss[-1], val_loss[-1]],
                    [train_acc[-1], val_acc[-1]],
                ]
            )

            print("Epoch finished.")
            print("Training loss: {:.4f} - Validation loss: {:.4f}".format(
                train_loss[-1], val_loss[-1]))
            print("Training accuracy: {:.4f} - Validation accuracy: {:.4f}".format(
                train_acc[-1], val_acc[-1]))

            print("-" * 100)
            print("\n\n")

    # Test
    print("Testing...")
    test_dataset = AFEWSPDnetDataset(
            directory_path = args.dataset_path,
            preload = True,
            shuffle = False,
            subset = "val",
            rng = torch.Generator().manual_seed(args.seed),
            device = args.device,
            verbose = 1)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False)

    test_loss = 0
    test_acc = 0
    spdnet.eval()
    with torch.no_grad():
        for X, y in tqdm.tqdm(test_loader):
            X = X.to(args.device).type(args.dtype)
            y = y.to(args.device)


            y_pred = spdnet(X)
            loss= loss_fn(y_pred, y)
            acc = (y_pred.argmax(dim=1) == y).float().mean() * 100
            test_loss += loss.item()
            test_acc += acc

        test_loss /= len(test_loader)
        test_acc /= len(test_loader)

    print("Test loss: {:.4f}".format(test_loss))
    print("Test accuracy: {:.4f}".format(test_acc))
    writer.add_scalars('Loss/test', {model_name: test_loss})
    writer.add_scalars('Accuracy/test', {model_name: test_acc})
    writer.flush()
    writer.close()

