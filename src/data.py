# ========================================
# FileName: data.py
# Date: 29 juin 2023 - 16:38
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: Data classes and utilities
# =========================================

import os
import glob
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
from scipy.stats import wishart
from itertools import product
from tqdm import tqdm

# Type hints
from typing import (
    Optional, Iterable, Tuple
)
from numpy.typing import ArrayLike


# ========================================
# Class: FixedWishartDataset
# ========================================
class FixedWishartDataset(Dataset):
    """Dataset of Wishart distributed matrices with a fixed degree of freedom
    and scale matrix."""

    def __init__(self, df: int, scale: ArrayLike, size: Optional[int] = 1000,
                 seed: Optional[int] = None,
                 device: Optional[torch.device] = None) -> None:
        """Constructor of the dataset with a given degree of freedom
        and scale matrix.

        Parameters
        ----------
        df : int
            Degree of freedom of the Wishart distribution.

        scale : ArrayLike
            Scale matrix of the Wishart distribution.

        size : int, optional
            Number of samples in the dataset, by default 1000.

        seed : int, optional
            Seed for the random number generator, by default None.

        device : torch.device, optional
            Device on which the data is stored, by default None.
        """
        self.df = df
        self.scale = scale
        self.size = size
        self.seed = seed
        self.device = device

        # Create the random number generator
        self.rng = np.random.default_rng(seed)

        # Create the Wishart distribution
        self.wishart = wishart(df, scale)

    def __len__(self) -> int:
        """Returns the size of the dataset"""
        return self.size

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns the idx-th sample of the dataset"""

        if idx >= self.size:
            raise IndexError(f"Index {idx} out of range"
                             " (size of the dataset is {self.size})")

        # Generate the random matrix
        X = self.wishart.rvs(
                size=1,
                random_state=self.rng).astype(np.float32)

        # Convert to tensor
        X = torch.from_numpy(X).to(self.device)

        return X

    def get_all(self) -> torch.Tensor:
        """Returns all the samples of the dataset"""
        return torch.from_numpy(self.wishart.rvs(
            size=self.size, random_state=self.rng)).to(self.device)

    def __repr__(self) -> str:
        """Returns the representation of the dataset"""
        return f"WishartDataset(df={self.df}, scale={self.scale}, "\
               f"size={self.size}, seed={self.seed})"


# ========================================
# Class: WishartDataset
# ========================================
class WishartDataset(Dataset):
    """Dataset of Wishart distributed matrices with a variable degree of
    freedom and scale matrix."""

    def __init__(self, df: Iterable[int], scale: Iterable[ArrayLike],
                 size: Optional[int] = 1000,
                 seed: Optional[int] = None,
                 device: Optional[torch.device] = None) -> None:
        """Constructor of the dataset with a given degree of freedom
        and scale matrix.

        Parameters
        ----------
        df : Iterable[int]
            Iterable of degree of freedom of the Wishart distribution.

        scale : Iterable[ArrayLike]
            Iterable of scale matrix of the Wishart distribution.

        size : int, optional
            Number of samples in the dataset per combination of parameters,
            by default 1000.

        seed : int, optional
            Seed for the random number generator, by default None.

        device : torch.device, optional
            Device on which the data is stored, by default None.
        """
        self.df = df
        self.scale = scale
        self.size = size
        self.seed = seed
        self.device = device

        # Create the random number generator
        self.rng = np.random.default_rng(seed)

        # Create a mapping between the index of the dataset and the
        # index of the parameters + the number of samples per combination
        self.idx_to_param = list(product(
            range(len(self.df)), range(len(self.scale)), range(self.size)
        ))
        self.total_size = len(self.idx_to_param)

    def __len__(self) -> int:
        """Returns the size of the dataset"""
        return self.total_size

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns the idx-th sample of the dataset"""

        # Get the index of the parameters and the index per combination
        idx_df, idx_scale, idx_sample = self.idx_to_param[idx]

        # Check if the index is valid
        if idx_df >= len(self.df) or idx_scale >= len(self.scale) \
                or idx_sample >= self.size:
            raise IndexError(f"Index {idx} out of range"
                             " (size of the dataset is {self.total_size})")

        # Generate the random matrix
        X = wishart(self.df[idx_df], self.scale[idx_scale]).rvs(
            size=1, random_state=self.rng).astype(np.float32)

        # Convert to tensor
        X = torch.from_numpy(X).to(self.device)

        return X

    def __repr__(self) -> str:
        """Returns the representation of the dataset"""
        return f"WishartDataset(df={self.df}, scale={self.scale}, "\
            f"size={self.size}, seed={self.seed})"


# ========================================
# AFEW SPnet dataset
# ========================================
def loadmat_spdnet(file_path: str):
    """Load .mat file from spdnet paper datasets.

    Parameters
    -----------
    file_path: str
        Path to the mat file

    Returns
    -------
    array-like
        Numpy array of the SPD matrix
    """
    return loadmat(file_path)['Y1']


class AFEWSPDnetDataset(Dataset):
    """Dataset from paper: A Riemannian Network for SPD Matrix Learning, CVPR 2017.

    The dataset is provided by the authors from their GitHub repository:
        https://github.com/zhiwu-huang/SPDNet/tree/master

    It consists in pre-computed SPD matrices from AFEW dataset, which doesn't
    help reproducibility. We suppose that the data is already downloaded and 
    unzipped in directory_path."""


    def __init__(self, directory_path: str, preload: bool = False,
                 shuffle: bool = False, subset: str = "train",
                 rng: Optional[torch.Generator] = None,
                 device: Optional[torch.device] = None,
                 verbose: int = 0) -> None:
        """Constructor of the dataset.

        Parameters
        ----------
        directory_path : str
            Path to the directory containing the dataset.

        subset : str, optional
            Subset of the dataset to load, by default "train".

        preload : bool, optional
           Whether to preload the dataset in memory, by default False.

        shuffle : bool, optional
           Whether to shuffle the dataset, by default False.

        rng : np.random.Generator, optional
           Random number generator, by default None.

        device : torch.device, optional
           Device on which the data is stored, by default None.

        verbose : int, optional
            Verbosity level, by default 0.
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory {directory_path} does not exist")

        super().__init__()
        self.directory_path = directory_path
        self.preload = preload
        self.device = device
        self.verbose = verbose
        self.shuffle = shuffle
        self.rng = rng
        self.subset = subset
        self.labels_str = ["Angry", "Disgust", "Fear", "Happy", "Neutral",
                           "Sad", "Surprise"]

        self.d = 400

        self._discover_files()
        self._preload_data()
        self._shuffle()

    def _shuffle(self) -> None:
        """Shuffle the dataset"""
        if self.shuffle:
            indexes = torch.randperm(len(self.list_files), generator=self.rng)
            self.list_files = [self.list_files[i] for i in indexes]
            self.labels = [self.labels[i] for i in indexes]

    def _discover_files(self) -> None:
        """Discovers the files in the dataset directory"""
        self.list_files = glob.glob(
                os.path.join(self.directory_path, self.subset, "**/*.mat")
        )
        if len(self.list_files) == 0:
            self.list_files = glob.glob(
                os.path.join(self.directory_path, "spdface_400_inter_histeq",
                             self.subset, "**/*.mat")
        )
        # Find the labels from the file names. -1 because labels start at 0
        # in torch
        self.labels = [int(os.path.basename(os.path.dirname(x))) - 1
                       for x in self.list_files]
        if len(self.labels) == 0:
            print(f"No data found for subset {self.subset}")

    def __len__(self) -> int:
        return len(self.list_files)

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        if self.preload:
            return self.data[index], self.labels[index]
        else:
            X = torch.from_numpy(
                    loadmat_spdnet(self.list_files[index])).to(self.device)
            return X, self.labels[index]

    def __repr__(self) -> str:
        string = f"AFEWSPnetDataset(directory_path={self.directory_path}, "
        string += f"preload={self.preload}, subset={self.subset})"
        return string

    def _preload_data(self) -> None:
        """Preload the data"""
        if self.preload:
            if self.verbose:
                pbar = tqdm(total=len(self.list_files))
            self.data = []
            for file in self.list_files:
                if self.verbose:
                    filename = os.path.basename(file)
                    pbar.set_description(f"Loading ({filename})")
                    pbar.refresh()
                self.data.append(
                    torch.from_numpy(loadmat_spdnet(file)).to(self.device)
                )

                if self.verbose:
                    pbar.update(1)

            if self.verbose:
                pbar.close()
        self.n_classes = len(np.unique(self.labels))


class SPMartiDataset(Dataset):
    """Pre-computed covariance matrices data described in:
       https://marti.ai/qfin/2020/02/03/sp500-sharpe-vs-corrmats.html
       Accessed: 11/09/23 3:40pm.

    The dataset corresponds to financial data from S&P evaluations 
    with classes:
        * correlation matrices associated to a stressed market,
        * correlation matrices associated to a rally market,
        * correlation matrices associated to a normal market.
    
    It consists in pre-computed SPD matrices. We suppose that the data is
    already downloaded and unzipped in directory_path."""

    def __init__(self, directory_path: str, preload: bool = False,
                 shuffle: bool = False,
                 rng: Optional[torch.Generator] = None,
                 device: Optional[torch.device] = None,
                 verbose: int = 0) -> None:
        """Constructor of the dataset.

        Parameters
        ----------
        directory_path : str
            Path to the directory containing the dataset.

        preload : bool, optional
           Whether to preload the dataset in memory, by default False.

        shuffle : bool, optional
           Whether to shuffle the dataset, by default False.

        rng : np.random.Generator, optional
           Random number generator, by default None.

        device : torch.device, optional
           Device on which the data is stored, by default None.

        verbose : int, optional
            Verbosity level, by default 0.
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory {directory_path} does not exist")

        super().__init__()
        self.directory_path = directory_path
        self.preload = preload
        self.device = device
        self.verbose = verbose
        self.shuffle = shuffle
        self.rng = rng
        self.labels_str = ["Stressed", "Normal", "Rally"]

        self.d = 80

        self._discover_files()
        self._preload_data()
        self._shuffle()

    def _shuffle(self) -> None:
        """Shuffle the dataset"""
        if self.shuffle:
            indexes = torch.randperm(len(self.list_files), generator=self.rng)
            self.list_files = [self.list_files[i] for i in indexes]
            self.labels = [self.labels[i] for i in indexes]

    def _discover_files(self) -> None:
        """Discovers the files in the dataset directory"""
        self.list_files = glob.glob(
                os.path.join(self.directory_path, "**/*.npy")
        )
        if len(self.list_files) == 0:
            self.list_files = glob.glob(
                os.path.join(self.directory_path, "*.npy")
            )
        # Find the labels from the file names
        self.labels = [int(os.path.basename(x).split('class_')[-1].split('.npy')[0])
                       for x in self.list_files]

    def __len__(self) -> int:
        return len(self.list_files)

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        if self.preload:
            return self.data[index], self.labels[index]
        else:
            X = torch.from_numpy(
                    np.load(self.list_files[index])).real.to(self.device)
            return X, self.labels[index]

    def __repr__(self) -> str:
        string = f"SPMartiDataset(directory_path={self.directory_path}, "
        string += f"preload={self.preload})"
        return string

    def _preload_data(self) -> None:
        """Preload the data"""
        if self.preload:
            if self.verbose:
                pbar = tqdm(total=len(self.list_files))
            self.data = []
            for file in self.list_files:
                if self.verbose:
                    pbar.set_description(f"Loading {file}")
                    pbar.refresh()
                self.data.append(
                    torch.from_numpy(np.load(file)).real.to(self.device)
                )

                if self.verbose:
                    pbar.update(1)

            if self.verbose:
                pbar.close()
        self.n_classes = len(np.unique(self.labels))


class HDM05Dataset(Dataset):
    """Dataset of action recongition from: 
    https://resources.mpi-inf.mpg.de/HDM05/, that has been preprocessed
    to generate SPD matrices from the skeleton data. The preprocessing has
    been done and the data has been downloaded from:
    https://github.com/zhiwu-huang/SPDNet"""

    def __init__(self, directory_path: str, preload: bool = False,
                shuffle: bool = False,
                rng: Optional[torch.Generator] = None,
                device: Optional[torch.device] = None,
                verbose: int = 0) -> None:
        """Constructor of the dataset.
        Parameters
        ----------
        directory_path : str
            Path to the directory containing the dataset.

        preload : bool, optional
           Whether to preload the dataset in memory, by default False.

        shuffle : bool, optional
           Whether to shuffle the dataset, by default False.

        rng : np.random.Generator, optional
           Random number generator, by default None.

        device : torch.device, optional
           Device on which the data is stored, by default None.

        verbose : int, optional
            Verbosity level, by default 0.
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory {directory_path} does not exist")

        super().__init__()
        self.directory_path = directory_path
        self.preload = preload
        self.device = device
        self.verbose = verbose
        self.shuffle = shuffle
        self.rng = rng
        self.d = 93
        self._discover_files()
        self._preload_data()
        self._shuffle()


    def _shuffle(self) -> None:
        """Shuffle the dataset"""
        if self.shuffle:
            indexes = torch.randperm(len(self.list_files), generator=self.rng)
            self.list_files = [self.list_files[i] for i in indexes]
            self.labels = [self.labels[i] for i in indexes]


    def _discover_files(self) -> None:
        """Discovers the files in the dataset directory"""
        self.list_files = glob.glob(
                os.path.join(self.directory_path, "feature/**/*.mat")
        )
        if len(self.list_files) == 0:
            self.list_files = glob.glob(
                os.path.join(self.directory_path, "*.mat")
            )
        # Find the labels from the files directory names
        _str_labels = [os.path.basename(os.path.dirname(x)) for x in self.list_files]
    
        # Convert the labels to integers and get the mapping between the
        # integer and the label
        unique_labels = np.unique(_str_labels)
        self.label_to_int = {label: i for i, label in enumerate(unique_labels)}
        self.int_to_label = {i: label for i, label in enumerate(unique_labels)}
        self.labels = [self.label_to_int[label] for label in _str_labels]
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.labels_str = [self.int_to_label[i] for i in range(len(unique_labels))]
        self.n_classes = len(unique_labels)
        self.n_features = 93

    def _preload_data(self) -> None:
        """Preload the data"""
        if self.preload:
            if self.verbose:
                pbar = tqdm(total=len(self.list_files))

            self.data = []
            for file in self.list_files:
                if self.verbose:
                    pbar.set_description(f"Loading {file}")
                    pbar.refresh()

                self.data.append(
                    torch.from_numpy(loadmat_spdnet(file))
                )

                if self.verbose:
                    pbar.update(1)

            if self.verbose:
                pbar.close()

    def __len__(self) -> int:
        return len(self.list_files)

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        if self.preload:
            return self.data[index], self.labels[index]
        else:
            X = torch.from_numpy(
                    loadmat_spdnet(self.list_files[index])).to(self.device)
            return X, self.labels[index]
