# ========================================
# FileName: utils.py
# Date: 29 juin 2023 - 17:14
# Author: Ammar Mian
# Email: ammar.mian@univ-smb.fr
# GitHub: https://github.com/ammarmian
# Brief: Utilities functions
# =========================================

import torch
from typing import Optional
import logging
from rich.logging import RichHandler
from typing import List, Tuple, Union, Callable, Dict
from math import prod

import matplotlib.pyplot as plt

# ========================================
# Logging and formatting
# ========================================
class MatplotlibTrainingVisualizer:
    """Class to visualize training progress of one or several models."""

    def __init__(self, list_metrics: List[Dict],
                list_models: List[Dict], layout: Optional[Tuple[int, int]] = None,
                 figure_options: Optional[Dict] = None,
                 axes_options: Optional[Dict] = None) -> None:
        """Initialize the TrainingVisualizer object

        Parameters
        ----------
        list_metrics : List[Dict]
            List of dictionaries containing the metrics to plot for each model.
            Each dictionary should contain the following keys:
                - "name": Name of the metric
            
            And can contain the following keys:
                - scale: Scale of the metric (linear or log)
                - ylim: Limits of the y axis
                - ylabel: Label of the y axis

        list_models : List[Dict]
            List of dictionaries containing the models to plot. Each dictionary
            should contain the following keys:
                - "name": Name of the model

            And can contain the following keys:
                - plot_options: Options for the plot of the model that will be
                passed to the matplotlib plot function

        layout : Optional[Tuple[int, int]], optional
            Layout of the figure, by default None. Meaning that the figure will
            be a square with the number of subplots equal to the number of
            metrics.
        
        figure_options : Optional[Dict], optional
            Options for the figure, by default None

        axes_options : Optional[Dict], optional
            Options for the axes, by default None
        """
        
        assert len(list_metrics) > 0, "At least one metric must be provided"
        assert len(list_models) > 0, "At least one model must be provided"
        assert all([isinstance(metric, dict) for metric in list_metrics]), \
                "Metrics must be provided as dictionaries"
        assert all([isinstance(model, dict) for model in list_models]), \
                "Models must be provided as dictionaries"

        self.n_metrics = len(list_metrics)
        if layout is None:
            if not self.n_metrics == 1:
                self.layout = (1, 1)
            elif self.n_metrics == 2:
                self.layout = (2, 1)
            else:
                self.layout = (self.n_metrics // 2, 2)
        else:
            assert prod(layout) >= self.n_metrics, \
                    f"Layout {layout} is not big enough to display " \
                    f"{self.n_metrics} metrics"
            self.layout = layout

        self.list_metrics = list_metrics
        self.list_models = list_models
        self.figure_options = figure_options
        self.axes_options = axes_options
        self.n_models = len(list_models)
        self.mapping_metric_axes = lambda i: (i // self.layout[0], i % self.layout[0])

        # Create the figure
        self.figure, self.axes = plt.subplots(*self.layout, **self.figure_options)
        if self.layout[1] == 1:
            self.axes = self.axes.reshape(1, self.n_metrics)
        elif self.layout[0] == 1:
            self.axes = self.axes.reshape(self.n_metrics, 1)

        # Setting axes options
        if self.axes_options is not None:
            for key, value in self.axes_options.items():
                setattr(self.axes, key, value)

        # Create the plots
        self.plots = []
        for i, metric in enumerate(self.list_metrics):
            self.plots.append([])
            _i, _j = self.mapping_metric_axes(i)
            for model in self.list_models:
                self.plots[i].append(
                    self.axes[_i, _j].plot([], [], label=model['name'],
                                           **model["plot_options"])[0]
                )

            # Setting metric options
            if "scale" in metric:
                self.axes[_i, _j].set_yscale(metric["scale"])
            if "ylim" in metric:
                self.axes[_i, _j].set_ylim(metric["ylim"])
            if "ylabel" in metric:
                self.axes[_i, _j].set_ylabel(metric["ylabel"])
            else:
                self.axes[_i, _j].set_ylabel(metric["name"])
            self.axes[_i, _j].set_xlabel("Epochs")

        # Initialize the data
        self.data = []
        for i, metric in enumerate(self.list_metrics):
            self.data.append([])
            for model in self.list_models:
                self.data[i].append([])

        # Initialize the legend
        self.legend = []
        for i, metric in enumerate(self.list_metrics):
            self.legend.append([])
            _i, _j = self.mapping_metric_axes(i)
            for j, model in enumerate(self.list_models):
                self.legend[i].append(
                    self.axes[_i, _j].legend(loc="upper right")
                )

        # Initialize the epoch counter
        self.epoch = 0

    def __enter__(self):
        self.figure.canvas.draw()
        plt.show(block=False)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:  # An exception occurred inside the `with` block
            print(f"Exception: {exc_type}, {exc_value}, {traceback}")
        plt.close("all")

    def update(self, data: List[List[float]]) -> None:
        """Update the plots with the new data.

        Parameters
        ----------
        data : List[List[float]]
            List of data to update the plots with. The first dimension
            corresponds to the metrics, the second to the models.
        """

        assert len(data) == self.n_metrics, \
                f"Data must be a list of {self.n_metrics} elements"
        assert all([len(data[i]) == self.n_models for i in range(self.n_metrics)]), \
                f"Each element of data must be a list of {self.n_models} elements"

        # Update the data
        for i, _ in enumerate(self.list_metrics):
            for j, _ in enumerate(self.list_models):
                self.data[i][j].append(data[i][j])

        # Update the plots
        for i, _ in enumerate(self.list_metrics):
            _i, _j = self.mapping_metric_axes(i)
            for j, _ in enumerate(self.list_models):
                self.plots[i][j].set_data(
                        list(range(self.epoch + 1)), self.data[i][j]
                    )
                self.axes[_i, _j].relim()
                self.axes[_i, _j].autoscale_view()

        self.epoch += 1

        # Update the figure
        plt.draw()
        plt.pause(0.05)


def setup_logging(level: str = "INFO", format: str ="%(message)s",
                  markup: bool = True) -> logging.Logger:
    """Setup logging to use rich handler

    Parameters
    ----------
    level : str, optional
        Logging level, by default "INFO"

    format : str, optional
        Format of the logging messages, by default "%(messages)"

    markup : bool, optional
        Whether to use rich markup, by default True

    Returns
    -------
    logging.Logger
        Logger object
    """
    logging.basicConfig(
        level=level, format=format, datefmt="[%X]",
        handlers=[RichHandler(markup=markup,
                            rich_tracebacks=True)]
    )
    return logging.getLogger("rich")


def format_params(params: dict, markup: bool = True) -> str:
    """Format a dictionary of parameters for logging
    Parameters
    ----------
    params : dict
        Dictionary of parameters
    markup : bool, optional
        Whether to use rich markup, by default True
    Returns
    -------
    str
        Formatted string
    """
    string = ""
    for key, value in params.items():
        if markup:
            string += "    :black_circle_for_record: "
        else:
            string += "    - "
        string += f"{key}: {value}\n"
    return string


# ========================================
# SPD matrix generation
# TODO: Generate with Toepitz structure,
#       Generate with circulant structure
#       Generate with given eigenvalues distribution
#       Generate with Stiefel manifold
# ========================================
def generate_spd_matrices(dim: int, batch_size: Optional[int] = 1,
                          device: Optional[str] = "cpu",
                          seed: Optional[int] = 0,
                          eps: Optional[float] = 1e-6) -> torch.Tensor:
    """
    Generate a batch of symmetric positive definite matrices

    Parameters
    ----------
    dim : int
        Dimension of the matrices

    batch_size : int, optional
        Batch size of the matrices, by default 1

    device : str, optional
        Device to put the matrices on, by default "cpu"

    seed : int, optional
        Seed for the random number generator, by default 0

    eps : float, optional
        Small value to add to the diagonal to ensure positive definiteness,
        by default 1e-6

    Returns
    -------
    torch.Tensor
        Batch of symmetric positive definite matrices
    """
    # Set the seed
    torch.manual_seed(seed)

    # Generate a batch of random matrices
    X = torch.randn(batch_size, dim, dim, device=device)

    # Compute the product of the matrices with their transposes
    X = torch.bmm(X, X.transpose(1, 2))

    # Add a small value to the diagonal to ensure positive definiteness
    X += eps * torch.eye(dim, device=device).expand(batch_size, dim, dim)

    if batch_size == 1:
        return X.squeeze(0)

    return X


# ========================================
# Functions to do checks
# ========================================
def is_spd(X: torch.Tensor, tol: Optional[float] = 1e-4) -> bool:
    """
    Check if a matrix is symmetric positive definite

    Parameters
    ----------
    X : torch.Tensor
        Matrix to check

    tol : float, optional
        Tolerance for the check, by default 1e-6

    Returns
    -------
    bool
        True if the matrix is symmetric positive definite, False otherwise
    """
    # Check if symmetric
    if not torch.allclose(X, X.t(), rtol=tol, atol=tol):
        return False

    # Check if positive definite
    try:
        torch.linalg.cholesky(X)
        return True
    except RuntimeError:
        distance = torch.linalg.norm(X - X.t())
        logging.debug("Matrix is not positive definite with tolerance {}"
                      .format(tol))
        logging.debug("Distance to symmetric matrix: {}".format(distance))
        return False
