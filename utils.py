import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import expit, logit
import jax.tree_util as jtu

import json

from typing import Tuple, Literal, Optional, Dict, Any

from flowjax.flows import masked_autoregressive_flow, triangular_spline_flow
from flowjax.distributions import Normal, Transformed, StudentT

import optax
import equinox as eqx

from tqdm import tqdm, trange

from pathlib import Path

import warnings

# -----------------------------
# Data cleaning utilities
# -----------------------------

def _handle_special_values(
    samples: jnp.ndarray,
    strategy: str = "clip",
    fill_value: float | None = None,
    clip_percentile: float = 99.9,
) -> tuple[jnp.ndarray, dict]:
    """Handle infinite and NaN values in data.

    Args:
        samples: Input samples, shape (N, D).
        strategy: Method to handle special values. Options:
            - "clip": Replace inf with max/min finite values at clip_percentile.
            - "fill": Replace inf/nan with fill_value.
            - "drop": Mark rows for removal (returns mask).
        fill_value: Value to use when strategy="fill".
        clip_percentile: Percentile to use for clipping (e.g., 99.9 means clip
            to 99.9th percentile of finite values).

    Returns:
        cleaned: Cleaned samples, shape (N, D) or (N', D) if strategy="drop".
        info: Dictionary with cleaning statistics.
    """
    samples_np = np.array(samples)
    n_inf = np.isinf(samples_np).sum()
    n_nan = np.isnan(samples_np).sum()
    
    info = {
        "n_inf": int(n_inf),
        "n_nan": int(n_nan),
        "n_total_special": int(n_inf + n_nan),
        "strategy": strategy,
    }

    if strategy == "clip":
        # Compute percentile bounds from finite values only
        for col_idx in range(samples_np.shape[1]):
            col = samples_np[:, col_idx]
            finite_mask = np.isfinite(col)
            
            if np.any(finite_mask):
                finite_vals = col[finite_mask]
                lower_bound = np.percentile(finite_vals, 100 - clip_percentile)
                upper_bound = np.percentile(finite_vals, clip_percentile)
                
                # Replace +inf with upper bound, -inf with lower bound
                col = np.where(np.isposinf(col), upper_bound, col)
                col = np.where(np.isneginf(col), lower_bound, col)
                
                # Replace NaN with median
                if np.any(np.isnan(col)):
                    median_val = np.median(finite_vals)
                    col = np.where(np.isnan(col), median_val, col)
                
                samples_np[:, col_idx] = col
                info[f"col_{col_idx}_upper_bound"] = float(upper_bound)
                info[f"col_{col_idx}_lower_bound"] = float(lower_bound)

    elif strategy == "fill":
        if fill_value is None:
            raise ValueError("fill_value must be provided when strategy='fill'")
        samples_np = np.nan_to_num(
            samples_np, 
            nan=fill_value, 
            posinf=fill_value, 
            neginf=fill_value,
        )
        info["fill_value"] = fill_value

    elif strategy == "drop":
        # Return mask of finite rows
        finite_mask = np.all(np.isfinite(samples_np), axis=1)
        samples_np = samples_np[finite_mask]
        info["n_dropped"] = int((~finite_mask).sum())
        info["n_remaining"] = int(finite_mask.sum())
        info["mask"] = finite_mask

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return jnp.array(samples_np), info


# -----------------------------
# Low-level JAX transformation utilities
# -----------------------------

def _classify_columns(samples: jnp.ndarray) -> jnp.ndarray:
    """Classify each column according to the transformation rule.

    Labels:
        0: "none"
        1: "log" (strictly positive, max > 2π)
        2: "angle_2pi" (0 < x < 2π and max > π)
        3: "angle_pi" (0 < x < π)
        4: "angle_signed_pi" (-π < x < 0 < max < π)

    Args:
        samples: Raw samples, shape (N, D).

    Returns:
        labels: Integer labels per column, shape (D,).
    """
    col_min = jnp.min(samples, axis=0)
    col_max = jnp.max(samples, axis=0)

    is_pos = col_min > 0.0
    is_log = is_pos & (col_max > 2 * jnp.pi)
    is_ang_2pi = is_pos & (col_max < 2 * jnp.pi) & (col_max > jnp.pi)
    is_ang_pi = is_pos & (col_max < jnp.pi)
    is_signed = (col_min < 0.0) & (col_max > 0.0) & (col_min > -jnp.pi) & (col_max < jnp.pi)

    labels = jnp.zeros_like(col_min, dtype=jnp.int32)
    labels = jnp.where(is_log, 1, labels)
    labels = jnp.where(is_ang_2pi, 2, labels)
    labels = jnp.where(is_ang_pi, 3, labels)
    labels = jnp.where(is_signed, 4, labels)

    return labels


def _label_to_name(label: int) -> str:
    """Convert integer label to human-readable transformation name.

    Args:
        label: Integer transform label.

    Returns:
        name: String name of the transform.
    """
    mapping = {
        0: "none",
        1: "log",
        2: "angle_2pi",
        3: "angle_pi",
        4: "angle_signed_pi",
    }
    return mapping[int(label)]


@jax.jit
def _forward_transform_array(samples: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Apply column-wise forward transforms in a vectorized, JIT-compiled way.

    Args:
        samples: Input samples, shape (N, D).
        labels: Integer transform labels per column, shape (D,).

    Returns:
        transformed: Transformed samples, shape (N, D).
    """
    def transform_column(col: jnp.ndarray, label: jnp.ndarray) -> jnp.ndarray:
        """Transform a single column according to its label.

        Args:
            col: Column values, shape (N,).
            label: Integer transform label.

        Returns:
            transformed_col: Transformed column, shape (N,).
        """
        def case_log(x):          return jnp.log(x)
        def case_angle_2pi(x):    return logit(x / (2 * jnp.pi))
        def case_angle_pi(x):     return logit(x / jnp.pi)
        def case_signed_pi(x):    return logit((x + jnp.pi) / (2 * jnp.pi))

        return jax.lax.switch(
            label,
            (
                lambda x: x,          # 0: none
                case_log,             # 1
                case_angle_2pi,       # 2
                case_angle_pi,        # 3
                case_signed_pi,       # 4
            ),
            col,
        )

    transformed = jax.vmap(transform_column, in_axes=(1, 0), out_axes=1)(samples, labels)
    return transformed


@jax.jit
def _inverse_transform_array(samples: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Invert column-wise transforms in a vectorized, JIT-compiled way.

    Args:
        samples: Transformed samples, shape (N, D).
        labels: Integer transform labels per column, shape (D,).

    Returns:
        recovered: Recovered samples in original space, shape (N, D).
    """
    def inverse_column(col: jnp.ndarray, label: jnp.ndarray) -> jnp.ndarray:
        """Invert the transform for a single column.

        Args:
            col: Transformed column values, shape (N,).
            label: Integer transform label.

        Returns:
            recovered_col: Recovered column, shape (N,).
        """
        def case_log_inv(x):          return jnp.exp(x)
        def case_angle_2pi_inv(x):    return expit(x) * (2 * jnp.pi)
        def case_angle_pi_inv(x):     return expit(x) * jnp.pi
        def case_signed_pi_inv(x):    return expit(x) * (2 * jnp.pi) - jnp.pi

        return jax.lax.switch(
            label,
            (
                lambda x: x,            # 0: none
                case_log_inv,           # 1
                case_angle_2pi_inv,     # 2
                case_angle_pi_inv,      # 3
                case_signed_pi_inv,     # 4
            ),
            col,
        )

    recovered = jax.vmap(inverse_column, in_axes=(1, 0), out_axes=1)(samples, labels)
    return recovered


@jax.jit
def _apply_whitening(
    samples_transformed: jnp.ndarray,
    mu: jnp.ndarray,
    L: jnp.ndarray
) -> jnp.ndarray:
    """Apply whitening using provided mean and Cholesky factor."""
    # Handle 1D case (single column)
    if samples_transformed.ndim == 1 or samples_transformed.shape[1] == 1:
        if L.ndim == 2 and L.shape == (1, 1):
             std = L[0, 0]
        else:
             std = jnp.diag(L)
             
        # Broadcast subtraction and division
        x_white = ((samples_transformed - mu) / std).astype(jnp.float64)
    else:
        # Multi-dimensional case
        x_centered = samples_transformed - mu
        # Solve L @ y = x_centered.T  => y = L^-1 @ x_centered.T
        x_white = jnp.linalg.solve(L, x_centered.T).T.astype(jnp.float64)
        
    return x_white


@jax.jit
def _whiten(samples_transformed: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute statistics and whiten transformed data."""
    mu = jnp.mean(samples_transformed, axis=0)
    
    # Handle 1D case (single column)
    if samples_transformed.ndim == 1 or samples_transformed.shape[1] == 1:
        std = jnp.std(samples_transformed, axis=0)
        std = jnp.where(std == 0, 1.0, std)  # Avoid division by zero
        
        # Ensure consistent shapes for 1D
        if samples_transformed.ndim == 1:
            L = std.reshape(1, 1)
            mu = mu.reshape(1)
        else:
            L = jnp.diag(std)
            
        x_white = _apply_whitening(samples_transformed, mu, L)
    else:
        # Multi-dimensional case
        cov = jnp.cov(samples_transformed, rowvar=False)
        regularization = 1e-6
        cov = cov + regularization * jnp.eye(cov.shape[0])
        L = jnp.linalg.cholesky(cov)
        
        x_white = _apply_whitening(samples_transformed, mu, L)
    
    return x_white, mu, L


@jax.jit
def _inverse_whiten(
    x_white: jnp.ndarray,
    mu: jnp.ndarray,
    L: jnp.ndarray,
) -> jnp.ndarray:
    """Invert whitening transform.

    Args:
        x_white: Whitened samples, shape (N, D).
        mu: Mean used for whitening, shape (D,).
        L: Cholesky factor used for whitening, shape (D, D).

    Returns:
        x_rec: Reconstructed transformed samples, shape (N, D).
    """
    # Handle 1D case
    if x_white.ndim == 1 or (x_white.ndim == 2 and x_white.shape[1] == 1):
        if L.ndim == 2 and L.shape == (1, 1):
            # Extract scalar from 1x1 matrix
            std = L[0, 0]
            if x_white.ndim == 1:
                x_rec = x_white * std + mu[0]
            else:
                x_rec = x_white * std + mu
        else:
            x_rec = x_white * jnp.diag(L) + mu
    else:
        # Multi-dimensional case
        x_rec = (L @ x_white.T).T + mu
    
    return x_rec



# -----------------------------
# High-level Data class
# -----------------------------

class Data:
    """Container for tabular data with JAX-based transforms and whitening.

    This class:
      * Extracts numeric samples from a pandas DataFrame.
      * Handles infinite and NaN values.
      * Classifies each column into a transformation type (log/angle/none).
      * Applies forward and inverse transforms using JAX (jit + vmap).
      * Performs whitening in transformed space and can invert it.
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        cols: list[str] | None = None,
        transforms: list[Literal["none", "log", "angle_2pi", "angle_pi", "angle_signed_pi", "choose"]] | None = None,
        handle_inf: Literal["clip", "fill", "drop", "none"]= "none",
        inf_fill_value: float | None = None,
        inf_clip_percentile: float = 99.9,
        mask: np.ndarray |None = None,
        # Argument for internal use or manual override
        transform_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Data container.

        You can either specify a mask of handle_inf. Masking will take precedence.

        Args:
            data_df: Input DataFrame.
            cols: Optional subset of columns to use.
            handle_inf: Strategy for handling inf/nan values ("clip", "fill", "drop").
            transform_params: If provided, bypasses the calculation of transformation
                types and whitening statistics, and uses these instead. Used by `from_saved`.
            mask: Which values to keep.
        """
        self.full_data_df = data_df.copy()
        self.data_df = data_df[cols].copy() if cols is not None else data_df.copy()

        self.cols = self.data_df.columns.tolist()
        
        if mask is not None:
            self.data_df = self.data_df.loc[mask]      
            self.samples = jnp.array(self.data_df.to_numpy())
            self.cleaning_info= {
                "strategy": "drop",
                "mask": mask,
                "n_dropped": np.sum(~mask),
                "n_remaining": np.sum(mask),
            }
        elif handle_inf != "none":
            samples_raw = jnp.array(self.data_df.to_numpy())
            self.samples, self.cleaning_info = _handle_special_values(
                samples_raw,
                strategy=handle_inf,
                fill_value=inf_fill_value,
                clip_percentile=inf_clip_percentile,
            )
            if handle_inf == "drop" and self.cleaning_info.get("n_dropped", 0) > 0:
                print(f"Dropped {self.cleaning_info['n_dropped']} rows with inf/nan values")
                self.data_df = self.data_df.loc[self.cleaning_info["mask"]]
        else:
            self.samples = jnp.array(self.data_df.to_numpy())
            self.cleaning_info = {"strategy": "none"}

        self.N, self.dim = self.samples.shape

        # 2. Determine Column Transforms
        if transform_params is not None:
            # Case A: Load from saved params (Override)
            if transform_params['cols'] != self.cols:
                raise ValueError(
                    f"Column mismatch. Saved columns: {transform_params['cols']}, "
                    f"Current columns: {self.cols}"
                )
            
            self._labels = jnp.array(transform_params['labels'])
            # Ensure mu/L are JAX arrays
            self.mu = jnp.array(transform_params['mu'])
            self.L = jnp.array(transform_params['L'])
            self.transformations = [_label_to_name(lbl) for lbl in np.array(self._labels)]
            
            # Apply forward transform
            self.samples_transformed = _forward_transform_array(self.samples, self._labels)
            
            # Apply whitening using SAVED statistics
            self.whitened_data = _apply_whitening(self.samples_transformed, self.mu, self.L)
            
        else:
            # Case B: Compute from scratch
            if transforms:
                if "choose" in transforms:
                    raise NotImplementedError("For now you have to specify all transforms")
                lbl_map = {"none": 0, "log": 1, "angle_2pi": 2, "angle_pi": 3, "angle_signed_pi": 4}
                lbls = [lbl_map.get(tr, 0) for tr in transforms]
                self._labels = jnp.array(lbls)
            else:
                self._labels = _classify_columns(self.samples)
            
            self.transformations = [_label_to_name(lbl) for lbl in np.array(self._labels)]
            self.samples_transformed = _forward_transform_array(self.samples, self._labels)

            # Compute whitening statistics
            self.whitened_data, self.mu, self.L = _whiten(self.samples_transformed)

    # ------------- Save / Load Logic -------------

    def save(self, path: str | Path, save_data: bool = False):
        """Save transformations and whitening statistics to disk.
        
        Creates a directory at `path` containing:
        - params.json: Metadata (columns, transformation names/labels).
        - mu.npy: Whitening mean vector.
        - L.npy: Whitening Cholesky matrix.
        - labels.npy: Transformation labels.
        - data.csv: (Optional) The cleaned data frame if save_data=True.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Convert JAX arrays to numpy for serialization
        mu_np = np.array(self.mu)
        L_np = np.array(self.L)
        labels_np = np.array(self._labels)

        # Save Arrays
        np.save(path / "mu.npy", mu_np)
        np.save(path / "L.npy", L_np)
        np.save(path / "labels.npy", labels_np)
        if self.cleaning_info.get("mask", None) is not None:
            np.save(path / "mask.npy", np.array(self.cleaning_info["mask"]))

        # Save Metadata
        metadata = {
            "cols": self.cols,
            "transformations": self.transformations,
            "cleaning_info": {k: str(v) for k, v in self.cleaning_info.items()},
            "has_data": save_data
        }
        
        with open(path / "params.json", "w") as f:
            json.dump(metadata, f, indent=4)
        
        # Save Data (save the full data, which then recreates with the metadata the correct data_df including the transformations)
        if save_data:
            self.full_data_df.to_csv(path / "full_data.csv", index=False)
            print(f"Data and transformations saved to {path}")
        else:
            print(f"Transformations saved to {path}")

    @classmethod
    def from_saved(
        cls, 
        path: str | Path, 
        data_df: Optional[pd.DataFrame] = None, 
        handle_inf: str = "clip"
    ):
        """Instantiate a Data object using transformations saved on disk.

        Args:
            path: Directory containing the saved parameters.
            data_df: New data to transform. If None, tries to load 'full_data.csv' from path.
            handle_inf: Strategy for handling inf/nan in the NEW data.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")

        # Load Metadata
        with open(path / "params.json", "r") as f:
            metadata = json.load(f)

        # Handle Data Loading
        if data_df is None:
            data_path = path / "full_data.csv"
            if data_path.exists():
                data_df = pd.read_csv(data_path)
            else:
                raise ValueError(
                    f"data_df was not provided and {data_path} not found. "
                    "Cannot reconstruct Data object without data."
                )

        # Load Arrays
        mu = np.load(path / "mu.npy")
        L = np.load(path / "L.npy")
        labels = np.load(path / "labels.npy")
        try:
            mask = np.load(path / "mask.npy")
        except FileNotFoundError:
            mask = None
            print("No masking file on disk. If data was masked will try to reconstruct from metadata.")

        if mask is None:
            mask = np.array(ast.literal_eval(metadata["mask"]), dtype=bool) if metadata.get("mask", None) is not None else None

        # Construct params dictionary
        transform_params = {
            "cols": metadata["cols"],
            "labels": labels,
            "mu": mu,
            "L": L
        }

        # Instantiate (cols must match explicitly or be sliced by the constructor via 'cols')
        # We pass the saved column list to ensure the new dataframe is sliced in the same order.
        return cls(
            data_df=data_df,
            cols=np.array(metadata["cols"]),
            handle_inf=handle_inf,
            transform_params=transform_params,
            mask = mask
        )
    # ------------- basic accessors -------------

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.N

    def __getitem__(self, idx) -> jnp.ndarray:
        """Return the raw sample at the given index."""
        return self.samples[idx]

    def get_cleaning_info(self) -> dict:
        """Return information about special value handling.

        Returns:
            info: Dictionary with cleaning statistics.
        """
        return self.cleaning_info.copy()

    def get_column(self, col_name: str) -> jnp.ndarray:
        """Return a single column as a JAX array.

        Args:
            col_name: Name of the column.

        Returns:
            column: Column values, shape (N,).
        """
        return jnp.array(self.data_df[col_name].to_numpy())

    def get_column_index(self, col_name: str) -> int:
        """Return integer index of a column by name.

        Args:
            col_name: Column name.

        Returns:
            index: Zero-based column index.
        """
        return self.cols.index(col_name)

    def get_dataframe(self) -> pd.DataFrame:
        """Return a copy of the active DataFrame (possibly a column subset).

        Returns:
            df: Copy of the active DataFrame.
        """
        return self.data_df.copy()

    def get_full_dataframe(self) -> pd.DataFrame:
        """Return a copy of the original full DataFrame.

        Returns:
            df: Copy of the original full DataFrame.
        """
        return self.full_data_df.copy()

    def get_columns(self) -> list[str]:
        """Return the list of column names used in this object.

        Returns:
            cols: List of column names.
        """
        return self.cols.copy()

    def get_samples(self) -> jnp.ndarray:
        """Return raw samples as a JAX array.

        Returns:
            samples: Raw samples, shape (N, D).
        """
        return self.samples.copy()

    def get_samples_df(self) -> pd.DataFrame:
        """Return raw samples as a pandas DataFrame.

        Returns:
            df: DataFrame of raw samples, shape (N, D).
        """
        return pd.DataFrame(np.array(self.samples), columns=self.cols)

    def get_samples_transformed(self) -> jnp.ndarray:
        """Return forward-transformed samples as a JAX array.

        Returns:
            samples_t: Transformed samples, shape (N, D).
        """
        return self.samples_transformed.copy()

    def get_samples_transformed_df(self) -> pd.DataFrame:
        """Return forward-transformed samples as a pandas DataFrame.

        Returns:
            df: DataFrame of transformed samples, shape (N, D).
        """
        return pd.DataFrame(np.array(self.samples_transformed), columns=self.cols)

    def get_shape(self) -> tuple[int, int]:
        """Return the shape of the underlying samples.

        Returns:
            shape: Tuple (N, D).
        """
        return self.samples.shape

    # ------------- transforms -------------

    def forward_transform(
        self,
        samples: jnp.ndarray | np.ndarray | None = None,
    ) -> jnp.ndarray:
        """Apply the learned column-wise forward transform to samples.

        Args:
            samples: Optional samples in original space, shape (N, D).
                If None, uses the internally stored raw samples.

        Returns:
            transformed: Transformed samples, shape (N, D).
        """
        if samples is None:
            arr = self.samples
        else:
            arr = jnp.array(samples)
        return _forward_transform_array(arr, self._labels)

    def inverse_transform(self, samples: jnp.ndarray | np.ndarray) -> jnp.ndarray:
        """Invert the column-wise transform.

        Args:
            samples: Transformed samples, shape (N, D).

        Returns:
            recovered: Recovered samples in original space, shape (N, D).
        """
        arr = jnp.array(samples)
        return _inverse_transform_array(arr, self._labels)

    # ------------- whitening -------------

    def whiten_data(
        self,
        samples: jnp.ndarray | np.ndarray | None = None,
        transform: bool = True
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Whiten (zero-mean, unit-covariance) the forward-transformed data.

        If samples are provided, they are first forward-transformed using the
        existing column transform rules, then whitened.

        Args:
            samples: Optional samples in original space, shape (N, D).
                If None, uses the stored transformed samples.

        Returns:
            x_white: Whitened samples, shape (N, D).
            mu: Mean in transformed space, shape (D,).
            L: Cholesky factor of covariance, shape (D, D).
        """
        if samples is None:
            x_t = self.samples_transformed if transform else self.samples
        else:
            x_t = self.forward_transform(samples) if transform else samples

        x_white, mu, L = _whiten(x_t)
        self.whitened_data, self.mu, self.L = x_white, mu, L
        return x_white, mu, L

    def inverse_whiten_data(self, x_white: jnp.ndarray | np.ndarray, inv_transform: bool = True) -> jnp.ndarray:
        """Invert whitening and then invert the column-wise transform.

        Args:
            x_white: Whitened samples, shape (N, D).

        Returns:
            x_rec: Samples recovered in original data space, shape (N, D).
        """
        x_w = jnp.array(x_white)
        x_t_rec = _inverse_whiten(x_w, self.mu, self.L)
        x_rec = self.inverse_transform(x_t_rec) if inv_transform else x_t_rec
        return x_rec


# -----------------------------
# Dataset class for supervised learning
# -----------------------------

class Dataset:
    """Supervised learning dataset with X (features) and y (targets).

    This class wraps Data objects for features and targets, providing
    convenient methods for batching, splitting, and integration with JAX-based
    machine learning workflows (e.g., FlowJAX, normalizing flows).
    """

    def __init__(
        self,
        X: Data | jnp.ndarray | np.ndarray,
        y: Data | jnp.ndarray | np.ndarray | None = None,
        use_whitened_X: bool = False,
        use_whitened_y: bool = False,
        train_indices: list[int] | None = None,
        val_indices: list[int] | None = None,
    ):
        """Initialize a supervised dataset.

        Args:
            X: Feature data. Can be a Data object or raw array of shape (N, D_x).
            y: Optional target data. Can be a Data object or raw array of shape 
                (N,) or (N, D_y). If None, creates an unsupervised dataset 
                (useful for density estimation).
            use_whitened_X: If True and X is a Data object, use whitened features.
            use_whitened_y: If True and y is a Data object, use whitened targets.
        """
        # Handle X data
        if isinstance(X, Data):
            self.X_data = X
            if use_whitened_X:
                self.X = X.whitened_data
            else:
                self.X = X.samples_transformed
        else:
            self.X_data = None
            self.X = jnp.array(X)

        # Handle y data
        if y is None:
            self.y_data = None
            self.y = None
        elif isinstance(y, Data):
            self.y_data = y
            if use_whitened_y:
                self.y = y.whitened_data
            else:
                self.y = y.samples_transformed
        else:
            self.y_data = None
            self.y = jnp.array(y)

        self.N = self.X.shape[0]
        self.dim_x = self.X.shape[1]
        if self.y_data:
            self.dim_y = self.y.shape[1]
        else:
            self.dim_y = 0
        self.use_whitened_X = use_whitened_X
        self.use_whitened_y = use_whitened_y

        if self.y is not None and len(self.y) != self.N:
            raise ValueError(
                f"X and y must have same length. Got X: {self.N}, y: {len(self.y)}"
            )

        self.train_indices = train_indices
        self.val_indices = val_indices

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.N

    def __getitem__(self, idx) -> tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray:
        """Return sample(s) at the given index.

        Args:
            idx: Integer index or slice.

        Returns:
            sample: If supervised, returns (X[idx], y[idx]). Otherwise X[idx].
        """
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

    def save(self, path: str | Path, save_data: bool = False):
        """Save dataset structure, transforms, and optionally the data itself.

        Args:
            path: Target directory.
            save_data: If True, saves the underlying data (csv) inside subfolders.
                       Only works if the dataset was constructed from Data objects.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config = {
            "use_whitened_X": self.use_whitened_X,
            "use_whitened_y": self.use_whitened_y,
            "has_y": self.y is not None,
            "is_X_Data_obj": self.X_data is not None,
            "is_y_Data_obj": self.y_data is not None,
            "train_indices": self.train_indices.tolist() if self.train_indices is not None else None,
            "val_indices": self.val_indices.tolist() if self.val_indices is not None else None,
        }

        with open(path / "dataset_info.json", "w") as f:
            json.dump(config, f, indent=4)

        # Save X
        if self.X_data is not None:
            self.X_data.save(path / "X", save_data=save_data)
        elif save_data:
            # If it's a raw array, we just save the array
            np.save(path / "X_raw.npy", np.array(self.X))
            print("Saved raw X array (Dataset was not initialized with Data object).")

        # Save y
        if self.y is not None:
            if self.y_data is not None:
                self.y_data.save(path / "y", save_data=save_data)
            elif save_data:
                np.save(path / "y_raw.npy", np.array(self.y))
                print("Saved raw y array.")
        
        print(f"Dataset configuration saved to {path}")

    @classmethod
    def from_saved(cls, path: str | Path):
        """Reconstruct Dataset from saved files.
        
        Requires that the Data objects were saved with save_data=True, OR
        that the Dataset was saved with raw arrays.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")
        
        with open(path / "dataset_info.json", "r") as f:
            config = json.load(f)

        # Load X
        if config["is_X_Data_obj"]:
            # Data.from_saved will look for data.csv inside path/X/
            X = Data.from_saved(path / "X")
        else:
            # Try to load raw array
            if (path / "X_raw.npy").exists():
                X = jnp.array(np.load(path / "X_raw.npy"))
            else:
                raise ValueError("Saved Dataset contains no X data (Data object or raw .npy missing).")

        # Load y
        y = None
        if config["has_y"]:
            if config["is_y_Data_obj"]:
                y = Data.from_saved(path / "y")
            else:
                if (path / "y_raw.npy").exists():
                    y = jnp.array(np.load(path / "y_raw.npy"))
                else:
                    raise ValueError("Saved Dataset indicates y exists, but data not found.")

        return cls(
            X=X, 
            y=y, 
            use_whitened_X=config["use_whitened_X"], 
            use_whitened_y=config["use_whitened_y"],
            train_indices=config["train_indices"],
            val_indices=config["val_indices"],
        )

    def get_features(self) -> jnp.ndarray:
        """Return all features.

        Returns:
            X: Feature array, shape (N, D_x).
        """
        return self.X

    def get_targets(self) -> jnp.ndarray | None:
        """Return all targets.

        Returns:
            y: Target array, shape (N,) or (N, D_y), or None if unsupervised.
        """
        return self.y

    def split(
        self,
        key: jax.random.PRNGKey,
        train_frac: float = 0.8,
    ) -> tuple["Dataset", "Dataset", jnp.ndarray, jnp.ndarray]:
        """Randomly split dataset into train and validation sets.

        Note: These datasets will not have any knowledge of the original Data objects (i.e., no transforms or whitening or the original full_data_df).
        However, this current dataset will know about how these split datasets were created, so you can always start from here.
        
        Args:
            key: JAX random key for shuffling.
            train_frac: Fraction of data to use for training (0 < train_frac < 1).

        Returns:
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
        """
        if not 0 < train_frac < 1:
            raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")

        indices = jr.permutation(key, self.N)
        n_train = int(self.N * train_frac)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        X_train = self.X[train_idx]
        X_val = self.X[val_idx]

        if self.y is not None:
            y_train = self.y[train_idx]
            y_val = self.y[val_idx]
        else:
            y_train = None
            y_val = None

        train_dataset = Dataset(X_train, y_train, 
                               use_whitened_X=False, use_whitened_y=False)
        val_dataset = Dataset(X_val, y_val, 
                             use_whitened_X=False, use_whitened_y=False)

        # Preserve Data object references if they exist
        train_dataset.X_data = self.X_data
        val_dataset.X_data = self.X_data
        train_dataset.y_data = self.y_data
        val_dataset.y_data = self.y_data

        self.train_indices = train_idx
        self.val_indices = val_idx

        return train_dataset, val_dataset, train_idx, val_idx

    def get_split_datasets(self,) -> tuple["Dataset" | None, "Dataset" | None]:
        """Return train and validation datasets based on stored indices.

        Note: These datasets will not have any knowledge of the original Data objects (i.e., no transforms or whitening or the original full_data_df).
        However, this current dataset will know about how these split datasets were created, so you can always start from here.

        Returns:
            train_dataset: Training dataset or None if no train_indices.
            val_dataset: Validation dataset or None if no val_indices.
        """
        train_dataset = None
        val_dataset = None

        if self.train_indices is not None:
            train_slice = np.s_[self.train_indices, :]
            X_train = self.X[train_slice]
            if self.y is not None:
                y_train = self.y[train_slice]
            else:
                y_train = None
            train_dataset = Dataset(X_train, y_train, 
                                   use_whitened_X=False, use_whitened_y=False)
            train_dataset.X_data = self.X_data
            train_dataset.y_data = self.y_data

        if self.val_indices is not None:
            val_slice = np.s_[self.val_indices, :]
            X_val = self.X[val_slice]
            if self.y is not None:
                y_val = self.y[val_slice]
            else:
                y_val = None
            val_dataset = Dataset(X_val, y_val, 
                                 use_whitened_X=False, use_whitened_y=False)
            val_dataset.X_data = self.X_data
            val_dataset.y_data = self.y_data

        return train_dataset, val_dataset

    def batch_iterator(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
        shuffle: bool = True,
    ):
        """Create a batch iterator for training.

        Args:
            key: JAX random key for shuffling.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle data before batching.

        Yields:
            batch: Dictionary with 'X' and optionally 'y' keys containing batches.
        """
        indices = jnp.arange(self.N)

        if shuffle:
            indices = jr.permutation(key, indices)

        n_batches = self.N // batch_size

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_idx = indices[start_idx:end_idx]

            batch = {"X": self.X[batch_idx]}
            if self.y is not None:
                batch["y"] = self.y[batch_idx]

            yield batch

    def to_original_space_X(self, X_transformed: jnp.ndarray) -> jnp.ndarray:
        """Transform feature samples back to original data space.

        Args:
            X_transformed: Samples in transformed/whitened space, shape (N, D_x).

        Returns:
            X_original: Samples in original space, shape (N, D_x).

        Raises:
            ValueError: If the dataset was not created from a Data object.
        """
        if self.X_data is None:
            raise ValueError(
                "Cannot transform X to original space: X was not created from Data object"
            )

        if self.use_whitened_X:
            return self.X_data.inverse_whiten_data(X_transformed)
        else:
            return self.X_data.inverse_transform(X_transformed)

    def to_original_space_y(self, y_transformed: jnp.ndarray) -> jnp.ndarray:
        """Transform target samples back to original data space.

        Args:
            y_transformed: Targets in transformed/whitened space, shape (N,) or (N, D_y).

        Returns:
            y_original: Targets in original space, shape (N,) or (N, D_y).

        Raises:
            ValueError: If the dataset was not created with y as a Data object.
        """
        if self.y_data is None:
            raise ValueError(
                "Cannot transform y to original space: y was not created from Data object"
            )

        if self.use_whitened_y:
            return self.y_data.inverse_whiten_data(y_transformed)
        else:
            return self.y_data.inverse_transform(y_transformed)

    def to_original_space(
        self, 
        X_transformed: jnp.ndarray, 
        y_transformed: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Transform both features and targets back to original space.

        Args:
            X_transformed: Features in transformed/whitened space, shape (N, D_x).
            y_transformed: Optional targets in transformed/whitened space, 
                shape (N,) or (N, D_y).

        Returns:
            X_original: Features in original space, shape (N, D_x).
            y_original: Targets in original space if y_transformed provided, 
                otherwise None.
        """
        X_original = self.to_original_space_X(X_transformed)
        
        if y_transformed is not None:
            y_original = self.to_original_space_y(y_transformed)
            return X_original, y_original
        
        return X_original, None


# -----------------------------
# Flowjax learning and flow functions
# -----------------------------

def create_flow_from_config(config: dict):
    """Create flow from config.

    Args:
        config: dict containing e.g. key, data_dim and flow parameters

    Returns:
        (untrained) flow.
    """

    if config["base_dist"] == "Normal":
        base_dist = Normal(jnp.zeros(config["data_dim"]), jnp.ones(config["data_dim"]))
    elif config["base_dist"] == "StudentT":
        dof = config.get("dof", 5) # standard value of 5 degrees of freedom
        base_dist = StudentT(df = jnp.full((config["data_dim"]), dof))
    else:
        warnings.warn("Unknown base distribution in config; only 'Normal' available.")

    key = jax.random.key(config["key"])
    cond_dim = config.get("cond_dim")
    if config["type"] == "MAF":
        flow = masked_autoregressive_flow(
            key=key,
            base_dist=base_dist,
            flow_layers=config["flow_layers"],
            nn_width=config["nn_width"],
            nn_depth=config["nn_depth"],
            cond_dim = config.get("cond_dim")
        )
    elif config["type"] == "spline":
        flow = triangular_spline_flow(
            key=key,
            base_dist=base_dist,
            flow_layers=config["flow_layers"],
            knots=config["knots"],
        )

    return flow

def train_flow(flow: Transformed, train_dataset: Dataset, val_dataset: Dataset, train_weights: jnp.array |None = None, val_weights: jnp.array |None = None, epochs: int =1000, patience: int =100, batch_size: int =2048, learning_rate: float = 1e-3, noise: bool = False) -> Tuple[Transformed, dict]:
    """Train method for flow.

    Args:
        flow: initial flow to be trained.
        train_dataset: Dataset containing the training data.
        val_dataset: Dataset containing the validation data to prevent overfitting.
        epochs: number of epochs.
        patience: how many epochs to keep training even when val loss does not improve.
        batch_size: size of batches to be used in training. Too small will hit CPU bottleneck.
        learning_rate: learning rate of adam optimizer.

    Returns:
        trained flow.
        dictionary containing the training loss information.
    
    """
    
    has_cond = hasattr(train_dataset, 'y') and train_dataset.y is not None
    
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(flow, eqx.is_array))

    @eqx.filter_value_and_grad
    def loss_fn(flow, x, condition, weights=None):
        logp = flow.log_prob(x, condition=condition)
    
        if weights is None:
            return -jnp.mean(logp)
        else:
            # optional stabilization
            # weights = weights / (jnp.mean(weights) + 1e-12)
            return -(jnp.sum(weights * logp) / jnp.sum(weights))

    @eqx.filter_jit
    def train_step(flow, opt_state, x, condition, weights=None):
        loss, grads = loss_fn(flow, x, condition, weights)
        updates, opt_state = optimizer.update(grads, opt_state)
        flow = eqx.apply_updates(flow, updates)
        return flow, opt_state, loss

    @eqx.filter_jit
    def eval_step(flow, x, condition, weights=None):
        logp = flow.log_prob(x, condition=condition)
    
        if weights is None:
            return -jnp.mean(logp)
        else:
            # optional stabilization
            # weights = weights / (jnp.mean(weights) + 1e-12)
            return -(jnp.sum(weights * logp) / jnp.sum(weights))
            

    train_losses, val_losses = [], []
    N_train, N_val = len(train_dataset), len(val_dataset)
    best_val_loss, best_epoch, best_flow = np.inf, 0, flow
    
    epoch_pbar = trange(1, epochs + 1, desc="Training")

    use_weights = False
    if train_weights is not None and val_weights is not None: 
        if train_weights.max() > 1e2 or train_weights.min() < 1e-2:
            warnings.warn(f"Training weights span large range; Consider normalizing in loss function; code is there, only need to comment out.")
        if val_weights.max() > 1e2 or val_weights.min() < 1e-2:
            warnings.warn(f"Validation weights span large range; Consider normalizing in loss function; code is there, only need to comment out.")

        use_weights = True

    for epoch in epoch_pbar:
        # Shuffle indices
        perm = np.random.permutation(N_train)
        perm_v = np.random.permutation(N_val)
        
        x_train = train_dataset.X[perm]
        y_train = train_dataset.y[perm] if has_cond else None
        
        x_val = val_dataset.X[perm_v]
        y_val = val_dataset.y[perm_v] if has_cond else None
        
        # training
        batch_losses = []
        for i in range(0, N_train - batch_size + 1, batch_size):
            bx = x_train[i : i + batch_size]
            by = y_train[i : i + batch_size] if has_cond else None
            
            bw = (
                train_weights[perm][i : i + batch_size]
                if use_weights else None
            )

            # add some noise for regularization
            if noise:
                key = jax.random.key(i*(epoch+1))
                bx = bx + jax.random.normal(key, shape = bx.shape) * 1e-1 # add small amount of noise
        
            flow, opt_state, loss = train_step(flow, opt_state, bx, by, bw)
            batch_losses.append(float(loss))
            
        curr_train_loss = float(jnp.mean(jnp.stack(batch_losses)))
        train_losses.append(curr_train_loss)
    
        # validation
        val_batch_losses = []
        for i in range(0, N_val - batch_size + 1, batch_size):
            vbx = x_val[i : i + batch_size]
            vby = y_val[i : i + batch_size] if has_cond else None
            
            vw = (
                val_weights[perm_v][i : i + batch_size]
                if use_weights else None
            )
        
            val_loss = eval_step(flow, vbx, vby, vw)
            val_batch_losses.append(val_loss)

        curr_val_loss = float(jnp.mean(jnp.stack(val_batch_losses))) if val_batch_losses else np.inf 
        val_losses.append(curr_val_loss)
    
        # early stopping, monitoring
        if curr_val_loss < best_val_loss:
            best_val_loss, best_epoch, best_flow = curr_val_loss, epoch, flow
        
        epoch_pbar.set_postfix(
            train=f"{curr_train_loss:.4f}",
            val=f"{curr_val_loss:.4f}",
            best_val=f"{best_val_loss:.4f}",
            patience=f"{patience - (epoch - best_epoch)}",
        )
        
        if epoch - best_epoch > patience or np.isnan(curr_train_loss):
            break

        if np.isnan(curr_train_loss) or np.isinf(curr_train_loss):
            print(f"\nEarly stopping at epoch {epoch}: Divergence detected.")
            print(f"Train loss {curr_train_loss}")
            break

    meta = {
        "train_losses": train_losses, 
        "val_losses": val_losses, 
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }

    return best_flow, meta

def save_flow(filename: str |Path, flow: Transformed, config: dict) -> None:
    """Save flow to disk.

    Args:
        filename: where to store the flow.
        flow: jax flow (Transform).
        config: dictionary holding all necessary information to reconstruct flow.

    """
    
    # 1. Separate the model into 'arrays' and 'non-arrays' (functions, etc.)
    # We only care about saving the 'arrays'
    arrays, static = eqx.partition(flow, eqx.is_array)
    
    # 2. Flatten only the arrays into a list
    leaves, _ = jtu.tree_flatten(arrays)
    
    # 3. Save to npz. We tell numpy NOT to try pickling anything (allow_pickle=False)
    # this ensures we are only saving pure numerical data.
    np.savez(filename, 
             config_json=json.dumps(config), 
             *leaves)

def load_flow(filename: str |Path, constructor_func) -> Tuple[Transformed, dict]:
    """Load flow from disk.

    Args:
        filename: where to find flow.
        constructor_func: function that specifies how to construct flow (see above).

    Returns:
        flow
        metadata dict containing the config of the flow (layers, ...)
    """

    with np.load(filename) as data:
        config = json.loads(str(data['config_json']))
        
        # 1. Rebuild the skeleton (this creates the functions/static parts)
        skeleton = constructor_func(config)
        
        # 2. Partition the skeleton
        arrays, static = eqx.partition(skeleton, eqx.is_array)
        
        # 3. Get the tree structure of the array part
        array_leaves, array_treedef = jtu.tree_flatten(arrays)
        
        # 4. Load the arrays from disk (skipping the config_json key)
        loaded_leaves = [data[k] for k in data.files if k != 'config_json']
        
        # 5. Reconstruct the array-tree and combine with the static-tree
        new_arrays = jtu.tree_unflatten(array_treedef, loaded_leaves)
        loaded_flow = eqx.combine(new_arrays, static)
        
    return loaded_flow, config

def evaluate_marginal_ks_test(x_true: np.ndarray, x_gen: np.ndarray, downsample: int = 1, p_threshold: float = 0.02) -> list[[float, float, bool]]:
    """Evaluate the ks test for the data.

    Args:
        x_true: ground truth data.
        x_gen: generated (test) data.
        downsample: downsampling factor (just selects ::n).
        p_threshold: credible interval for rejection (reject H0 if smaller than p_threshold).

    Returns:
        test results containing (D, p_value, same distribution or not given p_threshold)
    """

    x_true, x_gen = np.array(x_true), np.array(x_gen)
    statistics = []
    assert x_true.shape == x_gen.shape, "Test and Truth array have different shapes."
    for i in range(x_true.shape[1]):
        stat, p_value = stats.ks_2samp(x_true[::downsample, i], x_gen[::downsample, i])
        statistics.append((stat, p_value, bool(p_value > p_threshold)))

    return statistics

def evaluate_summary_statistics(x_true: np.ndarray |pd.DataFrame, x_gen: np.ndarray |pd.DataFrame, verbose: bool = True) -> dict[pd.DataFrame]:
    """Evaluate the pandas built-in summary statistics.

    Args:
        x_true: ground truth data.
        x_gen: generated (test) data.
        verbose: whether to print the output.

    Returns:
        dictionary containing true and generated statistics (summaries and correlation matrices).
    """

    df_true = pd.DataFrame(x_true)
    df_gen = pd.DataFrame(x_gen)
    
    summary_statistics = {
        "true_ss": df_true.describe(),
        "gen_ss": df_gen.describe(),
        "true_corr": df_true.corr(),
        "gen_corr": df_gen.corr(),
    }
    
    if verbose:
        for key, value in summary_statistics.items():
            print(key)
            print(value)
    
    return summary_statistics

