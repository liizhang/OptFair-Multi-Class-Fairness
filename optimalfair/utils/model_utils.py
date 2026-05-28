import torch
import numpy as np
import torch.nn.functional as F
import copy
import random
import sklearn.metrics
import os
from typing import Optional, Union

# data_info: {'train_num':train_num, 'test_num':test_num, 'Ylabel':Ylabel, 'Alabel':Alabel, 'A_num':A_num, 'Y_num':Y_num}

def mkdir(*args: str) -> tuple:
    for path in args:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    return args

def error_rate(y_true, y_preds, groups=None, w=None, n_groups=None):
  """Compute group-weighted error rate."""
  if groups is None or w is None:
    return np.mean(y_true != y_preds)
  else:
    if n_groups is None:
      group_names, groups = np.unique(groups, return_inverse=True)
      n_groups = len(group_names)
    return sum([
        w[a] * np.mean(y_true[groups == a] != y_preds[groups == a])
        for a in range(n_groups)
    ])


def delta_sp(y_preds, groups, n_classes, n_groups, ord=np.inf):
  """Compute violation of statistical parity."""
  y_preds = y_preds.astype(np.int64)
  pred_counts = np.array([
      np.bincount(y_preds[groups == a], minlength=n_classes)
      for a in range(n_groups)
  ])
  output_dists = pred_counts / np.sum(pred_counts, axis=1, keepdims=True)
  diffs = np.linalg.norm(output_dists[:, None, :] - output_dists[None, :, :],
                         ord=ord,
                         axis=2)
  return np.max(diffs)


def confusion_matrix(y_true, y_preds, groups, n_classes, n_groups, normalize='true'):
  """Compute group-wise confusion matrices (conditioned on y_true)."""
  return np.array([
      sklearn.metrics.confusion_matrix(y_true[groups == a],
                                       y_preds[groups == a],
                                       labels=np.arange(n_classes),
                                       normalize=normalize)
      for a in range(n_groups)
  ])


def delta_eo(y_true, y_preds, groups, n_classes, n_groups, ord=np.inf):
  """Compute violation of equalized odds."""
  conf_mtxs = confusion_matrix(
      y_true,
      y_preds,
      groups,
      n_classes,
      n_groups,
  ).reshape(n_groups, -1)  # shape = (n_groups, n_classes**2)
  with np.errstate(invalid='ignore'):  # Ignore groups with no positive examples
    # Pairwise differences
    diffs = np.linalg.norm(conf_mtxs[:, None, :] - conf_mtxs[None, :, :],
                           ord=ord,
                           axis=2)
    diffs = np.nan_to_num(diffs, nan=0.0)
  return np.max(diffs)


def delta_eopp(y_true, y_preds, groups, n_classes, n_groups, ord=np.inf):
  """
  Compute violation of (binary or multi-class) equalized opportunity (depending
  on `n_classes`).
  """
  conf_mtxs = confusion_matrix(
      y_true, y_preds, groups, n_classes,
      n_groups)  # shape = (n_groups, n_classes, n_classes)
  tprs = np.array([np.diag(conf_mtx) for conf_mtx in conf_mtxs
                  ])  # shape = (n_groups, n_classes)
  if n_classes == 2:
    tprs = tprs[:, 1].reshape(-1, 1)  # shape = (n_groups, 1)
  with np.errstate(invalid='ignore'):  # Ignore groups with no positive examples
    # Pairwise differences
    diffs = np.linalg.norm(tprs[:, None, :] - tprs[None, :, :], ord=ord, axis=2)
    diffs = np.nan_to_num(diffs, nan=0.0)
  return np.max(diffs)


# def calibration_error(probas, labels, n_bins=100, seed=0):
#   """Computes binned expected calibration error, with bins selected by k-means.
#   """
#   calib = BinningCalibrator(n_bins=n_bins,
#                             random_state=seed).fit(probas, labels)
#   # bins = calib.binning_fn_(probas)
#   # bin_to_proba = {b: probas[bins == b].mean(axis=0) for b in np.unique(bins)}
#   # probas_binned = np.array([bin_to_proba[b] for b in bins])
#   p = np.mean(probas, axis=0)
#   probas_cal = calib.predict_proba(probas)
#   p_cal = np.mean(probas_cal, axis=0)
#   return np.max(np.mean(np.abs(probas / p - probas_cal / p_cal), axis=0))


# Define some utility functions

def l1_projection_vectorized(x, B: float):
    """
    Project x onto the L1 ball {v: ||v||_1 <= B}, treating x as a vector (derived by mirror descent).
    Returns a tensor with the same shape as x.
    """
    # Flatten to 1D
    x_flat = x.view(-1)
    if B < 0:
        raise ValueError("B must be non-negative.")
    if B == 0:
        return torch.zeros_like(x)

    abs_x = x_flat.abs()
    s = abs_x.sum()
    if s <= B:  # already feasible
        return x.clone()

    # Sort |x| in descending order
    u, _ = torch.sort(abs_x, descending=True)
    cssv = torch.cumsum(u, dim=0) - B
    k = torch.arange(1, u.numel() + 1, device=x.device, dtype=x.dtype)

    # Duchi condition: find rho = max { k : u_k > cssv_k / k }
    cond = u > (cssv / k)
    idx = torch.where(cond)[0]
    rho = idx[-1].item()

    theta = cssv[rho] / (rho + 1.0)

    # Soft-threshold and reshape back
    w = torch.sign(x)* torch.clamp(torch.abs(x) - theta, min=0.0)
    return w

import torch

import torch

def solve_l1_constrained_quadratic(
    tau: torch.Tensor,
    t: torch.Tensor,
    B: float,
    xi: float,
    sigma: float = 1.0,
):
    """
    Solve:
        max_{||lambda||_1 <= B} lambda^T tau - xi||lambda||_1 - (1/(2*sigma))||lambda - t||_2^2

    Parameters
    ----------
    tau : torch.Tensor, any shape
    t   : torch.Tensor, same shape as tau
    B   : float, L1 budget
    xi  : float, L1 penalty
    sigma : float, quadratic parameter

    Returns
    -------
    lambda_star : torch.Tensor, same shape as tau
    """

    # Step 1: form v
    v = t + sigma * tau

    # Step 2: unconstrained soft-threshold
    thresh0 = sigma * xi
    lambda0 = torch.sign(v) * torch.maximum(torch.abs(v) - thresh0, torch.tensor(0.0, device=v.device))

    # Step 3: check L1 constraint
    l1_norm = torch.sum(torch.abs(lambda0))  # Sum over all elements in the tensor
    
    # If L1 norm is within budget, return the solution directly
    if l1_norm <= B:
        return lambda0

    # Step 4: L1 constraint active; find theta
    abs_v = torch.abs(v).flatten()  # Flatten to a 1D tensor
    u, _ = torch.sort(abs_v, descending=True)  # Sort descending for the whole vector
    cumsum_u = torch.cumsum(u, dim=0)  # Cumulative sum of sorted values

    # Find theta such that sum(max(|v|-theta,0)) = B
    rho = -1
    theta = 0.0
    for j in range(len(u)):
        tmp = (cumsum_u[j] - B) / (j + 1)
        if u[j] > tmp:
            rho = j
            theta = tmp
            break

    # Final solution
    lambda_star = torch.sign(v) * torch.maximum(torch.abs(v) - theta, torch.tensor(0.0, device=v.device))
    
    return lambda_star


def _soft_threshold(v: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    # Element-wise soft threshold: sign(v) * max(|v|-tau, 0)
    return v.sign() * torch.relu(v.abs() - tau)

import torch


def prox_l1_plus_l1ball(v: torch.Tensor, eta: float, xi: float, B: float) -> torch.Tensor:
    """
    Compute prox_{eta*(xi*||x||_1 + I_{||x||_1 <= B})}(v), i.e.
        argmin_{||x||_1 <= B} 0.5 * ||x - v||_2^2 + eta * xi * ||x||_1

    Notes
    -----
    Implementation follows:
      1) Soft-thresholding by eta*xi
      2) If the L1-ball constraint is violated, apply an additional global shrinkage
         (equivalently, project the nonnegative magnitudes onto the simplex of radius B).
    """
    if eta <= 0:
        raise ValueError("eta must be > 0")
    if xi < 0:
        raise ValueError("xi must be >= 0")
    if B < 0:
        raise ValueError("B must be >= 0")

    # If the L1-ball radius is zero, the only feasible point is the origin
    if B == 0:
        return torch.zeros_like(v)

    # Flatten to 1D for a single global projection, then reshape back
    v_flat = v.reshape(-1)

    # Step 1: soft-thresholding for the L1 penalty term
    # u = max(|v| - eta*xi, 0)
    u = torch.clamp(v_flat.abs() - (eta * xi), min=0)

    # If already feasible w.r.t. the L1-ball, we are done
    if u.sum() <= B:
        return (v_flat.sign() * u).reshape_as(v)

    # Step 2: enforce ||x||_1 <= B by finding theta such that sum(max(u - theta, 0)) = B
    # Sort u in descending order
    s, _ = torch.sort(u, descending=True)

    # Cumulative sums of sorted entries
    cssv = torch.cumsum(s, dim=0)

    # Indices 1..d (use the same dtype/device for numerical consistency)
    j = torch.arange(1, s.numel() + 1, device=v.device, dtype=v.dtype)

    # Find the largest rho satisfying: s_rho - (cssv_rho - B)/rho > 0
    cond = s - (cssv - B) / j > 0

    # cond should have at least one True when sum(u) > B and B >= 0
    # rho = torch.nonzero(cond, as_tuple=False)[-1].item()
    idx = torch.nonzero(cond, as_tuple=False).flatten()
    if idx.numel() == 0:
        return torch.zeros_like(v)  
    rho = idx[-1].item()

    # Compute the threshold theta
    theta = (cssv[rho] - B) / (rho + 1.0)

    # Apply the additional shrinkage and restore original signs
    w = torch.clamp(u - theta, min=0)
    x_flat = v_flat.sign() * w

    # Reshape back to the original shape
    return x_flat.reshape_as(v)

def check(name, t):
    print(name, t.shape, t.dtype,
          "nan", torch.isnan(t).any().item(),
          "inf", torch.isinf(t).any().item(),
          "neg", (t < 0).any().item(),
          "min", t.min().item(),
          "max", t.max().item())
    
def concat_xy_onehot_numpy_to_torch(X: np.ndarray, Y: np.ndarray, m: int, dtype=torch.float32):
    X = np.asarray(X)
    Y = np.asarray(Y).reshape(-1)              # -> (n,)

    # 1) handle NaN / invalid
    if not np.issubdtype(Y.dtype, np.integer):
        # if Y may be float but represents classes, require it to be finite and integer-valued
        if np.issubdtype(Y.dtype, np.floating):
            if not np.all(np.isfinite(Y)):
                bad = np.where(~np.isfinite(Y))[0][:10]
                raise ValueError(f"Y contains NaN/Inf at indices {bad.tolist()} (showing up to 10).")
            if not np.all(np.equal(Y, np.round(Y))):
                bad = np.where(~np.equal(Y, np.round(Y)))[0][:10]
                raise ValueError(f"Y contains non-integer floats at indices {bad.tolist()} (showing up to 10).")
        Y = Y.astype(np.int64, copy=False)
    else:
        Y = Y.astype(np.int64, copy=False)

    # one-hot: (n, m)
    Y_onehot = np.eye(m, dtype=X.dtype)[Y]   

    # concat: (n, d+m)
    XY = np.concatenate([X, Y_onehot], axis=1)

    # to torch tensor
    XY_t = torch.as_tensor(XY, dtype=dtype)
    return XY_t

import os, json, time
from typing import Any, Dict, Optional

def _to_jsonable(x: Any):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().item() if x.numel() == 1 else x.detach().cpu().tolist()
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
    except Exception:
        pass
    return x

class JSONLStepLogger:
    def __init__(self, run_dir: str, config: Optional[Dict[str, Any]] = None,
                 filename: str = "metrics.jsonl", flush_every: int = 1):
        os.makedirs(run_dir, exist_ok=True)
        self.metrics_path = os.path.join(run_dir, filename)
        self.flush_every = max(1, int(flush_every))
        self._f = open(self.metrics_path, "a", encoding="utf-8")
        self._cnt = 0

        if config is not None:
            with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as cf:
                json.dump(config, cf, ensure_ascii=False, indent=2)

    def log_step(self, round, metrics: Dict[str, Any], **extra: Any):
        record = {
            "round": round,
            "time": int(time.time()),
            **{k: _to_jsonable(v) for k, v in metrics.items()},
            **{k: _to_jsonable(v) for k, v in extra.items()},
        }
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._cnt += 1
        if self._cnt % self.flush_every == 0:
            self._f.flush()

    def close(self):
        try:
            self._f.flush()
        finally:
            self._f.close()

def make_run_dir(options) -> str:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join('log', options['data'].lower(), options['algorithm'].lower(), options['fairness_metric'].lower() + '-' + str(options['fairness_bound']) + '-' + options['model'] + '-' + run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
