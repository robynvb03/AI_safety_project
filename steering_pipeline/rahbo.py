import torch
import math
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import AnalyticAcquisitionFunction

# You provide: eval_one(x: (d,))
EvalOneFn = Callable[[torch.Tensor], float]


@dataclass
class RAHBOConfig:
    alpha: float = 1.0
    beta_f: float = 2.0
    beta_var: float = 2.0
    k: int = 5
    lambda_reg: float = 1e-6
    num_restarts: int = 10
    raw_samples: int = 256


class RiskAverseUCB(AnalyticAcquisitionFunction):
    # acq(x) = UCB_f(x) - alpha * LCB_var(x) where var model is on log(var).

    def __init__(self, model_f, model_logvar, alpha: float, beta_f: float, beta_var: float):
        super().__init__(model=model_f)
        self.model_f = model_f
        self.model_logvar = model_logvar
        self.alpha = float(alpha)
        self.beta_f = float(beta_f)
        self.beta_var = float(beta_var)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (batch, q, d), we assume q=1
        X_ = X.squeeze(-2)  # (batch, d)

        # Mean model UCB
        post_f = self.model_f.posterior(X_)
        mu_f = post_f.mean.squeeze(-1)
        sig_f = post_f.variance.clamp_min(1e-12).sqrt().squeeze(-1)
        ucb_f = mu_f + self.beta_f * sig_f

        # Variance model LCB (in variance space)
        post_lv = self.model_logvar.posterior(X_)
        mu_lv = post_lv.mean.squeeze(-1)
        sig_lv = post_lv.variance.clamp_min(1e-12).sqrt().squeeze(-1)
        lcb_logvar = mu_lv - self.beta_var * sig_lv
        lcb_var = torch.exp(lcb_logvar).clamp_min(1e-12)

        return ucb_f - self.alpha * lcb_var


def ensure_2d(Y: torch.Tensor) -> torch.Tensor:
    # makes sure Y is (n,1) not (n,) or scalar
    if Y.dim() == 0:
        return Y.view(1, 1)
    if Y.dim() == 1:
        return Y.unsqueeze(-1)
    return Y


def evaluate_k(eval_one: EvalOneFn, x: torch.Tensor, k: int, dtype, device) -> torch.Tensor:
    ys = []
    for _ in range(k):
        y = eval_one(x)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=dtype, device=device)
        else:
            y = y.to(dtype=dtype, device=device)

        y = y.reshape(1, 1)  
        ys.append(y)

    return torch.cat(ys, dim=0)


def fit_mean_gp(X: torch.Tensor, Y_mean: torch.Tensor, Yvar_mean: torch.Tensor) -> SingleTaskGP:
    Y_mean = ensure_2d(Y_mean)
    Yvar_mean = ensure_2d(Yvar_mean)
    model = SingleTaskGP(X, Y_mean, train_Yvar=Yvar_mean)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model

def fit_logvar_gp(X: torch.Tensor, S2: torch.Tensor) -> SingleTaskGP:
    S2 = ensure_2d(S2)
    Y_log = torch.log(S2.clamp_min(1e-12))
    model = SingleTaskGP(X, Y_log)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def rahbo_optimize( eval_one: EvalOneFn, bounds: torch.Tensor, n_init: int, n_iter: int, cfg: RAHBOConfig, seed: int = 0, ) -> Dict[str, Any]:
    
    torch.manual_seed(seed)
    device = bounds.device
    dtype = bounds.dtype
    d = bounds.shape[1]

    X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_init, d, device=device, dtype=dtype)

    Y_mean_list = []
    S2_list = []
    for i in range(n_init):
        yk = evaluate_k(eval_one, X[i], cfg.k, dtype, device)          
        m = yk.mean(dim=0)                                             
        s2 = yk.var(dim=0, unbiased=True).clamp_min(1e-12)           
        Y_mean_list.append(m)
        S2_list.append(s2)

    Y_mean = torch.cat(Y_mean_list, dim=0)    # (n,1)
    S2 = torch.cat(S2_list, dim=0)            # (n,1)

    #RAHBO loop 
    for t in range(1, n_iter + 1):
        # Fit var model on log variance
        model_logvar = fit_logvar_gp(X, S2)

        # Observation noise for the mean-of-k is S2/k (plus tiny reg for stability)
        Yvar_mean = (S2 / cfg.k).clamp_min(1e-12) + cfg.lambda_reg

        # Fit mean model (with heteroskedatsic noise)
        model_f = fit_mean_gp(X, Y_mean, Yvar_mean)

        # Risk-averse acquisition
        acq = RiskAverseUCB(model_f, model_logvar, cfg.alpha, cfg.beta_f, cfg.beta_var)

        # Pick next x by maximizing RAUCB
        x_next, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1, 
            num_restarts=cfg.num_restarts,
            raw_samples=cfg.raw_samples,
        )
        x_next = x_next.squeeze(0)

        yk = evaluate_k(eval_one, x_next, cfg.k, dtype, device)
        m_next = yk.mean(dim=0)                                   # (1,1)
        s2_next = yk.var(dim=0, unbiased=True).clamp_min(1e-12)    # (1,1)

        # Append
        X = torch.cat([X, x_next.view(1, -1)], dim=0)
        Y_mean = torch.cat([Y_mean, m_next], dim=0)
        S2 = torch.cat([S2, s2_next], dim=0)

        best_i = Y_mean.argmax().item()
        #debug printing 
        #print(f"iter {t:02d} | best_mean={Y_mean[best_i].item():.4f} | best_x={X[best_i].tolist()} | var={S2[best_i].item():.4f}")

    best_i = Y_mean.argmax().item()
    return {
        "X": X,
        "Y_mean": Y_mean,
        "S2": S2,
        "x_best": X[best_i],
        "y_best": Y_mean[best_i],
        "var_best": S2[best_i],
    }
