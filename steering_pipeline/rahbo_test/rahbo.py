import torch
import math
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
import matplotlib.pyplot as plt
from typing import Any, Dict
import csv
import os


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
    s2_min: float = 1e-8
    s2_max: float = 1e2


class RiskAverseUCB(AnalyticAcquisitionFunction):
    # acq(x) = UCB_f(x) - alpha * LCB_var(x) 
    def __init__(self, model_f, model_var, alpha: float, beta_f: float, beta_var: float):
        super().__init__(model=model_f)
        self.model_f = model_f
        self.model_var = model_var
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
        post_v = self.model_var.posterior(X_)
        mu_v = post_v.mean.squeeze(-1)
        sig_v = post_v.variance.clamp_min(1e-12).sqrt().squeeze(-1)
        lcb_v = (mu_v - self.beta_var * sig_v).clamp_min(1e-12)

        return ucb_f - self.alpha * lcb_v


def ensure_2d(Y: torch.Tensor) -> torch.Tensor:
    if Y.dim() == 0:
        return Y.view(1, 1)
    if Y.dim() == 1:
        return Y.unsqueeze(-1)
    return Y


def evaluate_k(eval_one: EvalOneFn, x: torch.Tensor, k: int, dtype, device) -> torch.Tensor:
    ys = []
    for _ in range(k):
        y = eval_one(x)
        y = torch.as_tensor(y, dtype=dtype, device=device).reshape(1, 1)
        ys.append(y)
    return torch.cat(ys, dim=0)


def fit_mean_gp(X: torch.Tensor, Y_mean: torch.Tensor, Yvar_mean: torch.Tensor) -> SingleTaskGP:
    Y_mean = ensure_2d(Y_mean)
    Yvar_mean = ensure_2d(Yvar_mean)
    model = SingleTaskGP(X, Y_mean, train_Yvar=Yvar_mean)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    #mll = VariationalELBO(model.likelihood, model, 250)
    fit_gpytorch_mll(mll)
    return model

def fit_var_gp(X: torch.Tensor, S2: torch.Tensor, s2_min: float, s2_max: float) -> SingleTaskGP:
    S2 = ensure_2d(S2).clamp(s2_min, s2_max)
    model = SingleTaskGP(X, S2)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    #mll = VariationalELBO(model.likelihood, model, num_data=250)
    fit_gpytorch_mll(mll)
    return model
 

def rahbo_optimize( eval_one: EvalOneFn, bounds: torch.Tensor, n_init: int, n_iter: int, cfg: RAHBOConfig, seed: int = 0, ) -> Dict[str, Any]:
    
    torch.manual_seed(seed)
    device = bounds.device
    dtype = bounds.dtype
    d = bounds.shape[1]

    log_path = "rahbo_samples.csv"
    write_header = not os.path.exists(log_path)
    log_f = open(log_path, "a", newline="")
    log_writer = csv.writer(log_f)
    if write_header:
        log_writer.writerow(["iter", "reward","variance",  "risk_reward"])
    log_f.flush()

    samples_path = "rahbo_samples_raw.csv"
    write_header_samples = not os.path.exists(samples_path)
    samples_f = open(samples_path, "a", newline="")
    samples_writer = csv.writer(samples_f)

    if write_header_samples:
        samples_writer.writerow(
            ["iter", "sample_idx", "reward"])


    X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_init, d, device=device, dtype=dtype)

    Y_mean_list = []
    S2_list = []
    for i in range(n_init):
        yk = evaluate_k(eval_one, X[i], cfg.k, dtype, device)
        for s in range(cfg.k):
            samples_writer.writerow( [0, s, float(yk[s].item())])

        m = yk.mean(dim=0)  
        s2 = yk.var(dim=0, unbiased=True).clamp_min(1e-12)           
        Y_mean_list.append(m)
        S2_list.append(s2)
        risk_reward = m.item() - cfg.alpha * s2.item()
        log_writer.writerow([0, float(m.item()), float(s2.item()), float(risk_reward)])                                       

    Y_mean = torch.cat(Y_mean_list, dim=0)    # (n,1)
    S2 = torch.cat(S2_list, dim=0)            # (n,1)

    #tracking for graph generation
    risk_best_mean_so_far = []
    risk_best_var_so_far = []
    mean_best_mean_so_far = []
    mean_best_var_so_far = []


    #RAHBO loop 
    for t in range(1, n_iter + 1):
        # Fit var model on variance
        model_var = fit_var_gp(X, S2, cfg.s2_min, cfg.s2_max)

        # Observation noise for the mean-of-k is S2/k (plus tiny reg for stability)
        with torch.no_grad():
            post_v_X = model_var.posterior(X)
            mu_v_X = post_v_X.mean.squeeze(-1)
            sig_v_X = post_v_X.variance.clamp_min(1e-12).sqrt().squeeze(-1)
            ucb_v_X = (mu_v_X + cfg.beta_var * sig_v_X).clamp_min(cfg.s2_min).unsqueeze(-1)

        Yvar_mean = (ucb_v_X / cfg.k).clamp_min(cfg.s2_min) + cfg.lambda_reg

        # Fit mean model (with heteroskedatsic noise)
        model_f = fit_mean_gp(X, Y_mean, Yvar_mean)

        # Risk-averse acquisition
        acq = RiskAverseUCB(model_f, model_var, cfg.alpha, cfg.beta_f, cfg.beta_var)
    
        # Pick next x by maximizing RAUCB
        x_next, _ = optimize_acqf( acq_function=acq, bounds=bounds, q=1, num_restarts=cfg.num_restarts, raw_samples=cfg.raw_samples, )
        x_next = x_next.squeeze(0)

        yk = evaluate_k(eval_one, x_next, cfg.k, dtype, device)

        for s in range(cfg.k):
            samples_writer.writerow( [t, s, float(yk[s].item())])

        m_next = yk.mean(dim=0)                                   # (1,1)
        s2_next = yk.var(dim=0, unbiased=True).clamp_min(1e-12)    # (1,1)
        risk_reward_next = m_next.item() - cfg.alpha * s2_next.item()
        log_writer.writerow([t, float(m_next.item()), float(s2_next.item()), float(risk_reward_next)])


        # Append
        X = torch.cat([X, x_next.view(1, -1)], dim=0)
        Y_mean = torch.cat([Y_mean, m_next], dim=0)
        S2 = torch.cat([S2, s2_next], dim=0)

        with torch.no_grad():
            # LCB mean
            post_f_X = model_f.posterior(X)
            mu_f_X = post_f_X.mean.squeeze(-1)
            sig_f_X = post_f_X.variance.clamp_min(1e-12).sqrt().squeeze(-1)
            lcb_f_X = mu_f_X - cfg.beta_f * sig_f_X

            # UCB variance
            post_v_X = model_var.posterior(X)
            mu_v_X = post_v_X.mean.squeeze(-1)
            sig_v_X = post_v_X.variance.clamp_min(1e-12).sqrt().squeeze(-1)
            ucb_v_X = (mu_v_X + cfg.beta_var * sig_v_X).clamp_min(cfg.s2_min)

            score_X = lcb_f_X - cfg.alpha * ucb_v_X

        obs_score = Y_mean.squeeze(-1) - cfg.alpha * S2.squeeze(-1)
        best_i_risk = obs_score.argmax().item()
        risk_best_mean_so_far.append(Y_mean[best_i_risk].item())
        risk_best_var_so_far.append(S2[best_i_risk].item())

        best_i_mean = Y_mean.argmax().item()
        mean_best_mean_so_far.append(Y_mean[best_i_mean].item())
        mean_best_var_so_far.append(S2[best_i_mean].item())

        #debug printing 
        print( f"iter {t:02d} | " f"meanbest={Y_mean[best_i_mean].item():.4f}, var@meanbest={S2[best_i_mean].item():.4e} | " f"riskbest_mean={Y_mean[best_i_risk].item():.4f}, riskbest_var={S2[best_i_risk].item():.4e} | " f"new_mean={m_next.item():.4f}, new_var={s2_next.item():.3e}")

    iters = list(range(1, n_iter + 1))
    log_f.close()


    # Plot 1: Risk-aware best so far
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Mean")
    ax1.plot(iters, [-m for m in risk_best_mean_so_far], color="blue")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Variance")
    ax2.plot(iters, risk_best_var_so_far, color="red", linestyle="--")
    plt.title("Risk-Aware Best So Far")
    ax1.grid(True)
    plt.savefig("riskaware.png", dpi=300, bbox_inches='tight')

    # Plot 2: Highest mean so far
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Mean")
    ax1.plot(iters, [-m for m in mean_best_mean_so_far], color="blue")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Variance")
    ax2.plot(iters, mean_best_var_so_far, color="red", linestyle="--")
    plt.title("Highest Mean So Far")
    ax1.grid(True)
    plt.savefig("bestmean.png", dpi=300, bbox_inches='tight')

    # final incumbents
    final_best_i_mean = Y_mean.argmax().item()
    final_best_i_risk = best_i_risk  # from last loop

    return {
        "X": X,
        "Y_mean": Y_mean,
        "S2": S2,
        # risk-aware incumbent
        "x_best": X[final_best_i_risk],
        "y_best": Y_mean[final_best_i_risk],
        "var_best": S2[final_best_i_risk],
    }

 