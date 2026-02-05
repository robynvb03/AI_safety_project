import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize
from botorch.models.transforms import Standardize, Normalize

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
torch.manual_seed(0)

bounds = torch.tensor([[-2.0, -2.0], [2.0, 2.0]], dtype=dtype, device=device)

def black_box(x: torch.Tensor) -> torch.Tensor:
    # Dummy function
    y = -(x[:, 0] ** 2 + x[:, 1] ** 2)
    return y.unsqueeze(-1)

# Initial data
d = bounds.shape[1]
n_init = 8
X = unnormalize(torch.rand(n_init, d, dtype=dtype, device=device), bounds)
Y = black_box(X)

# Bayesian Optimization Loop
n_iter = 250
for t in range(n_iter):
    # 1) Fit GP with normalization (recommended)
    model = SingleTaskGP(
        train_X=X,
        train_Y=Y,
        input_transform=Normalize(d=d),
        outcome_transform=Standardize(m=1)
    )
    model.to(device=device, dtype=dtype)
    
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    
    # 2) Acquisition function
    acq = LogExpectedImprovement(model=model, best_f=Y.max())
    
    # 3) Optimize acquisition
    x_next, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=256,
        options={"batch_limit": 5, "maxiter": 200},
    )
    
    # 4) Evaluate
    y_next = black_box(x_next)
    
    # 5) Update data
    X = torch.cat([X, x_next], dim=0)
    Y = torch.cat([Y, y_next], dim=0)
    
    # Optional: Print progress every 20 iterations
    if (t + 1) % 20 == 0:
        best_i = Y.argmax().item()
        print(f"Iteration {t+1:03d} | Best: {Y[best_i].item():.6f} at {X[best_i].tolist()}")

best_i = Y.argmax().item()
print("\n=== Final Results ===")
print(f"Best x: {X[best_i].tolist()}")
print(f"Best y: {Y[best_i].item():.6f}")
print(f"Total evaluations: {len(Y)}")