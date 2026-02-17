import torch
import math
import pytry
from sklearn import svm
import pickle
import rahbo
import steer
from steer import *


def black_box_steering(steering_vector):
    gen = prompt_generator(model_name=MODEL_NAME, steering_layer=STEERING_LAYER)
    score = gen(steering_vector)
    return (-1 *score) 


class RAHBOSweep(pytry.Trial):
    def params(self):

        self.param("risk aversion", alpha=1.0)
        self.param("exploration on mean", beta_f=2.0)
        self.param("conservativeness on variance", beta_var=2.0)

        self.param("k repeats per x", k=5)
        self.param("lambda reg", lambda_reg=1e-6)
        self.param("num restarts", num_restarts=20)
        self.param("raw samples", raw_samples=250)

        self.param("n_init", n_init=50)
        self.param("n_iter", n_iter=200)

        self.param("device", device="cuda")
        self.param("dtype", dtype="double")

    def evaluate(self, p):
        device = torch.device(p.device)
        dtype = torch.double if p.dtype == "double" else torch.float

        bounds = torch.tensor(
            [ [-10] * 768,
            [10] * 768 ],
            device=device,
            dtype=dtype,
        )

        cfg = rahbo.RAHBOConfig(
            alpha=float(p.alpha),
            beta_f=float(p.beta_f),
            beta_var=float(p.beta_var),
            k=int(p.k),
            lambda_reg=float(p.lambda_reg),
            num_restarts=int(p.num_restarts),
            raw_samples=int(p.raw_samples),
        )

        result = rahbo.rahbo_optimize(
            eval_one=black_box_steering,
            bounds=bounds,
            n_init=int(p.n_init),
            n_iter=int(p.n_iter),
            cfg=cfg,
            seed=int(p.seed),
        )

        xb = result["x_best"]
        return {
            "x_best": xb.tolist(),
            "y_best_mean": float(result["y_best"].item()),
            "var_best": float(result["var_best"].item()),
        }


if __name__ == "__main__":

    alphas = [0.5]
    beta_fs = [1.5]
    beta_vars = [1.0]

    for a in alphas:
        for bf in beta_fs:
            for bv in beta_vars: 
                RAHBOSweep().run(alpha=a, beta_f=bf, beta_var=bv, verbose=True)
