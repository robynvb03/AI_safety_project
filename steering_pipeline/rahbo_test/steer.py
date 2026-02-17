import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
import json
import csv

MODEL_NAME = "gpt2-small"
STEERING_LAYER = 6 
STEERING_VEC_FILE = "steering_vec.txt"

model = HookedTransformer.from_pretrained(MODEL_NAME)

class prompt_generator:
    def __init__(self, model_name: str = MODEL_NAME, steering_layer: int = STEERING_LAYER):
        self.model_name = model_name
        self.steering_layer = steering_layer
        self.model = model  

    def __call__(self, steering_vector):
        v = steering_vector
        if not torch.is_tensor(v):
            v = torch.tensor(v, dtype=torch.float32)

        d_model = self.model.cfg.d_model
        if v.ndim != 1 or v.shape[0] != d_model:
            raise ValueError(f"steering_vector must have shape [{d_model}], got {tuple(v.shape)}")

        delta = v.to(self.model.cfg.device).view(1, 1, -1)

        steer_hook_name = f"blocks.{self.steering_layer}.hook_resid_pre"
        capture_hook_name = f"blocks.{self.steering_layer + 1}.hook_resid_pre"

        captured = {"act": None}

        def steering_hook(resid: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            return resid + delta

        def capture_hook(resid: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            captured["act"] = resid.detach()
            return resid

        prompt = "What is the capital of France?"

        with self.model.hooks(fwd_hooks=[(steer_hook_name, steering_hook), (capture_hook_name, capture_hook)]):
            _ = self.model.generate(
                prompt,
                max_new_tokens=100,
                temperature=0.9,
                return_type="tokens",  
            )

        if captured["act"] is None:
            raise RuntimeError(f"Failed to capture activations at {capture_hook_name}")

        return float(captured["act"].sum().item())
