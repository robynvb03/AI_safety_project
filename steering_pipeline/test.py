import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

MODEL_NAME = "gpt2-small"
STEERING_LAYER = 6 #gpt2-small has 12 layers
REPETITIONS = 3
OUTPUT_FILE = "outputs.txt"
STEERING_VEC_FILE = "steering_vec.txt"
PROMPT = "It's raining cats and dogs"

model = HookedTransformer.from_pretrained(MODEL_NAME)

STEERING_VECTOR1 = torch.randn(model.cfg.d_model)


class prompt_generator:
    def __init__(self, model_name: str = MODEL_NAME, steering_layer: int = STEERING_LAYER):
        self.model_name = model_name
        self.steering_layer = steering_layer
        self.model = model

    def __call__(self, prompt, steering_vector, m: int, out_file: str = OUTPUT_FILE):
        # normalize steering vector
        v = steering_vector
        if not torch.is_tensor(v):
            v = torch.tensor(v, dtype=torch.float32)

        # check steering vector compatibility
        d_model = self.model.cfg.d_model
        if v.ndim != 1 or v.shape[0] != d_model:
            raise ValueError(
                f"Invalid Shape: steering_vector must have shape [{d_model}], got {tuple(v.shape)}"
            )

        # save steering vector to file
        with open(STEERING_VEC_FILE, "w", encoding="utf-8") as f:
            for x in v.tolist():
                f.write(f"{x}\n")

        delta = v.to(self.model.cfg.device).view(1, 1, -1)
        hook_name = f"blocks.{self.steering_layer}.hook_resid_pre"

        def steering_hook(resid: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            return resid + delta

        outputs = []
        for _ in range(m):
            with self.model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
                text = self.model.generate(
                    prompt,
                    max_new_tokens=100,
                    temperature=0.9,
                )
            outputs.append(text)

        with open(out_file, "w", encoding="utf-8") as f:
            for line in outputs:
                f.write(line.replace("\n", " ") + "\n")

        # cleanup hooks
        self.model.reset_hooks()

        return outputs


if __name__ == "__main__":
    gen = prompt_generator()
    results = gen(PROMPT, STEERING_VECTOR1, REPETITIONS)
