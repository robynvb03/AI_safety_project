import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
import json
import csv

MODEL_NAME = "gpt2-small"
STEERING_LAYER = 6 #gpt2-small has 12 layers
REPETITIONS = 2
OUTPUT_FILE = "outputs.txt"
STEERING_VEC_FILE = "steering_vec.txt"

model = HookedTransformer.from_pretrained(MODEL_NAME)



def prompt_stream_from_csv(csv_path: str, prompt_col: str = "prompt"):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if prompt_col not in reader.fieldnames:
            raise ValueError(f"CSV missing column '{prompt_col}'. Found: {reader.fieldnames}")
        for row in reader:
            p = (row.get(prompt_col) or "").strip()
            if p:
                yield {"prompt": p}

ds_stream = prompt_stream_from_csv("steering_test.csv", prompt_col="prompt")

class prompt_generator:
    def __init__(self, model_name: str = MODEL_NAME, steering_layer: int = STEERING_LAYER):
        self.model_name = model_name
        self.steering_layer = steering_layer
        self.model = model

    def __call__(self, steering_vector, m: int, out_file: str = OUTPUT_FILE):
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
        # adjust to streaming 
        for example in ds_stream:
            for _ in range(m):
                with self.model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
                    text = self.model.generate(
                        example["prompt"],
                        max_new_tokens=100,
                        temperature=0.9,
                    )
                
                """ PROMPT REMOVAL
                prompt_tokens = self.model.to_tokens(example["prompt"])
                gen_tokens = text1[:, prompt_tokens.shape[1]:]
                text = self.model.to_string(gen_tokens)[0]"""
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(text.replace("\n", " ") + "\n")
            
                outputs.append(text)

        with open(out_file, "w", encoding="utf-8") as f:
            for line in outputs:
                f.write(line.replace("\n", " ") + "\n")

        # cleanup hooks
        self.model.reset_hooks()

        return outputs
 
 
if __name__ == "__main__":
    gen = prompt_generator(model_name=MODEL_NAME, steering_layer=STEERING_LAYER)
    _ = gen([0] * 768, m=REPETITIONS, out_file=OUTPUT_FILE)

