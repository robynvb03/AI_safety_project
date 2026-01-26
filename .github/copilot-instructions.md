# Copilot instructions — nrc steering pipeline

Purpose
- Help AI coding agents quickly understand and modify this small TransformerLens-based steering pipeline.

Big picture
- This repo implements a simple steering pipeline that injects a vector into the residual stream of a TransformerLens `HookedTransformer` (see `steer.py`).
- Flow: prompt -> tokens -> `run_with_cache` (to compute `prompt_embedding`) -> optional steering vector -> generate via `model.generate` with `fwd_hooks`.

Key files
- `steer.py`: primary implementation. Look at `SteeringPipeline`, `_validate_vector`, and `run()` for the core logic and hook usage.
- `test.py`: a short, incomplete example; it demonstrates intended usage but contains minor bugs (undefined `text`, typos). Use `steer.py` as the canonical reference.

Important patterns & conventions
- Device selection: `SteeringPipeline` uses `device or ("cuda" if torch.cuda.is_available() else "cpu")`. Respect this pattern when adding CLI args.
- Hook naming: steering hook uses `blocks.{layer}.hook_resid_pre`. When adding hooks, follow this exact string format.
- Steering semantics: `apply_to` accepts `all` or `prompt_only`. `repetitions` multiplies the steering vector. Vectors must be 1-D of length `d_model`.
- Prompt embedding: computed as the mean of `cache['resid_pre', 0]` across prompt tokens (returned as `SteeringResult.prompt_embedding`).
- Output shape & meta: functions return a `SteeringResult` dataclass with `text`, `prompt_embedding` and `meta` dict. Keep `meta` keys consistent (`model_name`, `steered`, `steering_layer`, etc.).

Developer workflows
- Run the demo: `python steer.py` (downloads `gpt2-small` if needed). The script prints generated text and metadata.
- Install dependencies (minimum):
```bash
python -m pip install torch transformer_lens
```
- Model weights are loaded by `HookedTransformer.from_pretrained` — expect network/download on first run or use the transformer cache for offline use.

Common edits an AI agent may perform
- Add CLI flags: follow existing device/dtype defaults and expose `steering_layer`, `repetitions`, `apply_to`, `seed`.
- Add tests or examples: fix `test.py` by replacing `text` with `input_prompt` and correcting `stearing_layer` -> `steering_layer`.
- Change hook position: keep the same hook name format and update `meta['hook']` when changing the hook.

Integration points
- External dependency: `transformer_lens` (HookedTransformer API, hooks and `generate` with `fwd_hooks`).
- Torch: tensors, device, dtype, and randomness (`torch.manual_seed`).

Notes for agents
- Prefer `steer.py` as the ground truth. `test.py` is illustrative but buggy.
- When modifying model behavior, run the demo locally to observe generated text and `meta` to confirm hooks fired.
- Keep changes small and focused; preserve `SteeringResult.meta` keys for downstream tooling.

If anything here is unclear or you want more examples (unit tests, CLI, or offline model loading), tell me what to add.
