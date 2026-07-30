# Diffusion

`//examples/diffusion` runs one shot image generation from a model repository.
We support the following models, automatically detected from the model_type in the config.json:

Currently supported models:

- Tongyi-MAI/Z-Image

## Run

To load a model from HuggingFace directly:

```bash
# CPU
bazel run //examples/diffusion -- --model=hf://Tongyi-MAI/Z-Image
# CUDA
bazel run //examples/diffusion --@zml//platforms:cuda=true -- --model=hf://Tongyi-MAI/Z-Image
# ROCm
bazel run //examples/diffusion --@zml//platforms:rocm=true -- --model=hf://Tongyi-MAI/Z-Image
```

From a local directory:

```bash
bazel run //examples/diffusion --@zml//platforms:cuda=true -- --model=/var/models/Tongyi-MAI/Z-Image
```

From a single non-interative prompt:

```bash
bazel run //examples/diffusion --@zml//platforms:cuda=true -- --model=hf://Tongyi-MAI/Z-Image --prompt="A cute cat under the tree"
```

## Options

- `--model=<path>`: Required. Model repository to load. This can be a local path or a Hugging Face/S3 URI such as `hf://...` or `s3://...`.
- `--prompt="string>"`: Required. Runs a single prompt for image generation.
- `--negative-prompt="<string>"`: Optional negative prompt used for classifier-free guidance.
- `--guidance-scale=<float>`: Optional classifier-free guidance scale. Defaults to `5.0`; use `0` to disable guidance.
- `--cfg-normalization`: Optionally caps the guided prediction norm to the positive prediction norm.
- `--height=<pixels>` and `--width=<pixels>`: Output dimensions, each divisible by 16. Both default to `1024`.
- `--steps=<count>`: Number of denoising steps. Defaults to `50`.
- `--seqlen=<count>`: Maximum tokenized prompt length. Defaults to `512`.
- `--output=<path>`: PNG output path. Defaults to `zimage.png`.
