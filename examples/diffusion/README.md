# Diffusion

`//examples/diffusion` runs one shot image generation from a model repository.
These models are generally found on huggingface where the provider downloads the model weights.

Current supporting list is as follows:

- Tongyi-MAI/Z-Image

## Run

To load a model from HuggingFace directly:

```bash
#CPU
bazel run //examples/diffusion --model=hf://Tongyi-MAI/Z-Image
#CUDA
bazel run //examples/diffusion --@zml//platforms:cuda=true -- --model=hf://Tongyi-MAI/Z-Image
#ROCm
bazel run //examples/diffusion --@zml//platforms:rocm=true -- --model=hf://Tongyi-MAI/Z-Image
```

From a local directory

```bash
bazel run //examples//diffusion --@zml//platform:cuda=true -- --model=/var/models/Tongyi-MAI/Z-Image --prompt="A tesla cybertruck on mars"
```

## Options

- `--model=<path>`: Required. Model repository to load the weights. This can be a local path aswell as huggingface/S3 URI such as `hf://...` or `s3://...`
- `--prompt="string>"`: Required. Runs a single prompt for image generation.
- `--save-path=<path>`: Required. Required to save an image to a particular location.
- `--backend=<vanilla|cuda_fa2|cuda_fa3>`: Optional. Attention backend. If omitted, the program auto-selects one for the current platform.
