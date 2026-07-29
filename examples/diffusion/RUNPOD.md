# Z-Image on RunPod

This guide syncs a local ZML checkout to RunPod, builds the diffusion example
locally on the pod, downloads Z-Image once, and generates images from the local
model snapshot.

Do not add `--config=remote` to these commands. That configuration sends Bazel
actions and build-event data to BuildBuddy.

## 1. Configure the connection

Run these commands on the local machine from the ZML repository:

```bash
cd /path/to/zml

export RUNPOD_HOST=82.221.170.242
export RUNPOD_PORT=26843
export RUNPOD_USER=root
export RUNPOD_KEY="$HOME/.ssh/runpod"
```

Test SSH authentication:

```bash
nix-shell -p openssh --run "ssh -p $RUNPOD_PORT -i $RUNPOD_KEY -o StrictHostKeyChecking=accept-new $RUNPOD_USER@$RUNPOD_HOST 'nvidia-smi'"
```

The host must report driver version `580` or newer for ZML's CUDA 13.x
dependencies.

## 2. Sync ZML

Install `rsync` in a new RunPod container if it is not already available:

```bash
nix-shell -p openssh --run "ssh -p $RUNPOD_PORT -i $RUNPOD_KEY -o StrictHostKeyChecking=accept-new $RUNPOD_USER@$RUNPOD_HOST 'apt-get update && apt-get install -y rsync'"
```

Sync the working source tree, including uncommitted changes:

```bash
nix-shell -p rsync openssh --run "rsync -az --info=progress2 \
  --exclude=.git/ \
  --exclude=bazel-bin/ \
  --exclude=bazel-out/ \
  --exclude=bazel-testlogs/ \
  --exclude=bazel-zml/ \
  --exclude=.cache/ \
  -e 'ssh -p $RUNPOD_PORT -i $RUNPOD_KEY -o StrictHostKeyChecking=accept-new' \
  ./ $RUNPOD_USER@$RUNPOD_HOST:/workspace/zml/"
```

Connect to the pod:

```bash
nix-shell -p openssh --run "ssh -p $RUNPOD_PORT -i $RUNPOD_KEY $RUNPOD_USER@$RUNPOD_HOST"
```

The remaining commands run on the RunPod server.

## 3. Prepare storage

Use at least 60 GB of persistent workspace storage. Confirm the GPU and free
space:

```bash
nvidia-smi
df -h /workspace
```

Keep Bazel outputs off the smaller container root disk:

```bash
cd /workspace/zml

mkdir -p /workspace/.cache/bazel
mkdir -p /workspace/.cache/bazelisk
mkdir -p /workspace/tmp
mkdir -p /workspace/models/Z-Image

export BAZELISK_HOME=/workspace/.cache/bazelisk
export TMPDIR=/workspace/tmp
```

The repository uses Bazel 8.5.1. Confirm the installed command:

```bash
bazel --version
```

## 4. Test and build

Run the scheduler test:

```bash
bazel --output_user_root=/workspace/.cache/bazel test \
  //examples/diffusion:scheduler_test \
  --test_output=all
```

Build the CUDA binary:

```bash
bazel --output_user_root=/workspace/.cache/bazel build \
  --config=release \
  --@zml//platforms:cuda=true \
  --@zml//platforms:cpu=false \
  //examples/diffusion
```

Always pass the same `--output_user_root` value. Using the default
`/root/.cache/bazel` creates a separate cache and causes a full rebuild.

## 5. Download Z-Image once

The `hf://` VFS streams model weights again for every process. Download a
persistent local snapshot for repeated generation:

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 \
bazel --output_user_root=/workspace/.cache/bazel run //tools/hf -- \
  download Tongyi-MAI/Z-Image \
  --local-dir /workspace/models/Z-Image
```

The command resumes partial downloads. Keep
`HF_HUB_ENABLE_HF_TRANSFER=0` because the RunPod image may enable
`hf_transfer` while Bazel's isolated Python environment does not expose that
optional package.

Verify the snapshot:

```bash
du -sh /workspace/models/Z-Image
ls /workspace/models/Z-Image
```

## 6. Run a smoke test

This one-step run verifies CUDA selection, transformer execution, the scheduler,
VAE decoding, and PNG output:

```bash
bazel --output_user_root=/workspace/.cache/bazel run \
  --config=release \
  --@zml//platforms:cuda=true \
  --@zml//platforms:cpu=false \
  //examples/diffusion -- \
  --model=/workspace/models/Z-Image \
  --prompt="A red cube on a white table, studio photograph" \
  --height=256 \
  --width=256 \
  --steps=1 \
  --guidance-scale=0 \
  --seqlen=64 \
  --seed=42 \
  --output=/workspace/zimage-smoke.png
```

The runtime log must report `platform: cuda`.

Verify the output:

```bash
file /workspace/zimage-smoke.png
```

## 7. Generate a 512px image with the base model

`Tongyi-MAI/Z-Image` is the undistilled base model. It requires classifier-free
guidance and 28 to 50 denoising steps. Do not use the Turbo settings of 8 steps
and guidance scale 0 with this model; that leaves the output as decoded noise.

```bash
bazel --output_user_root=/workspace/.cache/bazel run \
  --config=release \
  --@zml//platforms:cuda=true \
  --@zml//platforms:cpu=false \
  //examples/diffusion -- \
  --model=/workspace/models/Z-Image \
  --prompt="A cinematic photograph of a red fox walking through a snowy pine forest, soft morning light, highly detailed" \
  --height=512 \
  --width=512 \
  --steps=28 \
  --guidance-scale=4 \
  --seqlen=256 \
  --seed=42 \
  --output=/workspace/zimage-512.png
```

For the official maximum-quality settings, use `--steps=50`. The base model
supports `--negative-prompt` and `--cfg-normalization`.

Subsequent runs reuse both `/workspace/.cache/bazel` and the model snapshot in
`/workspace/models/Z-Image`. Model weights still have to be read from local disk
and loaded into VRAM for each new process.

## 8. Benchmark cold end-to-end latency

Build the CUDA target first so the timed command does not include Bazel
compilation:

```bash
bazel --output_user_root=/workspace/.cache/bazel build \
  --config=release \
  --@zml//platforms:cuda=true \
  --@zml//platforms:cpu=false \
  //examples/diffusion
```

Then time the complete process:

```bash
/usr/bin/time -f $'\nwall=%e\nuser=%U\nsystem=%S\nmax_rss_kb=%M' \
bazel --output_user_root=/workspace/.cache/bazel run \
  --config=release \
  --@zml//platforms:cuda=true \
  --@zml//platforms:cpu=false \
  //examples/diffusion -- \
  --model=/workspace/models/Z-Image \
  --prompt="A red cube on a white table, studio photograph" \
  --height=256 \
  --width=256 \
  --steps=28 \
  --guidance-scale=4 \
  --seqlen=64 \
  --seed=42 \
  --output=/workspace/zml-benchmark.png
```

This is a cold end-to-end measurement. It includes model loading, runtime PJRT
compilation, inference, VAE decoding, and PNG writing. Bazel's action cache does
not persist PJRT executables between processes.

## 9. Copy generated images locally

Run this from the local machine:

```bash
nix-shell -p rsync openssh --run "rsync -az \
  -e 'ssh -p $RUNPOD_PORT -i $RUNPOD_KEY' \
  $RUNPOD_USER@$RUNPOD_HOST:/workspace/zimage-*.png ./"
```
