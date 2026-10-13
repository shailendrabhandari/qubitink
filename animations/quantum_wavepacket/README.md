# Quantum wave-packet interference

A GPU animation of an exact coherent-state superposition in a three-dimensional anisotropic harmonic trap. Thin curves trace the instantaneous probability current. Faint surfaces mark two probability-density levels. Frames contain no text.

This is a visualization of a single-particle wavefunction undergoing unitary evolution. It is not a quantum attractor, a quantum-computing simulation, or a set of measured particle trajectories.

## Run

Use an environment with NumPy, Pillow, ModernGL, glcontext, and imageio-ffmpeg. GPU rendering requires OpenGL 4.3 and an EGL context. The supplied renderer is intended for Linux systems with suitable GPU drivers.

```bash
python -m pip install -r requirements.txt
python run_quantum.py --self-test
python run_quantum.py --gpus 0,1 --probe
python run_quantum.py --gpus 0,1 --preview --output quantum_preview.mp4
python run_quantum.py --gpus 0,1 --bitrate 12000 --output quantum_filaments_90s.mp4
```

Choose the EGL device indices reported by `--probe`. They need not match CUDA indices. One device also works, for example `--gpus 0`. The default film is 90 seconds at 1080 × 1920 and 60 frames per second. The preview covers film seconds 24–34 at the normal playback rate. `--steps 384` increases streamline resolution. `--style shells` omits the curves. FFmpeg uses CPU H.264 encoding, while the field evaluation and image rendering use the selected GPUs. Each GPU renders a separate chronological section, and the sections are joined in order. The completed video is fully decoded to check for encoding errors.

See [MATHEMATICS.md](MATHEMATICS.md) for the model and the interpretation of the image. Code is covered by the repository's MIT licence. Keep rendered movies out of ordinary Git commits and attach a movie to a GitHub Release if desired.
