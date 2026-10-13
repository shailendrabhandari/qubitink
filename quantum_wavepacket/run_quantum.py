"""Launcher for quantum_wavepacket.py. Put both files in the same directory.

    python -m pip install numpy Pillow moderngl glcontext imageio-ffmpeg
    python run_quantum.py --self-test
    python run_quantum.py --probe
    python run_quantum.py --preview
    python run_quantum.py

No titles, labels, captions, or watermarks are drawn inside the video.
For a higher-bitrate local master:
    python run_quantum.py --bitrate 12000 --output quantum_master.mp4
For the other two EGL devices:
    python run_quantum.py --gpus 2,3 --preview
"""
from quantum_wavepacket import main

if __name__=='__main__':
    main()
