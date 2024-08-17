
# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq** and develop locally:

``` bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./

# to install the latest stable release (0.10.x)
# pip install fairseq
```

* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`
* If you use Docker make sure to increase the shared memory size either with `--ipc=host` or `--shm-size`
 as command line options to `nvidia-docker run` .

# Getting Started

The [full documentation](https://fairseq.readthedocs.io/) contains instructions
for getting started, training new models and extending fairseq with new model
types and tasks.


# For paper "Encoder–Decoder Calibration for Multimodal Machine Translation"

1. To preprocess text data, run the process.sh file.
2. Run Shell scripts in the examples file for paper training and testing.

- Paper [pdf](http://dx.doi.org/10.1109/TAI.2024.3354668) download connection.

# Citation

Please cite as:

``` bibtex
@ARTICLE{10401981,
  author={Tayir, Turghun and Li, Lin and Li, Bei and Liu, Jianquan and Lee, Kong Aik},
  journal={IEEE Transactions on Artificial Intelligence}, 
  title={Encoder–Decoder Calibration for Multimodal Machine Translation}, 
  year={2024},
  volume={5},
  number={8},
  pages={3965-3973},
  doi={10.1109/TAI.2024.3354668}}

```
