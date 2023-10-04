# Requirements

- Linux
- Python 3.5+
- PyTorch 1.11
- TensorBoard
- CUDA 11.0+
- GCC 4.9+
- 1.11 <= Numpy <= 1.23
- PyYaml
- Pandas
- h5py
- joblib

# Compilation

Part of NMS is implemented in C++. The code can be compiled by

```shell
cd ./libs/utils
python setup.py install --user # if in a venv, omit the --user flag
cd ../..
```

The code should be recompiled every time you update PyTorch.

## Install instructions on GPUClient container:
- the requirements are available in requirements.txt
  - note: before installing requirements, you should install PyTorch 1.11 in your Python (3.8 works) venv with:
  ```shell
  pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
  ```
- compile NMS, as shown above
- To reproduce VideoMAEv2's THUMOS results:
  ```shell
  python ./train.py ./configs/thumos_videomaev2.yaml --output reproduce
  ```

