# Adapted from MMPOSE dockerfile
# Adapted by Manu Ramesh

ARG PYTORCH="1.13.0"
ARG CUDA="11.6"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install xtcocotools
RUN pip install cython
RUN pip install xtcocotools

# Install MMEngine and MMCV
RUN pip install openmim
RUN mim install mmengine "mmcv>=2.0.0"

# Install MMPose
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmpose.git /mmpose
WORKDIR /mmpose
RUN git checkout main
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

#####################################
# For autocattlogger
RUN python -m pip install scikit-image==0.19.1
RUN python -m pip install shapely
RUN python -m pip install ndicts
RUN python -m pip install distinctipy
RUN python -m pip install joblib
RUN python -m pip install h5py

# Detectron2 - https://detectron2.readthedocs.io/en/latest/tutorials/install.html
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Version incompatibility fixes

#RUN python -m pip uninstall numpy
RUN python -m pip install numpy==1.24.4
#RUN python -m pip uninstall opencv-python
RUN python -m pip install opencv-python==4.11.0.86

RUN python -m pip install mmdet==3.2.0
RUN python -m pip install mmcv==2.1.0

# Add lines to clone AutoCattlogger public repo and to change workdir to the AutoCattlogger folder





