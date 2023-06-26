# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ubuntu:20.04

LABEL maintainer="Lightning-AI <https://github.com/Lightning-AI>"

ARG PYTHON_VERSION=3.9
ARG PYTORCH_VERSION=1.13

SHELL ["/bin/bash", "-c"]

# for skipping configurations
ENV \
    DEBIAN_FRONTEND="noninteractive" \
    MKL_THREADING_LAYER="GNU"

RUN apt-get update -qq --fix-missing && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        jq \
        libopenmpi-dev \
        unzip \
        wget \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-distutils \
        python${PYTHON_VERSION}-dev \
    && \
    update-alternatives --install /usr/bin/python${PYTHON_VERSION%%.*} python${PYTHON_VERSION%%.*} /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    # Cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*

RUN \
    wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll --no-check-certificate && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py

# ToDo: install follwong packages
#RUN \
#    tar -xzf /opt/poplar/${{ variables.poplar_sdk }}.tar.gz && \
#    pip debug --verbose && \
#    pip install ${{ variables.poplar_sdk }}/poptorch-*ubuntu*.whl

RUN \
    # Show what we have
    pip list && \
    python -c "import sys; assert sys.version[:3] == '$PYTHON_VERSION', sys.version"
    # python -c "import torch; assert torch.__version__.startswith('$PYTORCH_VERSION'), torch.__version__"
