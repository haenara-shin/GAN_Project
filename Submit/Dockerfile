# # Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# #
# # NVIDIA CORPORATION and its licensors retain all intellectual property
# # and proprietary rights in and to this software, related documentation
# # and any modifications thereto.  Any use, reproduction, disclosure or
# # distribution of this software and related documentation without an express
# # license agreement from NVIDIA CORPORATION is strictly prohibited.

# FROM nvcr.io/nvidia/pytorch:20.12-py3

# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1

# RUN pip install imageio-ffmpeg==0.4.3 pyspng==0.1.0
# RUN pip install jupyter

# WORKDIR /workspace
# COPY . .

# # Unset TORCH_CUDA_ARCH_LIST and exec.  This makes pytorch run-time
# # extension builds significantly faster as we only compile for the
# # currently active GPU configuration.
# RUN (printf '#!/bin/bash\nunset TORCH_CUDA_ARCH_LIST\nexec \"$@\"\n' >> /entry.sh) && chmod a+x /entry.sh
# ENTRYPOINT ["/entry.sh"]

# CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

###########################################
#
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

FROM nvcr.io/nvidia/pytorch:20.12-py3

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install jupyter click requests tqdm pyspng ninja imageio imageio-ffmpeg==0.4.3

#  stylegan2-ada-pytorch

# RUN mkdir src
ADD Submit /submit
WORKDIR /submit
#WORKDIR /workspace
# COPY . .


# RUN pip install -r requirements.txt

# Unset TORCH_CUDA_ARCH_LIST and exec.  This makes pytorch run-time
# extension builds significantly faster as we only compile for the
# currently active GPU configuration.
RUN (printf '#!/bin/bash\nunset TORCH_CUDA_ARCH_LIST\nexec \"$@\"\n' >> /entry.sh) && chmod a+x /entry.sh
# ENTRYPOINT ["/entry.sh"]

# CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]