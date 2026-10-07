# Adapted from https://github.com/facebookresearch/detectron2/blob/main/docker/Dockerfile
FROM ytaekoh2/omniembed:v2
ENV DEBIAN_FRONTEND noninteractive

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

