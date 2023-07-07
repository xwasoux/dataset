# Note: You can use any Debian/Ubuntu based image you want. 
# FROM  --platform=linux/amd64 mcr.microsoft.com/vscode/devcontainers/base:0-bullseye
FROM  mcr.microsoft.com/vscode/devcontainers/base:0-bullseye
# FROM ubuntu:20.04

# [Option] Install zsh
ARG INSTALL_ZSH="true"
# [Option] Upgrade OS packages to their latest versions
ARG UPGRADE_PACKAGES="false"
# [Option] Enable non-root Docker access in container
ARG ENABLE_NONROOT_DOCKER="true"
# [Option] Use the OSS Moby CLI instead of the licensed Docker CLI
ARG USE_MOBY="true"

# Enable new "BUILDKIT" mode for Docker CLI
ENV DOCKER_BUILDKIT=1
# Install needed packages and setup non-root user. Use a separate RUN statement to add your
# own dependencies. A user of "automatic" attempts to reuse an user ID if one already exists.
ARG USERNAME=vscode
ARG USER_UID=1002
ARG USER_GID=1002
# Proxy setting
ENV http_proxy="http://wwwproxy.osakac.ac.jp:8080"
ENV https_proxy="http://wwwproxy.osakac.ac.jp:8080"

RUN apt-get update 
# User setup
## Github : https://github.com/microsoft/vscode-dev-containers/blob/main/script-library/common-debian.sh
## raw data : https://raw.githubusercontent.com/microsoft/vscode-dev-containers/main/script-library/common-debian.sh
COPY library-scripts/common-debian.sh /tmp/library-scripts/
RUN /bin/bash /tmp/library-scripts/common-debian.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" "true" "true" 
# Use Docker script from script library to set things up
## Github : https://github.com/microsoft/vscode-dev-containers/blob/main/script-library/docker-debian.sh
## raw data : https://raw.githubusercontent.com/microsoft/vscode-dev-containers/main/script-library/docker-debian.sh
COPY library-scripts/docker-debian.sh /tmp/library-scripts/
RUN /bin/bash /tmp/library-scripts/docker-debian.sh "${ENABLE_NONROOT_DOCKER}" "/var/run/docker-host.sock" "/var/run/docker.sock" "${USERNAME}" 
# Python setup
## Github : https://github.com/microsoft/vscode-dev-containers/blob/main/script-library/python-debian.sh
## raw data : https://raw.githubusercontent.com/microsoft/vscode-dev-containers/main/script-library/python-debian.sh
# RUN bash -c "$(wget -qO- 'https://raw.githubusercontent.com/microsoft/vscode-dev-containers/main/script-library/python-debian.sh')" 
COPY library-scripts/python-debian.sh /tmp/library-scripts/
ENV PIPX_HOME=/usr/local/py-utils \
    PIPX_BIN_DIR=/usr/local/py-utils/bin
ENV PATH=${PATH}:${PIPX_BIN_DIR}
RUN /bin/bash /tmp/library-scripts/python-debian.sh "3.9.5" "/usr/local/python" "${PIPX_HOME}" "${USERNAME}"
# Rust setup
## Github : https://github.com/microsoft/vscode-dev-containers/blob/main/script-library/rust-debian.sh
## raw data : https://raw.githubusercontent.com/microsoft/vscode-dev-containers/main/script-library/rust-debian.sh
RUN bash -c "$(wget -qO- 'https://raw.githubusercontent.com/microsoft/vscode-dev-containers/main/script-library/rust-debian.sh')" 
# Clean up
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts/

USER vscode
RUN curl -sSL https://install.python-poetry.org | /usr/local/python/bin/python3.9 -

# RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3 - \
#     && cd /usr/local/bin \
#     && ln -s /opt/poetry/bin/poetry \
#     && poetry config virtualenvs.create false \
#     && poetry config virtualenvs.in-project true

# Setting the ENTRYPOINT to docker-init.sh will configure non-root access 
# to the Docker socket. The script will also execute CMD as needed.
ENTRYPOINT [ "/usr/local/share/docker-init.sh"]
CMD [ "sleep", "infinity" ]

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>