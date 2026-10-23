# Jenkins controller for ctgame-analytics: Jenkins LTS plus the tools that the Jenkinsfile calls
# (Python 3.13 for the tests, the Docker CLI for images, kubectl for the k3s deployment).
# Plugins come from plugins.txt and all settings, including the job, from casc.yaml, so the whole
# server can be rebuilt from Git with `docker compose up -d --build`.
FROM jenkins/jenkins:2.580.1-lts-jdk21

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Docker CLI and buildx; the daemon is the host's one, reached through /var/run/docker.sock.
COPY --from=docker:27.5.1-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker:27.5.1-cli /usr/local/libexec/docker/cli-plugins/docker-buildx \
     /usr/local/libexec/docker/cli-plugins/docker-buildx

# kubectl from the official release, checked against its published SHA-256.
ARG KUBECTL_VERSION=v1.31.4
RUN curl -fsSLo /usr/local/bin/kubectl "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
    && echo "$(curl -fsSL "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl.sha256")  /usr/local/bin/kubectl" \
       | sha256sum --check \
    && chmod +x /usr/local/bin/kubectl

USER jenkins

ENV JAVA_OPTS="-Djenkins.install.runSetupWizard=false" \
    CASC_JENKINS_CONFIG=/usr/share/jenkins/casc.yaml

COPY plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli --plugin-file /usr/share/jenkins/ref/plugins.txt

COPY casc.yaml /usr/share/jenkins/casc.yaml
