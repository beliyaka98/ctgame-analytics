# Container image of the ctgame command-line tool.
# The image fixes the Python version and every scientific dependency, so the analysis gives the
# same results on a laptop, on the Jenkins server and in the Kubernetes test cluster.
#
#   docker build -t ctgame-analytics .
#   docker run --rm ctgame-analytics --version
#   docker run --rm -v "$PWD/out:/work" ctgame-analytics simulate --out /work/data

ARG PYTHON_VERSION=3.13

# ---- build: turn the source tree into a wheel ---------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir build==1.2.2.post1
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m build --wheel --outdir /dist

# ---- runtime: only the installed wheel, no compilers and no source tree -------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

# Dependencies first, from the lock file: this heavy layer is cached until the lock changes.
COPY requirements/runtime.txt /tmp/runtime.txt
RUN pip install --no-cache-dir -r /tmp/runtime.txt \
    && useradd --create-home --uid 10001 ctgame \
    && mkdir /work && chown ctgame:ctgame /work

# Then the package itself; --no-deps keeps the locked versions untouched.
COPY --from=build /dist/*.whl /tmp/
RUN pip install --no-cache-dir --no-deps /tmp/*.whl && rm /tmp/*.whl /tmp/runtime.txt

# Build arguments change on every commit, and every RUN after an ARG is rebuilt when its value
# changes. Declaring them here, after the installs, keeps the dependency layers cached.
ARG VERSION=dev
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="ctgame-analytics" \
      org.opencontainers.image.description="Learner profiling, next-task recommendation and pilot evaluation for game-based critical-thinking tasks" \
      org.opencontainers.image.source="https://github.com/beliyaka98/ctgame-analytics" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}"

# Never run the analysis as root: real pilot data are personal data of minors.
USER 10001
WORKDIR /work

ENTRYPOINT ["ctgame"]
CMD ["--help"]
