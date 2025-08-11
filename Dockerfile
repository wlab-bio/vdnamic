# syntax=docker/dockerfile:1
# Minimal CPU-only image using micromamba for reproducible envs
FROM mambaorg/micromamba:1.5.8

# Make activation implicit for RUN/CMD
SHELL ["/bin/bash", "-lc"]
ENV MAMBA_DOCKERFILE_ACTIVATE=1

# Workdir for your repo
WORKDIR /opt/app

# Copy only the environment first to leverage layer caching
COPY environment.yml /tmp/environment.yml

# Create the conda environment named `project` from environment.yml
RUN micromamba create -y -n project -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Make the env default
ENV CONDA_DEFAULT_ENV=project
ENV PATH=/opt/conda/envs/project/bin:${PATH}

# Now copy the rest of the repository
COPY . /opt/app

# OPTIONAL: if you later add setup.py/pyproject.toml and want editable install, uncomment:
# RUN if [ -f "pyproject.toml" ] || [ -f "setup.py" ]; then \
#         python -m pip install --no-cache-dir -e . ; \
#     fi

# NOTE: We intentionally do NOT pip install requirements.txt here, since the
# environment is defined in environment.yml. This avoids mixing pip/conda for the
# same packages (e.g., faiss-cpu) which can cause conflicts. If you still want to
# use requirements.txt for extras, add them under the pip: section in environment.yml.

# Default command (adjust to your CLI or entrypoint)
CMD ["python", "-c", "import numpy, sklearn, faiss, annoy, pymetis, biopython; print('Container is ready')"]
