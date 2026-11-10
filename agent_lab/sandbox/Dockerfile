# agent_lab sandbox image: one container per agent.
# Non-root user, scientific Python stack pre-installed so agents rarely need pip.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git jq bc time procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        numpy pandas scipy statsmodels scikit-learn matplotlib seaborn \
        sympy numba networkx pyarrow requests beautifulsoup4 lxml \
        arch ta tabulate

# Agents run as an unprivileged user; /workspace is bind-mounted per agent.
RUN useradd --create-home --uid 1000 agent && mkdir -p /workspace && chown agent:agent /workspace
ENV MPLBACKEND=Agg PYTHONUNBUFFERED=1 PIP_USER=1 PATH=/home/agent/.local/bin:$PATH
USER agent
WORKDIR /workspace
CMD ["sleep", "infinity"]
