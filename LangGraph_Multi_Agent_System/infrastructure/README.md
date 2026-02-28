# LangGraph Multi-Agent System: Infrastructure

This directory contains the central infrastructure definitions and lifecycle management scripts required to develop and run the LangGraph Multi-Agent System on your local machine. 

Currently, the infrastructure provisions a robust **MongoDB 7.0** database tailored for storing semi-structured medical guidelines and drug records, alongside **Mongo Express** for convenient database introspection.

## Prerequisites

Before starting the infrastructure, ensure you have the following installed locally:

*   **Docker Desktop** (or Docker Engine + Docker Compose Plugin)
    *   *The Docker daemon must be running prior to executing these scripts.*
*   **Python 3** (configured via `uv` or standard `pip` environments)
*   **pymongo** (`pip install pymongo>=4.6.0`)

## Managing the Infrastructure

We utilize a centralized `manager.py` script that acts as a robust orchestrator. It seamlessly performs pre-flight validation checks (ensuring Docker is alive and networking ports are available) before it interacts with Docker Compose, and finishes by assuring database collections and indexes are initialized properly.

Execute all commands from the **root of the repository** replacing standard `python` with `uv run python` if you are using `uv`.

### 1. Starting the Stack

Starts the MongoDB and Mongo Express containers in the background, waits for them to become healthy, and initializes all necessary database collections and indexes automatically.

```bash
python -m infrastructure.manager start
```
*(If you are using `uv`):*
```bash
uv run python -m infrastructure.manager start
```

### 2. Stopping the Stack

Gracefully spins down and stops the running Docker Compose containers. Data volumes are preserved.

```bash
python -m infrastructure.manager stop
```

### 3. Restarting the Stack

A convenience command that issues a `stop` followed immediately by a `start`.

```bash
python -m infrastructure.manager restart
```

### 4. Viewing Status

Displays the current status of the `docker compose ps` command.

```bash
python -m infrastructure.manager status
```

### 5. Tailing Logs

Follows the live logs from the database containers. Press `Ctrl+C` to detach.

```bash
python -m infrastructure.manager logs
```

## Architecture Details

*   **`docker-compose.yml`**: The declarative source of truth for local container definitions.
*   **`manager.py`**: The CLI orchestrator designed to make running infrastructure foolproof.
*   **`validator.py`**: A suite of robust Python pre-flight checks designed to catch common initialization errors (like Docker daemon offline or Ports heavily bound) and provide readable error messages to developers.
*   **`database_init.py`**: The idempotent database migration script that connects to the healthy MongoDB instance and ensures `medical_guidelines` and `drugs` collections exist with their requisite full-text search and unique hashing indexes.
