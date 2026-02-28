"""
Infrastructure Manager
======================
CLI script to centralize the starting, stopping, and validation 
of all project infrastructure components.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from packaging import version # type: ignore

from infrastructure.validator import InfrastructureValidator
from infrastructure.database_init import initialize_mongodb_database_and_collections
from core.config import settings

logger = logging.getLogger(__name__)

_INFRASTRUCTURE_DIR = os.path.dirname(os.path.abspath(__file__))
_DOCKER_COMPOSE_FILE = os.path.join(_INFRASTRUCTURE_DIR, "docker-compose.yml")
MONGODB_URI = settings.mongodb_settings.mongodb_uri if hasattr(settings, "mongodb_settings") else "mongodb://admin:adminpassword@localhost:27017"

MONGODB_HEALTH_CHECK_MAX_ATTEMPTS = 30
MONGODB_HEALTH_CHECK_INTERVAL_SECONDS = 2


def run_docker_compose(command: list[str]) -> bool:
    """Execute a docker compose command."""
    
    # Modern docker v2 uses `docker compose` while v1 uses `docker-compose`
    # The validator ensures docker is available, so we assume `docker compose` 
    # as the modern standard.
    base_cmd = ["docker", "compose", "-f", _DOCKER_COMPOSE_FILE]
    full_cmd = base_cmd + command
    
    print(f"\n[MANAGER] Executing: {' '.join(full_cmd)}")
    
    try:
        result = subprocess.run(
            full_cmd,
            # capture_output=False to allow streams to print to terminal directly, preserving colors
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.error("Command timed out.")
        return False
    except FileNotFoundError:
        logger.error("Docker CLI not found.")
        return False

def check_docker_health() -> bool:
     validator = InfrastructureValidator()
     report = validator.run_all_precondition_checks(
         mongodb_uri=MONGODB_URI,
         check_mongodb_connection=False
     )
     print(report.summary)
     docker_checks = [r for r in report.results if "Docker" in r.check_name]
     return all(check.passed for check in docker_checks)


def wait_for_services() -> bool:
    """Poll services until healthy."""
    print(f"\n[MANAGER] Waiting for MongoDB to become healthy...")
    
    for attempt in range(1, MONGODB_HEALTH_CHECK_MAX_ATTEMPTS + 1):
        try:
            from pymongo import MongoClient
            client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
            client.admin.command("ping")
            client.close()
            print(f"[MANAGER] MongoDB is healthy (connected on attempt {attempt})")
            return True
        except Exception:
            if attempt < MONGODB_HEALTH_CHECK_MAX_ATTEMPTS:
                print(f"        Attempt {attempt}/{MONGODB_HEALTH_CHECK_MAX_ATTEMPTS} — "
                      f"retrying in {MONGODB_HEALTH_CHECK_INTERVAL_SECONDS}s...")
                time.sleep(MONGODB_HEALTH_CHECK_INTERVAL_SECONDS)
            else:
                print(f"[MANAGER] ERROR: Infrastructure did not become healthy.")
                return False
    return False

def start_infrastructure():
    if not check_docker_health():
        print("[MANAGER] ABORTED: Docker preconditions not met.")
        sys.exit(1)
        
    if run_docker_compose(["up", "-d"]):
        if wait_for_services():
            if initialize_mongodb_database_and_collections(MONGODB_URI):
                 print("\n" + "=" * 65)
                 print("  Infrastructure Successfully Started & Initialized")
                 print("=" * 65)
            else:
                 print("\n[MANAGER] WARNING: Containers started, but DB initialization failed.")
                 sys.exit(1)
        else:
             print("\n[MANAGER] WARNING: Containers started, but failed health checks.")
             sys.exit(1)
    else:
         print("\n[MANAGER] ERROR: Failed to start Docker Compose services.")
         sys.exit(1)

def stop_infrastructure():
    print("=" * 65)
    print("  Stopping Infrastructure...")
    print("=" * 65)
    
    if run_docker_compose(["down"]):
         print("\n[MANAGER] Infrastructure Successfully Stopped.")
    else:
         print("\n[MANAGER] ERROR: Failed to stop infrastructure.")
         sys.exit(1)
         
def restart_infrastructure():
    stop_infrastructure()
    start_infrastructure()

def main():
    parser = argparse.ArgumentParser(description="Manage project infrastructure.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    subparsers.add_parser("start", help="Start the infrastructure (validates docker, runs compose up, initializes DB)")
    subparsers.add_parser("stop", help="Stop the infrastructure (runs compose down)")
    subparsers.add_parser("restart", help="Restart the infrastructure")
    subparsers.add_parser("status", help="Check the current status of infrastructure services")
    subparsers.add_parser("logs", help="Tail logs from the infrastructure services")

    args = parser.parse_args()

    # Make project package resolvable for imports
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    if args.command == "start":
        start_infrastructure()
    elif args.command == "stop":
        stop_infrastructure()
    elif args.command == "restart":
        restart_infrastructure()
    elif args.command == "status":
        run_docker_compose(["ps"])
    elif args.command == "logs":
        # Follow logs until user kills
        run_docker_compose(["logs", "-f"])

if __name__ == "__main__":
    main()
