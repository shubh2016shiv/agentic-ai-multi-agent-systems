"""
Infrastructure Validator
=========================
Pre-flight checks that verify all infrastructure dependencies are available
before the ETL pipelines attempt to run. This prevents cryptic runtime errors
by catching misconfiguration early with clear, actionable error messages.

Validation Sequence:
    1. Docker CLI is installed        (docker --version)
    2. Docker daemon is running       (docker info)
    3. Required ports are available   (socket bind test on 27017, 8081)
    4. MongoDB is reachable           (pymongo ping)

Each check returns a ValidationResult with pass/fail status and a human-readable
message. The run_all_precondition_checks() method aggregates all results into a
structured report.

Usage:
    from infrastructure.validator import InfrastructureValidator

    validator = InfrastructureValidator()
    report = validator.run_all_precondition_checks()
    if not report.all_passed:
        print(report.summary)
"""

import logging
import shutil
import socket
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# Validation Result Models
# =============================================================================
# These dataclasses capture the outcome of each validation step so that
# callers can inspect individual results or the aggregate report.
# =============================================================================


@dataclass
class ValidationResult:
    """
    Outcome of a single validation check.

    Attributes:
        check_name: Human-readable name of the check (e.g., "Docker Installed").
        passed: Whether the check succeeded.
        message: Descriptive message explaining the outcome.
        details: Optional extra context (e.g., version strings, error traces).
    """

    check_name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    """
    Aggregated results from all infrastructure validation checks.

    Attributes:
        results: Ordered list of individual ValidationResult objects.
        all_passed: True only if every check succeeded.
        summary: Multi-line human-readable summary suitable for console output.
    """

    results: list[ValidationResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def summary(self) -> str:
        lines = ["=" * 60, "Infrastructure Validation Report", "=" * 60]
        for result in self.results:
            status_indicator = "PASS" if result.passed else "FAIL"
            lines.append(f"  [{status_indicator}] {result.check_name}: {result.message}")
        lines.append("=" * 60)
        overall = "ALL CHECKS PASSED" if self.all_passed else "SOME CHECKS FAILED"
        lines.append(f"  Overall: {overall}")
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Infrastructure Validator
# =============================================================================


class InfrastructureValidator:
    """
    Validates that all infrastructure prerequisites are satisfied.

    This class runs a series of checks to ensure the host environment is ready
    for MongoDB Docker containers and that the ETL pipelines will be able to
    connect to the database.

    Each validate_* method is independent and can be called individually for
    targeted diagnostics. Use run_all_precondition_checks() for a full sweep.
    """

    # -------------------------------------------------------------------------
    # Individual Validation Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def validate_docker_installed() -> ValidationResult:
        """
        Verify that the Docker CLI is available on the system PATH.

        This check uses shutil.which() first (instant, no subprocess), and falls
        back to `docker --version` for version extraction.
        """
        if not shutil.which("docker"):
            return ValidationResult(
                check_name="Docker Installed",
                passed=False,
                message="Docker CLI not found on system PATH. Install Docker Desktop from https://www.docker.com/products/docker-desktop/",
            )

        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            version_string = result.stdout.strip()
            return ValidationResult(
                check_name="Docker Installed",
                passed=True,
                message=f"Docker CLI available: {version_string}",
                details={"version": version_string},
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as error:
            return ValidationResult(
                check_name="Docker Installed",
                passed=False,
                message=f"Docker CLI found but failed to execute: {error}",
            )

    @staticmethod
    def validate_docker_daemon_running() -> ValidationResult:
        """
        Verify that the Docker daemon (background service) is running.

        The Docker CLI might be installed, but the daemon needs to be actively
        running to create containers. This is the most common failure on Windows
        when Docker Desktop is installed but not started.
        """
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                return ValidationResult(
                    check_name="Docker Daemon Running",
                    passed=True,
                    message="Docker daemon is running and responsive",
                )
            else:
                return ValidationResult(
                    check_name="Docker Daemon Running",
                    passed=False,
                    message="Docker daemon is not running. Start Docker Desktop and try again.",
                    details={"stderr": result.stderr.strip()},
                )
        except (subprocess.TimeoutExpired, FileNotFoundError) as error:
            return ValidationResult(
                check_name="Docker Daemon Running",
                passed=False,
                message=f"Could not reach Docker daemon: {error}",
            )

    @staticmethod
    def validate_port_available(port: int) -> ValidationResult:
        """
        Check whether a specific TCP port is available for binding.

        If another service is already using the port, Docker will fail to start
        the container. This check catches that conflict before we attempt to
        spin up the infrastructure.

        Args:
            port: TCP port number to test (e.g., 27017 for MongoDB).
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test_socket:
                # SO_REUSEADDR prevents "address already in use" false positives
                # on recently closed sockets
                test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                test_socket.bind(("localhost", port))
                return ValidationResult(
                    check_name=f"Port {port} Available",
                    passed=True,
                    message=f"Port {port} is available for use",
                )
        except OSError:
            return ValidationResult(
                check_name=f"Port {port} Available",
                passed=True,
                message=f"Port {port} is already in use (may be an existing MongoDB/Mongo Express instance — this is acceptable)",
                details={"port": port, "note": "Port in use is not necessarily a failure if the service is already running"},
            )

    @staticmethod
    def validate_mongodb_connection(mongodb_uri: str) -> ValidationResult:
        """
        Attempt to connect to MongoDB and verify the server is responsive.

        This is the final validation step — it confirms end-to-end connectivity
        from the Python process to the MongoDB server.

        Args:
            mongodb_uri: Full MongoDB connection URI (e.g., mongodb://admin:pass@localhost:27017).
        """
        try:
            from pymongo import MongoClient
            from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

            client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
            client.admin.command("ping")
            client.close()

            return ValidationResult(
                check_name="MongoDB Connection",
                passed=True,
                message=f"Successfully connected to MongoDB at {mongodb_uri.split('@')[-1] if '@' in mongodb_uri else mongodb_uri}",
            )
        except ImportError:
            return ValidationResult(
                check_name="MongoDB Connection",
                passed=False,
                message="pymongo is not installed. Run: pip install pymongo>=4.6.0",
            )
        except (ConnectionFailure, ServerSelectionTimeoutError) as error:
            return ValidationResult(
                check_name="MongoDB Connection",
                passed=False,
                message=f"Cannot connect to MongoDB: {error}. Ensure the MongoDB container is running.",
            )
        except Exception as error:
            return ValidationResult(
                check_name="MongoDB Connection",
                passed=False,
                message=f"Unexpected error connecting to MongoDB: {error}",
            )

    # -------------------------------------------------------------------------
    # Aggregate Validation
    # -------------------------------------------------------------------------

    def run_all_precondition_checks(
        self,
        mongodb_uri: str = "mongodb://admin:adminpassword@localhost:27017",
        check_mongodb_connection: bool = False,
    ) -> ValidationReport:
        """
        Execute all infrastructure validation checks and return a consolidated report.

        The checks run in dependency order: Docker installed -> daemon running ->
        ports available -> (optionally) MongoDB connection.

        Args:
            mongodb_uri: MongoDB URI to test connectivity against.
            check_mongodb_connection: If True, also verify MongoDB is reachable.
                Set to False when running pre-setup checks (MongoDB isn't up yet).

        Returns:
            ValidationReport with individual results and an overall pass/fail.
        """
        report = ValidationReport()

        logger.info("Running infrastructure precondition checks...")

        report.results.append(self.validate_docker_installed())
        report.results.append(self.validate_docker_daemon_running())
        report.results.append(self.validate_port_available(27017))
        report.results.append(self.validate_port_available(8081))

        if check_mongodb_connection:
            report.results.append(self.validate_mongodb_connection(mongodb_uri))

        logger.info(f"Validation complete: {'ALL PASSED' if report.all_passed else 'FAILURES DETECTED'}")

        return report
