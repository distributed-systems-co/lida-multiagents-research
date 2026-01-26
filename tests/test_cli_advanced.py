"""
Tests for advanced CLI features.

Tests cover:
- Configuration profiles
- Service orchestration
- Cluster management
- Pipeline execution
- Metrics collection
- Distributed locking
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def run_cli(*args, timeout=30):
    """Run a CLI command and return the result."""
    cmd = [sys.executable, "-m", "src.cli.main"] + list(args)
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result


# =============================================================================
# Test Profile Commands
# =============================================================================

class TestProfileCommands:
    def test_profile_help(self):
        """Test profile --help output."""
        result = run_cli("profile", "--help")
        assert result.returncode == 0
        assert "create" in result.stdout
        assert "list" in result.stdout
        assert "use" in result.stdout
        assert "env" in result.stdout

    def test_profile_list_empty(self):
        """Test profile list with no profiles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env["HOME"] = tmpdir  # Use temp dir for config
            result = subprocess.run(
                [sys.executable, "-m", "src.cli.main", "profile", "list"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                env=env,
            )
            assert "No profiles" in result.stdout or "profile create" in result.stdout

    def test_profile_create_and_show(self):
        """Test creating and showing a profile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env["HOME"] = tmpdir

            # Create profile
            result = subprocess.run(
                [sys.executable, "-m", "src.cli.main", "profile", "create", "test",
                 "--redis-port", "6380", "--api-port", "2041"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                env=env,
            )
            assert result.returncode == 0
            assert "Created profile" in result.stdout

            # Show profile
            result = subprocess.run(
                [sys.executable, "-m", "src.cli.main", "profile", "show", "test"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                env=env,
            )
            assert result.returncode == 0
            assert "6380" in result.stdout
            assert "2041" in result.stdout

    def test_profile_env_export(self):
        """Test profile env export format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env["HOME"] = tmpdir

            # Create profile
            subprocess.run(
                [sys.executable, "-m", "src.cli.main", "profile", "create", "envtest",
                 "--redis-port", "6381", "--api-port", "2042"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                env=env,
            )

            # Export env
            result = subprocess.run(
                [sys.executable, "-m", "src.cli.main", "profile", "env", "envtest"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                env=env,
            )
            assert result.returncode == 0
            assert "export REDIS_PORT=6381" in result.stdout
            assert "export API_PORT=2042" in result.stdout
            assert "eval" in result.stdout


# =============================================================================
# Test Orchestrate Commands
# =============================================================================

class TestOrchestrateCommands:
    def test_orchestrate_help(self):
        """Test orchestrate --help output."""
        result = run_cli("orchestrate", "--help")
        assert result.returncode == 0
        assert "start" in result.stdout
        assert "stop" in result.stdout
        assert "status" in result.stdout
        assert "dependency" in result.stdout.lower()

    def test_orchestrate_status(self):
        """Test orchestrate status command."""
        result = run_cli("orchestrate", "status")
        assert result.returncode == 0
        assert "Service Status" in result.stdout or "redis" in result.stdout.lower()


# =============================================================================
# Test Cluster Commands
# =============================================================================

class TestClusterCommands:
    def test_cluster_help(self):
        """Test cluster --help output."""
        result = run_cli("cluster", "--help")
        assert result.returncode == 0
        assert "add" in result.stdout
        assert "remove" in result.stdout
        assert "status" in result.stdout
        assert "deploy" in result.stdout

    def test_cluster_list_empty(self):
        """Test cluster list with no nodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env["HOME"] = tmpdir
            result = subprocess.run(
                [sys.executable, "-m", "src.cli.main", "cluster", "list"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                env=env,
            )
            assert "No nodes" in result.stdout or "cluster add" in result.stdout

    def test_cluster_add_node(self):
        """Test adding a cluster node."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env["HOME"] = tmpdir

            result = subprocess.run(
                [sys.executable, "-m", "src.cli.main", "cluster", "add", "testnode",
                 "--host", "test.example.com", "--user", "testuser"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                env=env,
            )
            assert result.returncode == 0
            assert "Added node" in result.stdout
            assert "testnode" in result.stdout


# =============================================================================
# Test Pipeline Commands
# =============================================================================

class TestPipelineCommands:
    def test_pipeline_help(self):
        """Test pipeline --help output."""
        result = run_cli("pipeline", "--help")
        assert result.returncode == 0
        assert "--preset" in result.stdout
        assert "deploy" in result.stdout
        assert "test" in result.stdout
        assert "ci" in result.stdout
        assert "--dry-run" in result.stdout

    def test_pipeline_dry_run(self):
        """Test pipeline dry run mode."""
        result = run_cli("pipeline", "--preset", "test", "--dry-run")
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout
        assert "lint" in result.stdout.lower()
        assert "test" in result.stdout.lower()

    def test_pipeline_deploy_dry_run(self):
        """Test deploy pipeline dry run."""
        result = run_cli("pipeline", "--preset", "deploy", "--dry-run")
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout

    def test_pipeline_ci_dry_run(self):
        """Test CI pipeline dry run."""
        result = run_cli("pipeline", "--preset", "ci", "--dry-run")
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout
        assert "install" in result.stdout.lower()


# =============================================================================
# Test Metrics Commands
# =============================================================================

class TestMetricsCommands:
    def test_metrics_help(self):
        """Test metrics --help output."""
        result = run_cli("metrics", "--help")
        assert result.returncode == 0
        assert "show" in result.stdout
        assert "watch" in result.stdout
        assert "export" in result.stdout

    def test_metrics_show(self):
        """Test metrics show command."""
        result = run_cli("metrics", "show")
        assert result.returncode == 0
        assert "Metrics" in result.stdout
        assert "service" in result.stdout.lower()


# =============================================================================
# Test Lock Commands
# =============================================================================

class TestLockCommands:
    def test_lock_help(self):
        """Test lock --help output."""
        result = run_cli("lock", "--help")
        assert result.returncode == 0
        assert "acquire" in result.stdout
        assert "status" in result.stdout
        assert "release" in result.stdout


# =============================================================================
# Test Advanced Module Directly
# =============================================================================

class TestAdvancedModule:
    def test_profile_manager_import(self):
        """Test ProfileManager can be imported."""
        from src.cli.advanced import ProfileManager
        assert ProfileManager is not None

    def test_service_orchestrator_import(self):
        """Test ServiceOrchestrator can be imported."""
        from src.cli.advanced import ServiceOrchestrator
        assert ServiceOrchestrator is not None

    def test_pipeline_import(self):
        """Test Pipeline can be imported."""
        from src.cli.advanced import Pipeline, PipelineStep
        assert Pipeline is not None
        assert PipelineStep is not None

    def test_metrics_collector_import(self):
        """Test MetricsCollector can be imported."""
        from src.cli.advanced import MetricsCollector
        assert MetricsCollector is not None

    def test_profile_manager_operations(self):
        """Test ProfileManager basic operations."""
        from src.cli.advanced import ProfileManager

        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProfileManager(Path(tmpdir))

            # Create profile
            profile = pm.create("test", redis_port=6380, api_port=2041)
            assert profile.name == "test"
            assert profile.redis_port == 6380

            # List profiles
            profiles = pm.list()
            assert len(profiles) == 1

            # Get profile
            fetched = pm.get("test")
            assert fetched is not None
            assert fetched.api_port == 2041

            # Delete profile
            assert pm.delete("test")
            assert pm.get("test") is None

    def test_pipeline_execution(self):
        """Test Pipeline execution."""
        from src.cli.advanced import Pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = Pipeline("test", Path(tmpdir))
            pipeline.add("echo", ["echo", "hello"])

            success, results = pipeline.run()
            assert success
            assert len(results) == 1
            assert results[0].success
            assert "hello" in results[0].stdout

    def test_metrics_collector_operations(self):
        """Test MetricsCollector basic operations."""
        from src.cli.advanced import MetricsCollector

        collector = MetricsCollector()

        # Record metric
        collector.record("test.metric", 42.0, {"label": "value"})

        # Get latest
        latest = collector.get_latest("test.metric")
        assert latest is not None
        assert latest.value == 42.0
        assert latest.labels["label"] == "value"

        # Get stats
        collector.record("test.metric", 50.0)
        collector.record("test.metric", 38.0)
        stats = collector.get_stats("test.metric")
        assert stats["count"] == 3
        assert stats["min"] == 38.0
        assert stats["max"] == 50.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestAdvancedIntegration:
    def test_all_advanced_commands_have_help(self):
        """Test all advanced commands have help."""
        commands = ["profile", "orchestrate", "cluster", "pipeline", "metrics", "lock"]

        for cmd in commands:
            result = run_cli(cmd, "--help")
            assert result.returncode == 0, f"Command {cmd} --help failed"

    def test_profile_workflow(self):
        """Test complete profile workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env["HOME"] = tmpdir

            # Create profile
            result = subprocess.run(
                [sys.executable, "-m", "src.cli.main", "profile", "create", "workflow",
                 "--redis-port", "6382", "--api-port", "2043", "--workers", "8"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                env=env,
            )
            assert result.returncode == 0

            # Set as default
            result = subprocess.run(
                [sys.executable, "-m", "src.cli.main", "profile", "use", "workflow"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                env=env,
            )
            assert result.returncode == 0

            # List shows profile
            result = subprocess.run(
                [sys.executable, "-m", "src.cli.main", "profile", "list"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                env=env,
            )
            assert "workflow" in result.stdout
            assert "6382" in result.stdout

            # Delete profile
            result = subprocess.run(
                [sys.executable, "-m", "src.cli.main", "profile", "delete", "workflow"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                env=env,
            )
            assert result.returncode == 0
            assert "Deleted" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
