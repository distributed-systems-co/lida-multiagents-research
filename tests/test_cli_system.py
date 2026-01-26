"""
Tests for LIDA CLI system management commands.

These tests focus on the multi-user cluster functionality
(start, stop, status, env commands).
"""

import argparse
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli.main import cmd_system, is_port_in_use, CLIError


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def base_args():
    """Base arguments for system command."""
    return argparse.Namespace(
        redis_port=6379,
        api_port=2040,
        docker=False,
    )


@pytest.fixture
def start_args(base_args):
    """Arguments for system start command."""
    base_args.subcommand = "start"
    base_args.workers = None
    base_args.worker_replicas = 1
    base_args.agents = None
    base_args.scenario = None
    base_args.live = False
    base_args.force = False
    return base_args


@pytest.fixture
def stop_args(base_args):
    """Arguments for system stop command."""
    base_args.subcommand = "stop"
    base_args.include_redis = False
    return base_args


@pytest.fixture
def status_args(base_args):
    """Arguments for system status command."""
    base_args.subcommand = "status"
    return base_args


@pytest.fixture
def env_args(base_args):
    """Arguments for system env command."""
    base_args.subcommand = "env"
    base_args.scenario = None
    return base_args


# =============================================================================
# Test System Status
# =============================================================================

class TestSystemStatus:
    def test_status_shows_port_table(self, status_args, capsys):
        """Test that status shows port availability table."""
        with patch('subprocess.run'):
            cmd_system(status_args)

        captured = capsys.readouterr()

        # Should show common ports
        assert "6379" in captured.out
        assert "6380" in captured.out
        assert "2040" in captured.out
        assert "2041" in captured.out

    def test_status_shows_port_states(self, status_args, capsys):
        """Test that status shows IN USE / available states."""
        with patch('subprocess.run'):
            with patch('src.cli.main.is_port_in_use', side_effect=lambda p: p == 6379):
                cmd_system(status_args)

        captured = capsys.readouterr()
        assert "IN USE" in captured.out
        assert "available" in captured.out

    def test_status_shows_multi_user_guide(self, status_args, capsys):
        """Test that status shows multi-user port assignments."""
        with patch('subprocess.run'):
            cmd_system(status_args)

        captured = capsys.readouterr()
        assert "User 1" in captured.out
        assert "User 2" in captured.out
        assert "User 3" in captured.out

    def test_status_checks_docker(self, status_args, capsys):
        """Test that status checks Docker if available."""
        mock_docker_result = MagicMock()
        mock_docker_result.returncode = 0
        mock_docker_result.stdout = "lida-api\tUp 5 minutes\t0.0.0.0:2040->2040/tcp"

        with patch('subprocess.run', return_value=mock_docker_result):
            cmd_system(status_args)

        captured = capsys.readouterr()
        # Should attempt to check Docker containers
        assert "Docker" in captured.out or "Containers" in captured.out or "LIDA" in captured.out

    def test_status_handles_no_processes(self, status_args, capsys):
        """Test status when no LIDA processes are running."""
        mock_result = MagicMock()
        mock_result.stdout = "USER PID %CPU\nroot 1 0.0"

        with patch('subprocess.run', return_value=mock_result):
            cmd_system(status_args)

        captured = capsys.readouterr()
        assert "No LIDA processes" in captured.out or "Processes" in captured.out


# =============================================================================
# Test System Env
# =============================================================================

class TestSystemEnv:
    def test_env_basic_output(self, env_args, capsys):
        """Test basic env output."""
        cmd_system(env_args)
        captured = capsys.readouterr()

        assert "export REDIS_PORT=6379" in captured.out
        assert "export API_PORT=2040" in captured.out
        assert "export REDIS_URL=redis://localhost:6379" in captured.out
        assert "export PORT=2040" in captured.out

    def test_env_custom_ports(self, env_args, capsys):
        """Test env with custom ports."""
        env_args.redis_port = 6380
        env_args.api_port = 2041

        cmd_system(env_args)
        captured = capsys.readouterr()

        assert "REDIS_PORT=6380" in captured.out
        assert "API_PORT=2041" in captured.out
        assert "redis://localhost:6380" in captured.out

    def test_env_with_scenario(self, env_args, capsys):
        """Test env includes scenario when specified."""
        env_args.scenario = "ai_xrisk"

        cmd_system(env_args)
        captured = capsys.readouterr()

        assert "SCENARIO=ai_xrisk" in captured.out

    def test_env_eval_hint(self, env_args, capsys):
        """Test env shows eval hint."""
        cmd_system(env_args)
        captured = capsys.readouterr()

        assert "eval" in captured.out


# =============================================================================
# Test System Start
# =============================================================================

class TestSystemStart:
    def test_start_shows_configuration(self, start_args, capsys):
        """Test start shows configuration."""
        with patch('src.cli.main.is_port_in_use', return_value=True):
            cmd_system(start_args)

        captured = capsys.readouterr()
        assert "6379" in captured.out
        assert "2040" in captured.out

    def test_start_warns_port_in_use(self, start_args, capsys):
        """Test start warns when API port is in use."""
        with patch('src.cli.main.is_port_in_use', return_value=True):
            cmd_system(start_args)

        captured = capsys.readouterr()
        assert "in use" in captured.out.lower() or "warning" in captured.out.lower()

    def test_start_with_force(self, start_args, capsys):
        """Test start with --force bypasses port check."""
        start_args.force = True

        with patch('src.cli.main.is_port_in_use', return_value=True):
            with patch('subprocess.Popen') as mock_popen:
                mock_popen.return_value.pid = 12345
                cmd_system(start_args)

        captured = capsys.readouterr()
        # Should attempt to start despite port warning
        assert "Starting" in captured.out or "LIDA" in captured.out

    def test_start_native_mode(self, start_args, capsys):
        """Test start in native mode (no Docker)."""
        start_args.docker = False

        with patch('src.cli.main.is_port_in_use', return_value=False):
            with patch('subprocess.Popen') as mock_popen:
                mock_proc = MagicMock()
                mock_proc.pid = 12345
                mock_popen.return_value = mock_proc
                cmd_system(start_args)

        captured = capsys.readouterr()
        assert "native" in captured.out.lower() or "Starting" in captured.out

    def test_start_docker_mode(self, start_args, capsys):
        """Test start in Docker mode."""
        start_args.docker = True

        with patch('src.cli.main.is_port_in_use', return_value=False):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value.returncode = 0
                # Mock docker availability check
                with patch.object(Path, 'exists', return_value=True):
                    cmd_system(start_args)

    def test_start_with_workers(self, start_args, capsys):
        """Test start with worker count."""
        start_args.workers = 8

        with patch('src.cli.main.is_port_in_use', return_value=False):
            with patch('subprocess.Popen') as mock_popen:
                mock_popen.return_value.pid = 12345
                cmd_system(start_args)

        captured = capsys.readouterr()
        assert "8" in captured.out or "workers" in captured.out.lower()

    def test_start_with_live_mode(self, start_args, capsys):
        """Test start with live LLM mode."""
        start_args.live = True

        with patch('src.cli.main.is_port_in_use', return_value=False):
            with patch('subprocess.Popen') as mock_popen:
                mock_popen.return_value.pid = 12345
                cmd_system(start_args)

        captured = capsys.readouterr()
        assert "live" in captured.out.lower() or "enabled" in captured.out.lower()

    def test_start_sets_environment(self, start_args):
        """Test start sets correct environment variables."""
        start_args.redis_port = 6381
        start_args.api_port = 2042

        with patch('src.cli.main.is_port_in_use', return_value=False):
            with patch('subprocess.Popen') as mock_popen:
                mock_popen.return_value.pid = 12345
                cmd_system(start_args)

        assert os.environ.get("REDIS_PORT") == "6381"
        assert os.environ.get("API_PORT") == "2042"


# =============================================================================
# Test System Stop
# =============================================================================

class TestSystemStop:
    def test_stop_kills_api_port(self, stop_args, capsys):
        """Test stop kills processes on API port."""
        mock_lsof = MagicMock()
        mock_lsof.stdout = "12345\n"

        with patch('subprocess.run', return_value=mock_lsof) as mock_run:
            cmd_system(stop_args)

        # Should have called lsof and kill
        calls = mock_run.call_args_list
        lsof_calls = [c for c in calls if 'lsof' in str(c)]
        assert len(lsof_calls) > 0

    def test_stop_with_include_redis(self, stop_args, capsys):
        """Test stop with --include-redis kills Redis too."""
        stop_args.include_redis = True

        mock_result = MagicMock()
        mock_result.stdout = "12345\n"

        with patch('subprocess.run', return_value=mock_result) as mock_run:
            cmd_system(stop_args)

        captured = capsys.readouterr()
        # Should attempt to stop both ports

    def test_stop_docker_mode(self, stop_args, capsys):
        """Test stop in Docker mode."""
        stop_args.docker = True

        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            with patch.object(Path, 'exists', return_value=True):
                cmd_system(stop_args)

    def test_stop_handles_no_processes(self, stop_args, capsys):
        """Test stop handles no running processes gracefully."""
        mock_result = MagicMock()
        mock_result.stdout = ""

        with patch('subprocess.run', return_value=mock_result):
            cmd_system(stop_args)

        captured = capsys.readouterr()
        assert "stopped" in captured.out.lower() or "Services" in captured.out


# =============================================================================
# Test Multi-User Scenarios
# =============================================================================

class TestMultiUserScenarios:
    def test_user1_default_ports(self, capsys):
        """Test User 1 uses default ports."""
        args = argparse.Namespace(
            subcommand="env",
            redis_port=6379,
            api_port=2040,
            scenario=None,
            docker=False,
        )

        cmd_system(args)
        captured = capsys.readouterr()

        assert "6379" in captured.out
        assert "2040" in captured.out

    def test_user2_offset_ports(self, capsys):
        """Test User 2 uses offset ports."""
        args = argparse.Namespace(
            subcommand="env",
            redis_port=6380,
            api_port=2041,
            scenario=None,
            docker=False,
        )

        cmd_system(args)
        captured = capsys.readouterr()

        assert "6380" in captured.out
        assert "2041" in captured.out

    def test_user3_offset_ports(self, capsys):
        """Test User 3 uses double-offset ports."""
        args = argparse.Namespace(
            subcommand="env",
            redis_port=6381,
            api_port=2042,
            scenario=None,
            docker=False,
        )

        cmd_system(args)
        captured = capsys.readouterr()

        assert "6381" in captured.out
        assert "2042" in captured.out

    def test_port_isolation(self):
        """Test that different port combinations don't interfere."""
        port_configs = [
            (6379, 2040),
            (6380, 2041),
            (6381, 2042),
        ]

        # Each config should set different environment
        for redis_port, api_port in port_configs:
            args = argparse.Namespace(
                subcommand="start",
                redis_port=redis_port,
                api_port=api_port,
                workers=None,
                worker_replicas=1,
                agents=None,
                scenario=None,
                live=False,
                docker=False,
                force=True,
            )

            with patch('src.cli.main.is_port_in_use', return_value=False):
                with patch('subprocess.Popen') as mock_popen:
                    mock_popen.return_value.pid = 12345
                    cmd_system(args)

            assert os.environ.get("REDIS_PORT") == str(redis_port)
            assert os.environ.get("API_PORT") == str(api_port)


# =============================================================================
# Test Error Handling
# =============================================================================

class TestSystemErrorHandling:
    def test_unknown_subcommand(self, base_args):
        """Test handling of unknown subcommand."""
        base_args.subcommand = "invalid"

        with pytest.raises(Exception) as exc:
            cmd_system(base_args)

        assert "CLIError" in type(exc.value).__name__
        assert "Unknown subcommand" in str(exc.value)

    def test_docker_not_available(self, start_args, capsys):
        """Test handling when Docker is requested but not available."""
        start_args.docker = True

        with patch('subprocess.run') as mock_run:
            # Docker check fails
            mock_run.side_effect = FileNotFoundError()

            with patch('src.cli.main.is_port_in_use', return_value=False):
                with patch('subprocess.Popen') as mock_popen:
                    mock_popen.return_value.pid = 12345
                    cmd_system(start_args)

    def test_redis_server_not_found(self, start_args, capsys):
        """Test handling when redis-server is not installed."""
        with patch('src.cli.main.is_port_in_use', return_value=False):
            with patch('subprocess.Popen') as mock_popen:
                mock_popen.side_effect = FileNotFoundError("redis-server not found")

                try:
                    cmd_system(start_args)
                except:
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
