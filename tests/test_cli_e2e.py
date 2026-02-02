"""
End-to-end tests for LIDA CLI.

These tests run actual CLI commands via subprocess to verify
real-world behavior.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def run_cli(*args, timeout=30, check=False, env=None):
    """Run a CLI command and return the result."""
    cmd = [sys.executable, "-m", "src.cli.main"] + list(args)

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=merged_env,
    )

    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )

    return result


# =============================================================================
# Test Help Commands
# =============================================================================

class TestHelpCommands:
    def test_main_help(self):
        """Test main --help output."""
        result = run_cli("--help")
        assert result.returncode == 0
        assert "LIDA" in result.stdout
        assert "system" in result.stdout
        assert "serve" in result.stdout
        assert "deliberate" in result.stdout

    def test_system_help(self):
        """Test system --help output."""
        result = run_cli("system", "--help")
        assert result.returncode == 0
        assert "start" in result.stdout
        assert "stop" in result.stdout
        assert "status" in result.stdout
        assert "env" in result.stdout

    def test_serve_help(self):
        """Test serve --help output."""
        result = run_cli("serve", "--help")
        assert result.returncode == 0
        assert "--port" in result.stdout
        assert "--host" in result.stdout
        assert "--live" in result.stdout

    def test_workers_help(self):
        """Test workers --help output."""
        result = run_cli("workers", "--help")
        assert result.returncode == 0
        assert "--count" in result.stdout
        assert "--redis-url" in result.stdout
        assert "--capacity" in result.stdout

    def test_quorum_help(self):
        """Test quorum --help output."""
        result = run_cli("quorum", "--help")
        assert result.returncode == 0
        assert "--preset" in result.stdout
        assert "realtime" in result.stdout
        assert "gdelt" in result.stdout

    def test_debate_help(self):
        """Test debate --help output."""
        result = run_cli("debate", "--help")
        assert result.returncode == 0
        assert "--topic" in result.stdout
        assert "--matchup" in result.stdout
        assert "doom_vs_accel" in result.stdout

    def test_demo_help(self):
        """Test demo --help output."""
        result = run_cli("demo", "--help")
        assert result.returncode == 0
        assert "--type" in result.stdout
        assert "quick" in result.stdout
        assert "swarm" in result.stdout

    def test_deliberate_help(self):
        """Test deliberate --help output."""
        result = run_cli("deliberate", "--help")
        assert result.returncode == 0
        assert "--port" in result.stdout
        assert "--topic" in result.stdout


# =============================================================================
# Test System Commands
# =============================================================================

class TestSystemCommands:
    def test_system_status(self):
        """Test system status command."""
        result = run_cli("system", "status")
        assert result.returncode == 0
        assert "Port Status" in result.stdout
        assert "6379" in result.stdout
        assert "2040" in result.stdout

    def test_system_env_default(self):
        """Test system env with default ports."""
        result = run_cli("system", "env")
        assert result.returncode == 0
        assert "export REDIS_PORT=6379" in result.stdout
        assert "export API_PORT=2040" in result.stdout

    def test_system_env_custom_ports(self):
        """Test system env with custom ports."""
        result = run_cli("system", "env", "--redis-port", "6380", "--api-port", "2041")
        assert result.returncode == 0
        assert "REDIS_PORT=6380" in result.stdout
        assert "API_PORT=2041" in result.stdout

    def test_system_env_with_scenario(self):
        """Test system env with scenario."""
        result = run_cli("system", "env", "--scenario", "test_scenario")
        assert result.returncode == 0
        assert "SCENARIO=test_scenario" in result.stdout


# =============================================================================
# Test Debate Commands
# =============================================================================

class TestDebateCommands:
    def test_debate_list(self):
        """Test debate --list shows topics and matchups."""
        result = run_cli("debate", "--list")
        assert result.returncode == 0
        assert "ai_pause" in result.stdout
        assert "doom_vs_accel" in result.stdout
        assert "Matchups" in result.stdout

    def test_debate_topics_present(self):
        """Test all expected debate topics are listed."""
        result = run_cli("debate", "--list")
        expected_topics = [
            "ai_pause",
            "lab_self_regulation",
            "xrisk_vs_present_harms",
            "scaling_hypothesis",
            "open_source_ai",
            "government_regulation",
        ]
        for topic in expected_topics:
            assert topic in result.stdout, f"Missing topic: {topic}"

    def test_debate_matchups_present(self):
        """Test all expected matchups are listed."""
        result = run_cli("debate", "--list")
        expected_matchups = [
            "doom_vs_accel",
            "labs_debate",
            "academics_clash",
            "ethics_vs_scale",
            "full_panel",
        ]
        for matchup in expected_matchups:
            assert matchup in result.stdout, f"Missing matchup: {matchup}"


# =============================================================================
# Test Demo Types
# =============================================================================

class TestDemoTypes:
    def test_demo_invalid_type(self):
        """Test demo with invalid type shows available types."""
        result = run_cli("demo", "--type", "invalid_type")
        assert result.returncode != 0
        # Should show available types in error

    def test_demo_types_in_help(self):
        """Test all demo types are in help."""
        result = run_cli("demo", "--help")
        expected_types = ["quick", "live", "streaming", "swarm", "persuasion", "hyperdash"]
        for demo_type in expected_types:
            assert demo_type in result.stdout, f"Missing demo type: {demo_type}"


# =============================================================================
# Test Quorum Presets
# =============================================================================

class TestQuorumPresets:
    def test_quorum_presets_in_help(self):
        """Test all quorum presets are documented."""
        result = run_cli("quorum", "--help")
        expected_presets = ["realtime", "gdelt", "mlx", "openrouter", "advanced"]
        for preset in expected_presets:
            assert preset in result.stdout, f"Missing preset: {preset}"


# =============================================================================
# Test Error Cases
# =============================================================================

class TestErrorCases:
    def test_unknown_command(self):
        """Test unknown command shows help."""
        result = run_cli("unknown_command")
        # Should fail and show usage
        assert result.returncode != 0 or "usage" in result.stdout.lower()

    def test_missing_required_arg(self):
        """Test missing required argument."""
        result = run_cli("deliberate")  # Missing --port
        assert result.returncode != 0
        assert "--port" in result.stderr or "required" in result.stderr.lower()

    def test_invalid_port_type(self):
        """Test invalid port type."""
        result = run_cli("serve", "--port", "not_a_number")
        assert result.returncode != 0


# =============================================================================
# Test Output Format
# =============================================================================

class TestOutputFormat:
    def test_system_status_has_headers(self):
        """Test system status has proper section headers."""
        result = run_cli("system", "status")
        assert "=" in result.stdout  # Header dividers
        assert "-" in result.stdout  # Section dividers

    def test_help_has_examples(self):
        """Test help includes examples."""
        result = run_cli("--help")
        assert "lida" in result.stdout
        # Should have example commands

    def test_system_help_has_examples(self):
        """Test system help has multi-user examples."""
        result = run_cli("system", "--help")
        assert "6379" in result.stdout
        assert "6380" in result.stdout


# =============================================================================
# Test Version
# =============================================================================

class TestVersion:
    def test_version_flag(self):
        """Test -v flag shows version."""
        result = run_cli("-v")
        assert result.returncode == 0
        assert "LIDA" in result.stdout or "0.1.0" in result.stdout


# =============================================================================
# Test Command Chaining (eval)
# =============================================================================

class TestCommandChaining:
    def test_env_output_is_valid_shell(self):
        """Test that env output can be eval'd in shell."""
        result = run_cli("system", "env", "--redis-port", "6379", "--api-port", "2040")
        assert result.returncode == 0

        # Each line should be valid export statement
        for line in result.stdout.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                assert line.startswith('export '), f"Invalid export: {line}"


# =============================================================================
# Test Performance
# =============================================================================

class TestPerformance:
    def test_help_is_fast(self):
        """Test --help responds quickly."""
        start = time.time()
        result = run_cli("--help")
        elapsed = time.time() - start

        assert result.returncode == 0
        assert elapsed < 3.0, f"Help took {elapsed:.2f}s (should be < 3s)"

    def test_system_status_is_fast(self):
        """Test system status is reasonably fast."""
        start = time.time()
        result = run_cli("system", "status")
        elapsed = time.time() - start

        assert result.returncode == 0
        assert elapsed < 5.0, f"Status took {elapsed:.2f}s (should be < 5s)"

    def test_debate_list_is_fast(self):
        """Test debate --list is fast."""
        start = time.time()
        result = run_cli("debate", "--list")
        elapsed = time.time() - start

        assert result.returncode == 0
        assert elapsed < 3.0, f"Debate list took {elapsed:.2f}s"


# =============================================================================
# Test Environment Variables
# =============================================================================

class TestEnvironmentVariables:
    def test_respects_redis_url_env(self):
        """Test that REDIS_URL environment is respected."""
        result = run_cli("workers", "--help")
        assert "redis://localhost:6379" in result.stdout  # default mentioned

    def test_env_command_format(self):
        """Test env command outputs proper format."""
        result = run_cli("system", "env")

        # Should be sourceable
        lines = [l for l in result.stdout.split('\n') if l.strip() and not l.startswith('#')]
        for line in lines:
            if line:
                assert '=' in line, f"Invalid env line: {line}"


# =============================================================================
# Integration Smoke Tests
# =============================================================================

class TestIntegrationSmoke:
    """Quick smoke tests to verify basic functionality."""

    def test_cli_loads(self):
        """Test CLI module loads without errors."""
        result = run_cli("--help")
        assert result.returncode == 0

    def test_all_commands_have_help(self):
        """Test all major commands have help."""
        commands = [
            "run", "simulate", "quorum", "debate", "demo",
            "workers", "serve", "system", "deliberate",
        ]

        for cmd in commands:
            result = run_cli(cmd, "--help")
            assert result.returncode == 0, f"Command {cmd} --help failed"

    def test_no_import_errors(self):
        """Test no import errors on load."""
        result = subprocess.run(
            [sys.executable, "-c", "from src.cli.main import main; print('OK')"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Import failed: {result.stderr}"
        assert "OK" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
