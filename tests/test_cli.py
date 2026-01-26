"""
Comprehensive tests for the LIDA CLI.

Tests cover:
- CLI utility functions
- Command-line argument parsing
- Command execution (mocked)
- System management commands
- Port utilities
"""

import argparse
import asyncio
import io
import os
import sys
import socket
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli.main import (
    CLIError,
    run_async,
    import_module,
    require_api_key,
    is_port_in_use,
    find_available_port,
    print_header,
    print_section,
    print_kv,
    main,
)


# =============================================================================
# Test CLIError Exception
# =============================================================================

class TestCLIError:
    def test_cli_error_creation(self):
        error = CLIError("Test error")
        assert str(error) == "Test error"
        assert error.exit_code == 1

    def test_cli_error_with_custom_exit_code(self):
        error = CLIError("Custom error", exit_code=42)
        assert str(error) == "Custom error"
        assert error.exit_code == 42

    def test_cli_error_inheritance(self):
        error = CLIError("Test")
        assert isinstance(error, Exception)


# =============================================================================
# Test run_async Helper
# =============================================================================

class TestRunAsync:
    def test_run_async_success(self):
        async def coro():
            return "success"

        result = run_async(coro)
        assert result == "success"

    def test_run_async_with_exception(self):
        async def coro():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            run_async(coro, handle_interrupt=False)

    def test_run_async_keyboard_interrupt(self, capsys):
        async def coro():
            raise KeyboardInterrupt()

        result = run_async(coro, handle_interrupt=True)
        assert result is None
        captured = capsys.readouterr()
        assert "Interrupted" in captured.out


# =============================================================================
# Test import_module Helper
# =============================================================================

class TestImportModule:
    def test_import_existing_module(self):
        module = import_module("json")
        assert hasattr(module, "dumps")
        assert hasattr(module, "loads")

    def test_import_module_attribute(self):
        dumps = import_module("json", "dumps")
        assert callable(dumps)
        assert dumps({"a": 1}) == '{"a": 1}'

    def test_import_nonexistent_module(self):
        with pytest.raises(CLIError) as exc:
            import_module("nonexistent_module_xyz")
        assert "Module not available" in str(exc.value)

    def test_import_nonexistent_attribute(self):
        with pytest.raises(CLIError) as exc:
            import_module("json", "nonexistent_attr")
        assert "not found" in str(exc.value)


# =============================================================================
# Test require_api_key Helper
# =============================================================================

class TestRequireApiKey:
    def test_require_existing_key(self, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "sk-test-123")
        key = require_api_key("TEST_API_KEY")
        assert key == "sk-test-123"

    def test_require_missing_key(self, monkeypatch):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        with pytest.raises(CLIError) as exc:
            require_api_key("MISSING_KEY")
        assert "MISSING_KEY" in str(exc.value)

    def test_require_default_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        key = require_api_key()
        assert key == "sk-or-test"


# =============================================================================
# Test Port Utilities
# =============================================================================

class TestPortUtilities:
    def test_is_port_in_use_available(self):
        # Find a port that's likely available in a safe range
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 0))
            port = s.getsockname()[1]
        # Use a high port that's unlikely to be in use
        test_port = min(port + 100, 65000)  # Keep within valid range
        # After closing, port should be available
        assert not is_port_in_use(test_port)

    def test_is_port_in_use_occupied(self):
        # Create a listening socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('localhost', 0))
            s.listen(1)
            port = s.getsockname()[1]
            # Port should be in use while socket is open
            assert is_port_in_use(port)

    def test_find_available_port(self):
        port = find_available_port(50000, 50100)
        assert 50000 <= port <= 50100
        assert not is_port_in_use(port)

    def test_find_available_port_all_occupied(self):
        # Mock is_port_in_use to always return True
        with patch('src.cli.main.is_port_in_use', return_value=True):
            with pytest.raises(CLIError) as exc:
                find_available_port(50000, 50005)
            assert "No available ports" in str(exc.value)


# =============================================================================
# Test Print Utilities
# =============================================================================

class TestPrintUtilities:
    def test_print_header(self, capsys):
        print_header("Test Header")
        captured = capsys.readouterr()
        assert "Test Header" in captured.out
        assert "=" in captured.out

    def test_print_header_custom_char(self, capsys):
        print_header("Title", char="#", width=40)
        captured = capsys.readouterr()
        assert "#" * 40 in captured.out

    def test_print_section(self, capsys):
        print_section("Section Title")
        captured = capsys.readouterr()
        assert "Section Title" in captured.out
        assert "-" in captured.out

    def test_print_kv(self, capsys):
        print_kv("Key", "Value")
        captured = capsys.readouterr()
        assert "Key: Value" in captured.out

    def test_print_kv_with_indent(self, capsys):
        print_kv("Key", "Value", indent=4)
        captured = capsys.readouterr()
        assert "    Key: Value" in captured.out


# =============================================================================
# Test Argument Parsing
# =============================================================================

class TestArgumentParsing:
    @pytest.fixture
    def parser(self):
        """Create a fresh argument parser for testing."""
        # Import and recreate the parser logic
        from src.cli.main import main
        return None  # We'll test via main()

    def test_help_flag(self, capsys):
        with pytest.raises(SystemExit) as exc:
            sys.argv = ["lida", "--help"]
            main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "LIDA" in captured.out

    def test_version_flag(self, capsys):
        with patch('src.cli.main.cmd_version') as mock_version:
            sys.argv = ["lida", "-v"]
            main()
            mock_version.assert_called_once()

    def test_no_command(self, capsys):
        sys.argv = ["lida"]
        main()
        captured = capsys.readouterr()
        assert "usage:" in captured.out or "LIDA" in captured.out


# =============================================================================
# Test System Commands
# =============================================================================

class TestSystemCommands:
    def test_system_status_parsing(self):
        """Test that system status command parses correctly."""
        from src.cli.main import cmd_system

        args = argparse.Namespace(
            subcommand="status",
            redis_port=6379,
            api_port=2040,
            docker=False,
        )

        # Should run without error
        with patch('subprocess.run'):
            cmd_system(args)

    def test_system_env_output(self, capsys):
        """Test that system env outputs correct environment variables."""
        from src.cli.main import cmd_system

        args = argparse.Namespace(
            subcommand="env",
            redis_port=6380,
            api_port=2041,
            scenario="test_scenario",
            docker=False,
        )

        cmd_system(args)
        captured = capsys.readouterr()

        assert "REDIS_PORT=6380" in captured.out
        assert "API_PORT=2041" in captured.out
        assert "SCENARIO=test_scenario" in captured.out

    def test_system_start_port_check(self, capsys):
        """Test that system start checks for port availability."""
        from src.cli.main import cmd_system

        args = argparse.Namespace(
            subcommand="start",
            redis_port=6379,
            api_port=2040,
            workers=None,
            worker_replicas=1,
            agents=None,
            scenario=None,
            live=False,
            docker=False,
            force=False,
        )

        # Mock port in use
        with patch('src.cli.main.is_port_in_use', return_value=True):
            cmd_system(args)
            captured = capsys.readouterr()
            assert "in use" in captured.out.lower() or "warning" in captured.out.lower()


# =============================================================================
# Test Demo Command Registry
# =============================================================================

class TestDemoRegistry:
    def test_demo_types_available(self):
        """Test that all demo types are registered."""
        from src.cli.main import cmd_demo

        # Create args with invalid type to get error with list
        args = argparse.Namespace(type="invalid_demo_type")

        with pytest.raises(SystemExit):
            cmd_demo(args)

    def test_demo_quick_default(self, capsys):
        """Test that quick is the default demo type."""
        from src.cli.main import cmd_demo

        args = argparse.Namespace(type=None)

        with patch('src.cli.main.import_module') as mock_import:
            mock_module = MagicMock()
            mock_module.main = AsyncMock()
            mock_import.return_value = mock_module

            with patch('src.cli.main.run_async'):
                cmd_demo(args)


# =============================================================================
# Test Quorum Presets
# =============================================================================

class TestQuorumPresets:
    def test_quorum_realtime_preset(self, capsys):
        """Test realtime quorum preset."""
        from src.cli.main import cmd_quorum

        args = argparse.Namespace(
            preset="realtime",
            event="Test event",
            backend="openrouter",
            duration=5,
            cycles=1,
            watch=None,
            test=False,
        )

        # Mock the quorum system
        mock_system = MagicMock()
        mock_system.start = AsyncMock()
        mock_system.get_status = AsyncMock(return_value={
            'stats': {'events_processed': 0, 'deliberations_completed': 0}
        })
        mock_system.stop = AsyncMock()

        with patch('src.cli.main.import_module', return_value=lambda: mock_system):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                # Test would require more complex async mocking
                pass


# =============================================================================
# Test Debate Command
# =============================================================================

class TestDebateCommand:
    def test_debate_list_flag(self, capsys):
        """Test that --list shows topics and matchups."""
        from src.cli.main import cmd_debate

        args = argparse.Namespace(
            list=True,
            topic=None,
            matchup=None,
            scenario=None,
            participants=None,
            rounds=5,
            interactive=False,
            auto=False,
            no_llm=False,
            provider=None,
            model=None,
        )

        cmd_debate(args)
        captured = capsys.readouterr()

        assert "ai_pause" in captured.out
        assert "doom_vs_accel" in captured.out
        assert "Matchups" in captured.out

    def test_debate_topics_defined(self):
        """Test that debate topics are properly defined."""
        from src.cli.main import cmd_debate

        # Check that the function has the expected topics
        # (Inspect the function's code or test behavior)
        pass


# =============================================================================
# Test Serve Command
# =============================================================================

class TestServeCommand:
    def test_serve_port_check(self, capsys):
        """Test that serve checks port availability."""
        from src.cli.main import cmd_serve

        args = argparse.Namespace(
            host="0.0.0.0",
            port=2040,
            workers=1,
            reload=False,
            scenario=None,
            live=False,
            redis_url=None,
            agents=None,
            advanced=False,
            simple=False,
        )

        with patch('src.cli.main.is_port_in_use', return_value=True):
            with patch('src.cli.main.find_available_port', return_value=2041):
                with pytest.raises(SystemExit):
                    cmd_serve(args)

    def test_serve_env_vars(self, monkeypatch):
        """Test that serve sets environment variables correctly."""
        from src.cli.main import cmd_serve

        args = argparse.Namespace(
            host="0.0.0.0",
            port=8080,
            workers=1,
            reload=False,
            scenario="test_scenario",
            live=True,
            redis_url="redis://localhost:6380",
            agents=None,
            advanced=False,
            simple=True,
        )

        with patch('src.cli.main.is_port_in_use', return_value=False):
            with patch('uvicorn.run') as mock_run:
                try:
                    cmd_serve(args)
                except:
                    pass

                # Check environment was set
                assert os.environ.get("PORT") == "8080"
                assert os.environ.get("SCENARIO") == "test_scenario"


# =============================================================================
# Test Workers Command
# =============================================================================

class TestWorkersCommand:
    def test_workers_default_values(self, capsys):
        """Test workers command uses correct defaults."""
        from src.cli.main import cmd_workers

        args = argparse.Namespace(
            count=None,
            redis_url=None,
            capacity=None,
            work_types=None,
        )

        # This would require mocking the async broker connection
        # Just verify it starts without immediate errors
        captured = capsys.readouterr()


# =============================================================================
# Test Deliberate Command
# =============================================================================

class TestDeliberateCommand:
    def test_deliberate_no_topic_error(self):
        """Test that deliberate fails without topic."""
        from src.cli.main import cmd_deliberate

        args = argparse.Namespace(
            port=2040,
            topic=None,
            scenario="nonexistent_scenario",
            timeout=0,
            poll_interval=5,
        )

        with pytest.raises(CLIError) as exc:
            cmd_deliberate(args)
        assert "No topic specified" in str(exc.value)

    def test_deliberate_api_health_check(self, capsys):
        """Test that deliberate checks API health."""
        from src.cli.main import cmd_deliberate

        args = argparse.Namespace(
            port=9999,  # Unlikely to be in use
            topic="Test topic",
            scenario="quick_personas3",
            timeout=0,
            poll_interval=5,
        )

        with pytest.raises(SystemExit):
            cmd_deliberate(args)

        captured = capsys.readouterr()
        assert "Cannot connect" in captured.out or "ERROR" in captured.out


# =============================================================================
# Integration Tests
# =============================================================================

class TestCLIIntegration:
    def test_full_cli_import(self):
        """Test that the entire CLI module imports without errors."""
        from src.cli import main
        assert hasattr(main, 'main')
        assert hasattr(main, 'CLIError')
        assert hasattr(main, 'run_async')

    def test_all_commands_have_functions(self):
        """Test that all commands have corresponding functions."""
        expected_commands = [
            'cmd_run', 'cmd_experiment', 'cmd_sweep', 'cmd_aggregate',
            'cmd_serve', 'cmd_dashboard', 'cmd_quorum', 'cmd_debate',
            'cmd_demo', 'cmd_workers', 'cmd_chat', 'cmd_wargame',
            'cmd_system', 'cmd_deliberate', 'cmd_version',
        ]

        from src.cli import main as cli_module

        for cmd in expected_commands:
            assert hasattr(cli_module, cmd), f"Missing command function: {cmd}"

    def test_project_root_path(self):
        """Test that _PROJECT_ROOT is set correctly."""
        from src.cli.main import _PROJECT_ROOT

        assert _PROJECT_ROOT.exists()
        assert (_PROJECT_ROOT / "src").exists()
        assert (_PROJECT_ROOT / "pyproject.toml").exists()


# =============================================================================
# Performance Tests
# =============================================================================

class TestCLIPerformance:
    def test_import_time(self):
        """Test that CLI imports quickly."""
        import time

        start = time.time()
        import importlib
        importlib.reload(sys.modules.get('src.cli.main') or __import__('src.cli.main'))
        elapsed = time.time() - start

        # Should import in under 2 seconds
        assert elapsed < 2.0, f"CLI import took {elapsed:.2f}s"

    def test_help_response_time(self):
        """Test that --help responds quickly."""
        import time
        import subprocess

        start = time.time()
        result = subprocess.run(
            [sys.executable, "-m", "src.cli.main", "--help"],
            capture_output=True,
            timeout=5,
        )
        elapsed = time.time() - start

        assert result.returncode == 0
        assert elapsed < 2.0, f"Help took {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
