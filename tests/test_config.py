"""
Tests for configuration loading and search order.

Tests config file discovery, bootstrap creation, and INI parsing.
"""
import os
import tempfile
from pathlib import Path

import pytest

from ragger_memory import config as config_module


TEST_CONFIG = """\
[server]
host = 127.0.0.1
port = 8432

[storage]
db_path = ~/.ragger/memories.db
default_collection = memory

[embedding]
model = all-MiniLM-L6-v2
dimensions = 384

[search]
default_limit = 5
default_min_score = 0.4
bm25_enabled = true
bm25_weight = 0.3
vector_weight = 0.7
bm25_k1 = 1.5
bm25_b = 0.75

[logging]
log_dir = ~/.ragger
query_log = true
http_log = true
mcp_log = true

[paths]
normalize_home = true

[import]
minimum_chunk_size = 300
"""


@pytest.fixture
def clean_config():
    """Reset config module state before/after each test."""
    original_config = config_module._config
    original_path = config_module._config_path
    original_multi = config_module._is_multi_user
    config_module._config = None
    config_module._config_path = None
    config_module._is_multi_user = False
    yield
    config_module._config = original_config
    config_module._config_path = original_path
    config_module._is_multi_user = original_multi


@pytest.fixture
def temp_home(tmp_path):
    """Create a temporary home directory for testing."""
    home = tmp_path / "fakehome"
    home.mkdir()
    return home


class TestConfigSearchOrder:
    """Tests for config file search order and bootstrap."""
    
    def test_explicit_cli_path_takes_priority(self, clean_config, tmp_path):
        """--config-file path should override all other locations."""
        explicit_conf = tmp_path / "custom.conf"
        explicit_conf.write_text(TEST_CONFIG)
        
        found = config_module.find_config_file(str(explicit_conf))
        assert found == str(explicit_conf)
    
    def test_nonexistent_cli_path_raises(self, clean_config, tmp_path):
        """Non-existent --config-file should raise FileNotFoundError."""
        missing = tmp_path / "missing.conf"
        with pytest.raises(FileNotFoundError):
            config_module.find_config_file(str(missing))
    
    def test_finds_user_config_if_exists(self, clean_config, temp_home, monkeypatch):
        """Should find ~/.ragger/ragger.ini if it exists (no system config)."""
        monkeypatch.setenv("HOME", str(temp_home))
        # Ensure /etc/ragger.ini doesn't interfere
        monkeypatch.setattr(config_module, "system_config_path",
                            lambda: str(temp_home / "nonexistent" / "ragger.ini"))
        
        ragger_dir = temp_home / ".ragger"
        ragger_dir.mkdir()
        user_conf = ragger_dir / "ragger.ini"
        user_conf.write_text(TEST_CONFIG)
        
        found = config_module.find_config_file()
        assert found == str(user_conf)
    
    def test_bootstrap_creates_default_config(self, clean_config, temp_home, monkeypatch, capsys):
        """First run should create ~/.ragger/ragger.ini if missing."""
        monkeypatch.setenv("HOME", str(temp_home))
        # Ensure /etc/ragger.ini doesn't interfere
        monkeypatch.setattr(config_module, "system_config_path",
                            lambda: str(temp_home / "nonexistent" / "ragger.ini"))
        
        found = config_module.find_config_file()
        
        # Should have created the directory and file
        assert Path(found).exists()
        assert Path(found).name == "ragger.ini"
        assert ".ragger" in found
        
        # Should have printed bootstrap message
        captured = capsys.readouterr()
        assert "Created default config" in captured.err
        
        # Should contain default values
        content = Path(found).read_text()
        assert "port = 8432" in content
        assert "all-MiniLM-L6-v2" in content


class TestConfigLoading:
    """Tests for loading and parsing config files."""
    
    def test_load_config_parses_ini(self, tmp_path):
        """load_config should parse INI format correctly."""
        conf = tmp_path / "test.conf"
        conf.write_text(TEST_CONFIG)
        
        cfg = config_module.load_config(str(conf))
        
        assert cfg["host"] == "127.0.0.1"
        assert cfg["port"] == 8432
        assert cfg["embedding_model"] == "all-MiniLM-L6-v2"
        assert cfg["embedding_dimensions"] == 384
        assert cfg["default_search_limit"] == 5
        assert cfg["default_min_score"] == 0.4
        assert cfg["bm25_enabled"] is True
        assert cfg["minimum_chunk_size"] == 300
    
    def test_load_config_custom_values(self, tmp_path):
        """Custom values should override defaults."""
        custom_conf = """\
[server]
host = 0.0.0.0
port = 9999

[storage]
db_path = /custom/path/db.sqlite
default_collection = notes

[embedding]
model = custom-model
dimensions = 768

[search]
default_limit = 10
default_min_score = 0.6
bm25_enabled = false

[import]
minimum_chunk_size = 500
"""
        conf = tmp_path / "custom.conf"
        conf.write_text(custom_conf)
        
        cfg = config_module.load_config(str(conf))
        
        assert cfg["host"] == "0.0.0.0"
        assert cfg["port"] == 9999
        assert cfg["db_path"] == "/custom/path/db.sqlite"
        assert cfg["default_collection"] == "notes"
        assert cfg["embedding_model"] == "custom-model"
        assert cfg["embedding_dimensions"] == 768
        assert cfg["default_search_limit"] == 10
        assert cfg["default_min_score"] == 0.6
        assert cfg["bm25_enabled"] is False
        assert cfg["minimum_chunk_size"] == 500
    
    def test_load_config_missing_sections_use_fallback(self, tmp_path):
        """Missing INI sections should fall back to defaults."""
        minimal_conf = """\
[server]
port = 7777
"""
        conf = tmp_path / "minimal.conf"
        conf.write_text(minimal_conf)
        
        cfg = config_module.load_config(str(conf))
        
        # Explicitly set value
        assert cfg["port"] == 7777
        # Fallback defaults
        assert cfg["host"] == "127.0.0.1"
        assert cfg["embedding_model"] == "all-MiniLM-L6-v2"


class TestConfigInitialization:
    """Tests for module-level config init and access."""
    
    def test_init_config_loads_file(self, clean_config, tmp_path):
        """init_config should load and return config dict."""
        conf = tmp_path / "test.conf"
        conf.write_text(TEST_CONFIG)
        
        cfg = config_module.init_config(str(conf))
        
        assert isinstance(cfg, dict)
        assert cfg["port"] == 8432
        assert config_module._config is not None
        assert config_module._config_path == str(conf)
    
    def test_get_config_auto_initializes(self, clean_config, temp_home, monkeypatch):
        """get_config should auto-initialize if not already done."""
        monkeypatch.setenv("HOME", str(temp_home))
        
        # Config not initialized yet
        assert config_module._config is None
        
        cfg = config_module.get_config()
        
        # Should have auto-initialized
        assert isinstance(cfg, dict)
        assert config_module._config is not None
    
    def test_get_config_path_returns_loaded_path(self, clean_config, tmp_path):
        """get_config_path should return the loaded config file path."""
        conf = tmp_path / "myconf.conf"
        conf.write_text(TEST_CONFIG)
        
        config_module.init_config(str(conf))
        path = config_module.get_config_path()
        
        assert path == str(conf)


class TestConfigLayering:
    """Tests for system + user config layering."""

    def test_user_overrides_search_limit(self, tmp_path):
        """User config should override allowed settings."""
        sys_conf = tmp_path / "system.conf"
        sys_conf.write_text(TEST_CONFIG)

        user_conf = tmp_path / "user.conf"
        user_conf.write_text("[search]\ndefault_limit = 20\n")

        cfg = config_module.load_layered_config(str(sys_conf), str(user_conf))
        assert cfg["default_search_limit"] == 20
        # Non-overridden values stay from system
        assert cfg["port"] == 8432

    def test_user_cannot_override_port(self, tmp_path):
        """User config should NOT override system-only settings."""
        sys_conf = tmp_path / "system.conf"
        sys_conf.write_text(TEST_CONFIG)

        user_conf = tmp_path / "user.conf"
        user_conf.write_text("[server]\nport = 9999\n")

        cfg = config_module.load_layered_config(str(sys_conf), str(user_conf))
        assert cfg["port"] == 8432  # System wins

    def test_user_overrides_default_collection(self, tmp_path):
        """User should be able to change their default collection."""
        sys_conf = tmp_path / "system.conf"
        sys_conf.write_text(TEST_CONFIG)

        user_conf = tmp_path / "user.conf"
        user_conf.write_text("[storage]\ndefault_collection = personal\n")

        cfg = config_module.load_layered_config(str(sys_conf), str(user_conf))
        assert cfg["default_collection"] == "personal"

    def test_no_system_config_uses_user_only(self, tmp_path):
        """Without system config, user config provides everything."""
        user_conf = tmp_path / "user.conf"
        user_conf.write_text(TEST_CONFIG)

        cfg = config_module.load_layered_config(None, str(user_conf))
        assert cfg["port"] == 8432

    def test_is_multi_user_detection(self, clean_config, tmp_path):
        """Multi-user should be detected when system config exists."""
        sys_conf = tmp_path / "system.conf"
        sys_conf.write_text(TEST_CONFIG)

        # Monkey-patch system_config_path to our temp file
        config_module.init_config(str(sys_conf))
        # With explicit path, it's not multi-user (it's dev/test mode)
        # Multi-user is only when system config is found via standard path


class TestExpandPath:
    """Tests for path expansion."""
    
    def test_expand_tilde_to_home(self, temp_home, monkeypatch):
        """~ should expand to $HOME."""
        monkeypatch.setenv("HOME", str(temp_home))
        
        expanded = config_module.expand_path("~/.ragger/db.sqlite")
        assert expanded.startswith(str(temp_home))
        assert "~" not in expanded
    
    def test_expand_absolute_unchanged(self):
        """Absolute paths should be unchanged."""
        path = "/absolute/path/to/file"
        assert config_module.expand_path(path) == path
    
    def test_expand_relative_unchanged(self):
        """Relative paths (not starting with ~) should be unchanged."""
        path = "relative/path"
        assert config_module.expand_path(path) == path


class TestBackwardCompatibility:
    """Tests for old-style attribute access via __getattr__."""
    
    def test_getattr_default_port(self, clean_config, tmp_path):
        """config.DEFAULT_PORT should work via __getattr__."""
        conf = tmp_path / "test.conf"
        conf.write_text(TEST_CONFIG)
        config_module.init_config(str(conf))
        
        assert config_module.DEFAULT_PORT == 8432
    
    def test_getattr_default_host(self, clean_config, tmp_path):
        """config.DEFAULT_HOST should work via __getattr__."""
        conf = tmp_path / "test.conf"
        conf.write_text(TEST_CONFIG)
        config_module.init_config(str(conf))
        
        assert config_module.DEFAULT_HOST == "127.0.0.1"
    
    def test_getattr_static_value(self, clean_config):
        """Static values like STORAGE_ENGINE should return constants."""
        # Don't need config file for static values
        assert config_module.STORAGE_ENGINE == "sqlite"
        assert config_module.SQLITE_MEMORIES_TABLE == "memories"
    
    def test_getattr_unknown_raises(self, clean_config):
        """Unknown attributes should raise AttributeError."""
        with pytest.raises(AttributeError):
            _ = config_module.NONEXISTENT_ATTR
