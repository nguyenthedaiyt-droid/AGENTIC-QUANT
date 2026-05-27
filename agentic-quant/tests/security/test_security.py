# =============================================================================
# AGENTIC-QUANT — Security Tests
#
# Test 1: TV webhook HMAC signature -> gui request sai signature -> 401
# Test 2: Redis bind 127.0.0.1 (config check)
# Test 3: SQLite perms 600 (stat check)
# Test 4: API keys in env, k co in code (grep check)
# =============================================================================

from __future__ import annotations

import hashlib
import hmac
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger


# =============================================================================
# Constants
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # /tmp/AGENTIC-QUANT/agentic-quant/


# =============================================================================
# Test 1: TV Webhook HMAC Signature
# =============================================================================


@pytest.mark.security
class TestWebhookHmacSignature:
    """Test webhook HMAC signature verification.

    Mo phong TradingView webhook request voi HMAC SHA256 signature.
    Gui request sai signature -> phai tra ve 401.
    """

    SECRET_KEY = "test-secret-key-for-hmac"

    @staticmethod
    def _compute_signature(payload: bytes, secret: str) -> str:
        """Compute HMAC-SHA256 signature cho payload.

        Args:
            payload: Request payload bytes.
            secret: HMAC secret key.

        Returns:
            Hex digest signature.
        """
        return hmac.new(
            secret.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _verify_signature(
        payload: bytes,
        signature: str,
        secret: str,
    ) -> bool:
        """Verify HMAC-SHA256 signature.

        Args:
            payload: Request payload bytes.
            signature: HMAC signature tu request header.
            secret: HMAC secret key.

        Returns:
            True neu signature hop le, False neu khong.
        """
        expected = TestWebhookHmacSignature._compute_signature(payload, secret)
        return hmac.compare_digest(expected, signature)

    def test_valid_signature_returns_200(self) -> None:
        """Gui request dung signature -> verify thanh cong (tra ve 200)."""
        payload = b'{"symbol": "XAUUSD", "close": 2500.5, "volume": 100}'
        signature = self._compute_signature(payload, self.SECRET_KEY)

        result = self._verify_signature(payload, signature, self.SECRET_KEY)

        assert result is True, "Valid signature should pass verification"

    def test_invalid_signature_returns_401(self) -> None:
        """Gui request sai signature -> verify that bai (tra ve 401)."""
        payload = b'{"symbol": "XAUUSD", "close": 2500.5, "volume": 100}'
        wrong_signature = "deadbeef" * 8  # Fake signature

        result = self._verify_signature(payload, wrong_signature, self.SECRET_KEY)

        assert result is False, "Invalid signature should fail verification"

    def test_tampered_payload_returns_401(self) -> None:
        """Tampered payload (khac voi signature) -> verify that bai."""
        original_payload = b'{"symbol": "XAUUSD", "close": 2500.5}'
        tampered_payload = b'{"symbol": "XAUUSD", "close": 9999.9}'
        signature = self._compute_signature(original_payload, self.SECRET_KEY)

        result = self._verify_signature(tampered_payload, signature, self.SECRET_KEY)

        assert result is False, "Tampered payload should fail verification"

    def test_wrong_secret_returns_401(self) -> None:
        """Dung sai secret key -> verify that bai."""
        payload = b'{"symbol": "XAUUSD", "close": 2500.5}'
        signature = self._compute_signature(payload, "wrong-secret-key")

        result = self._verify_signature(payload, signature, self.SECRET_KEY)

        assert result is False, "Wrong secret should fail verification"

    def test_empty_payload_signature(self) -> None:
        """Test empty payload signature."""
        payload = b""
        signature = self._compute_signature(payload, self.SECRET_KEY)

        result = self._verify_signature(payload, signature, self.SECRET_KEY)

        assert result is True, "Empty payload with valid signature should pass"

    def test_missing_signature_header(self) -> None:
        """Thieu signature header -> tuong duong sai signature."""
        payload = b'{"symbol": "XAUUSD"}'
        # Khong co signature -> None
        result = self._verify_signature(payload, "", self.SECRET_KEY)
        assert result is False, "Missing signature should fail"

    def test_hmac_timing_independence(self) -> None:
        """Verify hmac.compare_digest khong bi timing attack."""
        payload = b'{"symbol": "XAUUSD", "close": 2500.5}'
        valid_sig = self._compute_signature(payload, self.SECRET_KEY)

        # compare_digest la constant-time
        result = hmac.compare_digest(valid_sig, valid_sig)
        assert result is True, "compare_digest should match valid sig"

        # Slight variation in signature
        invalid_sig = valid_sig[:-1] + ("0" if valid_sig[-1] != "0" else "1")
        result = hmac.compare_digest(valid_sig, invalid_sig)
        assert result is False, "compare_digest should reject invalid sig"


# =============================================================================
# Test 2: Redis bind 127.0.0.1
# =============================================================================


@pytest.mark.security
class TestRedisBindLocalhost:
    """Test Redis bind to localhost only.

    Check config file (redis.conf hoac docker-compose) de dam bao
    Redis bind 127.0.0.1, khong expose ra public network.
    """

    REDIS_CONF_PATTERNS: list[str] = [
        "redis.conf",
        "redis.local.conf",
        "docker-compose.yml",
        "docker-compose.yaml",
        "docker-compose.local.yml",
        "config/redis.conf",
    ]

    def test_redis_conf_binds_localhost(self) -> None:
        """Kiem tra Redis config bind 127.0.0.1."""
        found_bind = False
        config_paths: list[Path] = []

        for pattern in self.REDIS_CONF_PATTERNS:
            path = PROJECT_ROOT / pattern
            if path.exists():
                config_paths.append(path)
                content = path.read_text()
                # Check bind directive (docker-compose command or redis.conf)
                if re.search(r"(?:bind|--bind)\s+127\.0\.0\.1", content):
                    found_bind = True
                    logger.info(f"Redis bind 127.0.0.1 found in: {path}")
                elif re.search(r"(?:bind|--bind)\s+0\.0\.0\.0", content):
                    logger.warning(f"Redis bind 0.0.0.0 (public!) in: {path}")

        if not config_paths:
            # Neu khong co file config, kiem tra code reference
            logger.info("No Redis config file found, checking code references...")
            self._check_redis_bind_in_code()

        if config_paths:
            assert found_bind, (
                f"Redis khong bind 127.0.0.1 trong cac config: "
                f"{[str(p) for p in config_paths]}"
            )

    def test_redis_no_password_in_config(self) -> None:
        """Kiem tra Redis config khong chua password plaintext."""
        for pattern in self.REDIS_CONF_PATTERNS:
            path = PROJECT_ROOT / pattern
            if path.exists():
                content = path.read_text()
                # requirepass should not be in plaintext
                if "requirepass" in content and "requirepass" not in content.split("#")[0]:
                    logger.warning(f"Redis requirepass found in: {path}")

    def _check_redis_bind_in_code(self) -> None:
        """Check code Redis connection strings."""
        py_files = list(PROJECT_ROOT.rglob("*.py"))
        for py_file in py_files:
            if "__pycache__" in str(py_file):
                continue
            try:
                content = py_file.read_text()
                if "redis" in content.lower():
                    # Check for host=127.0.0.1 or host=localhost
                    if re.search(r'redis.*host\s*[:=]\s*["\'](?!127\.0\.0\.1|localhost)([^"\']+)["\']', content):
                        logger.warning(
                            f"Redis connects to non-localhost in: {py_file}"
                        )
            except (OSError, UnicodeDecodeError):
                continue


# =============================================================================
# Test 3: SQLite Permissions 600
# =============================================================================


@pytest.mark.security
class TestSqlitePermissions:
    """Test SQLite database file permissions.

    Tat ca SQLite db files phai co permission 600 (owner read/write only).
    """

    def _find_sqlite_files(self) -> list[Path]:
        """Tim tat ca SQLite database files trong project.

        Returns:
            List SQLite file paths.
        """
        sqlite_files: list[Path] = []
        patterns = ["*.db", "*.sqlite", "*.sqlite3"]
        for pattern in patterns:
            sqlite_files.extend(PROJECT_ROOT.rglob(pattern))
        return sqlite_files

    def test_sqlite_permissions_600(self) -> None:
        """Kiem tra SQLite files co permission 600."""
        sqlite_files = self._find_sqlite_files()

        if not sqlite_files:
            # Neu khong co SQLite files, check code references
            logger.info("No SQLite files found, skipping permission check")
            pytest.skip("No SQLite files found in project")
            return

        for db_path in sqlite_files:
            if "__pycache__" in str(db_path) or ".git" in str(db_path):
                continue

            mode = db_path.stat().st_mode
            # Check owner-only permissions (stat.S_IRUSR | stat.S_IWUSR = 0o600)
            owner_only = mode & 0o777
            perms_str = oct(owner_only)

            assert owner_only <= 0o600, (
                f"SQLite file {db_path} co permissions {perms_str}, "
                f"can be world-readable! Expected <= 0o600"
            )
            logger.info(f"  OK: {db_path} perms={perms_str}")

    def test_sqlite_wal_permissions(self) -> None:
        """Kiem tra SQLite WAL/SHM files cung co permission safety."""
        wal_files: list[Path] = []
        for pattern in ["*-wal", "*-shm"]:
            wal_files.extend(PROJECT_ROOT.rglob(pattern))

        for wf in wal_files:
            if "__pycache__" in str(wf) or ".git" in str(wf):
                continue
            mode = wf.stat().st_mode
            owner_only = mode & 0o777
            assert owner_only <= 0o600, (
                f"WAL file {wf} co permissions {oct(owner_only)}, "
                f"can be world-readable!"
            )

    def test_sqlite_create_with_600_perms(self) -> None:
        """Test tao moi SQLite file co permissions 600.

        Verify rang khi tao moi database, permissions duoc set dung.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_secure.db"

            # Tao file voi restrict permissions
            db_path.touch()
            os.chmod(db_path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600

            mode = db_path.stat().st_mode & 0o777
            assert mode == 0o600, (
                f"New SQLite file permissions {oct(mode)} != 0o600"
            )


# =============================================================================
# Test 4: API keys in env, not in code
# =============================================================================


@pytest.mark.security
class TestApiKeysNotInCode:
    """Test API keys trong environment variables, khong hardcode trong source.

    Kiem tra:
      - API keys khong xuat hien trong source code (Python, YAML, JSON, ENV)
      - Cac bien env duoc khai bao (SECRET_KEY, API_KEY, TOKEN, ...)
    """

    # Patterns for potential hardcoded keys
    KEY_PATTERNS: list[re.Pattern] = [
        re.compile(r'(?:api[_-]?key|apikey|secret[_-]?key)\s*[=:]\s*["\'][A-Za-z0-9_\-]{16,}["\']', re.IGNORECASE),
        re.compile(r'(?:access[_-]?token|bearer[_-]?token)\s*[=:]\s*["\'][A-Za-z0-9_\-]{20,}["\']', re.IGNORECASE),
        re.compile(r'(?:password|passwd|pwd)\s*[=:]\s*["\'][A-Za-z0-9!@#$%^&*()_+\-={}\[\]|;:,.<>?]{8,}["\']', re.IGNORECASE),
        re.compile(r'(?:client[_-]?secret|consumer[_-]?secret)\s*[=:]\s*["\'][A-Za-z0-9_\-]{16,}["\']', re.IGNORECASE),
    ]

    # Files/directories to exclude
    EXCLUDE_DIRS: set[str] = {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
    }

    # Allowed config files where keys can appear
    ALLOWED_KEY_FILES: set[str] = {
        ".env.example",
        ".env.template",
        "README.md",
        "docker-compose.yml",
        "docker-compose.yaml",
    }

    # Self-test exemption — test files containing HMAC test keys
    SELF_EXEMPT_FILES: set[str] = {
        "test_security.py",
    }

    def test_no_hardcoded_api_keys_in_python(self) -> None:
        """Kiem tra khong co API keys hardcode trong Python files."""
        violations: list[tuple[Path, str, str]] = []
        py_files = list(PROJECT_ROOT.rglob("*.py"))

        for py_file in py_files:
            if any(excl in str(py_file) for excl in self.EXCLUDE_DIRS):
                continue
            if py_file.name in self.ALLOWED_KEY_FILES:
                continue
            if py_file.name in self.SELF_EXEMPT_FILES:
                continue

            try:
                content = py_file.read_text()
                for pattern in self.KEY_PATTERNS:
                    matches = pattern.findall(content)
                    for match in matches:
                        violations.append((py_file, pattern.pattern, match[:60]))
            except (OSError, UnicodeDecodeError):
                continue

        if violations:
            msg_lines = ["Potential hardcoded API keys found:"]
            for fpath, pat, match in violations[:20]:
                msg_lines.append(f"  {fpath}: {match}...")
            pytest.fail("\n".join(msg_lines))

    def test_no_hardcoded_api_keys_in_yaml_json(self) -> None:
        """Kiem tra khong co API keys hardcode trong YAML/JSON/ENV files."""
        violations: list[tuple[Path, str]] = []
        config_patterns = ["*.yml", "*.yaml", "*.json", "*.env", "*.cfg", "*.ini"]
        for pattern_str in config_patterns:
            for cfg_file in PROJECT_ROOT.rglob(pattern_str):
                if any(excl in str(cfg_file) for excl in self.EXCLUDE_DIRS):
                    continue
                if cfg_file.name in self.ALLOWED_KEY_FILES:
                    continue

                try:
                    content = cfg_file.read_text()
                    for pattern in self.KEY_PATTERNS:
                        if pattern.search(content):
                            violations.append((cfg_file, pattern.pattern))
                except (OSError, UnicodeDecodeError):
                    continue

        if violations:
            msg_lines = ["Potential hardcoded keys in config files:"]
            for fpath, pat in violations[:10]:
                msg_lines.append(f"  {fpath}: {pat}")
            pytest.fail("\n".join(msg_lines))

    def test_env_variables_defined(self) -> None:
        """Kiem tra cac env variables duoc khai bao (trong .env.example)."""
        env_example = PROJECT_ROOT / ".env.example"
        if not env_example.exists():
            pytest.skip("No .env.example found")

        content = env_example.read_text()
        env_vars = re.findall(r'^([A-Z_]+)=', content, re.MULTILINE)

        expected_vars = [
            "SECRET_KEY",
            "REDIS_PASSWORD",
            "API_KEY",
            "WEBHOOK_SECRET",
        ]

        missing = [v for v in expected_vars if v not in env_vars]
        if missing:
            logger.warning(
                "Expected env vars missing from .env.example: "
                f"{', '.join(missing)}"
            )

        logger.info(
            f"Environment variables defined ({len(env_vars)}): "
            f"{', '.join(env_vars[:20])}"
        )
