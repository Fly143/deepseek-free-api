"""配置管理模块 — DeepSeek 多账号管理 + 轮询负载均衡

敏感字段（token / password / cookie / headers / admin_password / mailcx_api_key）
落盘时用 Fernet 加密，前缀 enc:v1:。密钥在同目录 .secret_key（已 gitignore）。
旧明文配置可直接加载，下次 save 自动升级为密文。
"""

import json
import os
import threading
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, asdict, field

from cryptography.fernet import Fernet, InvalidToken


BASE_DIR = Path(__file__).parent.parent
CONFIG_FILE = BASE_DIR / "config.json"
LEGACY_FILE = BASE_DIR / "token.json"

ENC_PREFIX = "enc:v1:"
_SENSITIVE_ACCOUNT_FIELDS = ("token", "_password", "cookie", "headers")
DEFAULT_COMPRESSION_MODE = "compress"


class SecretBox:
    """本地密钥 + Fernet 加解密。密钥文件与 config.json 同目录。"""

    def __init__(self, config_path: Path):
        self.key_path = config_path.parent / ".secret_key"
        self._fernet: Optional[Fernet] = None
        self._lock = threading.RLock()

    def _load_or_create(self) -> Fernet:
        with self._lock:
            if self._fernet is not None:
                return self._fernet
            if self.key_path.exists():
                raw = self.key_path.read_bytes().strip()
                self._fernet = Fernet(raw)
                return self._fernet
            key = Fernet.generate_key()
            self.key_path.write_bytes(key)
            try:
                os.chmod(self.key_path, 0o600)
            except OSError:
                pass  # Windows
            self._fernet = Fernet(key)
            return self._fernet

    def encrypt(self, plaintext: str) -> str:
        if plaintext is None or plaintext == "":
            return ""
        if isinstance(plaintext, str) and plaintext.startswith(ENC_PREFIX):
            return plaintext
        token = self._load_or_create().encrypt(
            plaintext.encode("utf-8") if isinstance(plaintext, str) else str(plaintext).encode("utf-8")
        ).decode("ascii")
        return ENC_PREFIX + token

    def decrypt(self, value: str) -> str:
        if value is None or value == "":
            return ""
        if not isinstance(value, str) or not value.startswith(ENC_PREFIX):
            return value
        blob = value[len(ENC_PREFIX):].encode("ascii")
        try:
            return self._load_or_create().decrypt(blob).decode("utf-8")
        except (InvalidToken, Exception) as e:
            print(f"[Config] decrypt failed ({e}); check .secret_key")
            return ""


@dataclass
class DsAccount:
    """DeepSeek 账号配置（内存明文）"""
    account_label: str       # 手机号 或 "user@example.com"
    login_type: str          # "phone" 或 "email"
    _password: str = ""
    # 手机登录
    _mobile: str = ""
    _area_code: str = "+86"
    # 邮箱登录
    _email: str = ""
    # 会话信息
    token: str = ""
    session_id: str = ""
    headers: dict = field(default_factory=dict)
    cookie: str = ""
    # 状态
    login_time: str = ""
    is_valid: bool = False
    # Lock-avoidance (persisted)
    muted_until: float = 0.0          # unix ts; skip account until then
    cooldown_until: float = 0.0       # short cooldown after errors
    consecutive_errors: int = 0
    last_used_at: float = 0.0

    def to_dict(self):
        d = asdict(self)
        if self.token and len(self.token) > 28:
            d["token_masked"] = self.token[:20] + "..." + self.token[-8:]
        else:
            d["token_masked"] = "***"
        d.pop("token", None)
        d.pop("_password", None)
        d.pop("cookie", None)
        d.pop("headers", None)
        return d

    def to_save_dict(self, box: Optional[SecretBox] = None):
        """落盘：敏感字段加密。"""
        d = asdict(self)
        if box is not None:
            d["token"] = box.encrypt(self.token or "")
            d["_password"] = box.encrypt(self._password or "")
            d["cookie"] = box.encrypt(self.cookie or "")
            if self.headers:
                # headers 整体序列化后加密，避免泄露 authorization
                d["headers"] = box.encrypt(json.dumps(self.headers, ensure_ascii=False))
            else:
                d["headers"] = {}
        return d

    @classmethod
    def from_storage_dict(cls, data: dict, box: Optional[SecretBox] = None) -> "DsAccount":
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if box is not None:
            fields["token"] = box.decrypt(fields.get("token", "") or "")
            fields["_password"] = box.decrypt(fields.get("_password", "") or "")
            fields["cookie"] = box.decrypt(fields.get("cookie", "") or "")
            headers = fields.get("headers", {})
            if isinstance(headers, str) and headers:
                decrypted = box.decrypt(headers)
                try:
                    fields["headers"] = json.loads(decrypted) if decrypted else {}
                except (json.JSONDecodeError, TypeError):
                    fields["headers"] = {}
        return cls(**fields)


class ConfigManager:
    """配置管理器 — 线程安全 + 轮询负载均衡"""

    def __init__(self):
        self.config_file = CONFIG_FILE
        self.box = SecretBox(CONFIG_FILE)
        self.lock = threading.RLock()
        self.account_idx = 0
        self.accounts: List[DsAccount] = []
        self._proxy_url: str = ""
        self._passthrough: bool = False
        self._admin_password: str = "admin"
        self._compression_mode: str = DEFAULT_COMPRESSION_MODE
        # 批量注册：mail.cx 临时邮箱配置
        self._mailcx_api_key: str = ""
        self._mailcx_domain: str = ""
        self.load()

    def _migrate_legacy(self):
        """从旧的 token.json 迁移单账号数据"""
        if not LEGACY_FILE.exists():
            return False
        try:
            old = json.loads(LEGACY_FILE.read_text("utf-8"))
            if not old.get("token"):
                return False
            account_label = old.get("account", "")
            if not account_label:
                if old.get("_email"):
                    account_label = old["_email"]
                elif old.get("_mobile"):
                    account_label = f"{old.get('_area_code', '+86')} {old['_mobile']}"
                else:
                    account_label = "legacy_account"

            for acc in self.accounts:
                if acc.account_label == account_label:
                    print(f"[Config] 账号 {account_label} 已存在，跳过迁移")
                    return False

            account = DsAccount(
                account_label=account_label,
                login_type=old.get("login_type", "phone"),
                _password=old.get("_password", ""),
                _mobile=old.get("_mobile", ""),
                _area_code=old.get("_area_code", "+86"),
                _email=old.get("_email", ""),
                token=old.get("token", ""),
                session_id=old.get("session_id", ""),
                headers=old.get("headers", {}),
                cookie=old.get("cookie", ""),
                login_time=old.get("login_time", ""),
                is_valid=bool(old.get("token")),
            )
            self.accounts.append(account)
            self.save()
            LEGACY_FILE.rename(LEGACY_FILE.with_suffix(".json.bak"))
            print(f"[Config] 已从 token.json 迁移账号: {account_label}")
            return True
        except Exception as e:
            print(f"[Config] 迁移 token.json 失败: {e}")
            return False

    def load(self):
        """加载配置"""
        if not self.config_file.exists():
            if not self._migrate_legacy():
                self.save()
            return
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.accounts = [
                    DsAccount.from_storage_dict(acc, self.box)
                    for acc in data.get('accounts', [])
                ]
                self._proxy_url = data.get('proxy', '') or ''
                self._passthrough = data.get('passthrough', False)
                self._compression_mode = data.get('compression_mode', DEFAULT_COMPRESSION_MODE)
                self._admin_password = self.box.decrypt(data.get('admin_password', '') or '') or 'admin'
                self._mailcx_api_key = self.box.decrypt(data.get('mailcx_api_key', '') or '')
                self._mailcx_domain = data.get('mailcx_domain', '') or ''
        except Exception as e:
            print(f"[Config] 加载配置失败: {e}")
            self.accounts = []
            self.save()

    def save(self):
        """保存配置"""
        with self.lock:
            try:
                data = {
                    "accounts": [acc.to_save_dict(self.box) for acc in self.accounts],
                    "proxy": self._proxy_url or "",
                    "passthrough": self._passthrough,
                    "compression_mode": self._compression_mode,
                    "admin_password": self.box.encrypt(self._admin_password or "admin"),
                    "mailcx_api_key": self.box.encrypt(self._mailcx_api_key or ""),
                    "mailcx_domain": self._mailcx_domain or "",
                }
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[Config] 保存配置失败: {e}")

    def get_next_account(self) -> Optional[DsAccount]:
        """Pick a usable account: not muted, not in cooldown, prefer LRU."""
        import time as _time
        now = _time.time()
        with self.lock:
            if not self.accounts:
                return None
            usable = []
            for a in self.accounts:
                if not a.is_valid:
                    continue
                muted_until = float(getattr(a, "muted_until", 0) or 0)
                if muted_until and muted_until > now:
                    continue
                # Auto-clear expired mute
                if muted_until and muted_until <= now:
                    a.muted_until = 0.0
                cd = float(getattr(a, "cooldown_until", 0) or 0)
                if cd and cd > now:
                    continue
                usable.append(a)
            if not usable:
                return None
            # Prefer least-recently-used to spread load across accounts
            usable.sort(key=lambda a: float(getattr(a, "last_used_at", 0) or 0))
            account = usable[0]
            account.last_used_at = now
            self.account_idx = (self.account_idx + 1) % max(1, len(self.accounts))
            self.save()
            return account

    def mark_account_muted(self, label: str, mute_until: float = 0.0, banned: bool = False):
        """Persist mute/ban so the account is skipped until mute_until (or forever if banned)."""
        import time as _time
        with self.lock:
            for acc in self.accounts:
                if acc.account_label != label:
                    continue
                if banned:
                    acc.is_valid = False
                    acc.muted_until = 0.0
                else:
                    # Keep credentials but skip until mute expires
                    acc.muted_until = float(mute_until or 0) or (_time.time() + 86400)
                    acc.is_valid = True
                acc.consecutive_errors = int(getattr(acc, "consecutive_errors", 0) or 0) + 1
                self.save()
                print(f"[LockGuard] marked {label} muted_until={acc.muted_until} banned={banned}")
                return True
            return False

    def mark_account_cooldown(self, label: str, seconds: float = 60.0):
        import time as _time
        with self.lock:
            for acc in self.accounts:
                if acc.account_label != label:
                    continue
                acc.cooldown_until = _time.time() + float(seconds)
                acc.consecutive_errors = int(getattr(acc, "consecutive_errors", 0) or 0) + 1
                self.save()
                return True
            return False

    def mark_account_ok(self, label: str):
        with self.lock:
            for acc in self.accounts:
                if acc.account_label != label:
                    continue
                acc.consecutive_errors = 0
                acc.cooldown_until = 0.0
                self.save()
                return True
            return False

    def get_account_by_label(self, label: str) -> Optional[DsAccount]:
        with self.lock:
            for acc in self.accounts:
                if acc.account_label == label:
                    return acc
            return None

    def add_account(self, account: DsAccount) -> bool:
        with self.lock:
            for acc in self.accounts:
                if acc.account_label == account.account_label:
                    acc._password = account._password or acc._password
                    acc._mobile = account._mobile or acc._mobile
                    acc._area_code = account._area_code or acc._area_code
                    acc._email = account._email or acc._email
                    acc.login_type = account.login_type or acc.login_type
                    if account.token:
                        acc.token = account.token
                    if account.session_id:
                        acc.session_id = account.session_id
                    if account.headers:
                        acc.headers = account.headers
                    if account.cookie:
                        acc.cookie = account.cookie
                    acc.is_valid = account.is_valid or acc.is_valid
                    self.save()
                    return False
            self.accounts.append(account)
            self.save()
            return True

    def remove_account(self, label: str) -> bool:
        with self.lock:
            before = len(self.accounts)
            self.accounts = [a for a in self.accounts if a.account_label != label]
            if len(self.accounts) < before:
                self.save()
                return True
            return False

    def update_account(self, label: str, **kwargs):
        with self.lock:
            for acc in self.accounts:
                if acc.account_label == label:
                    for k, v in kwargs.items():
                        if hasattr(acc, k):
                            setattr(acc, k, v)
                    self.save()
                    return True
            return False

    def mark_invalid(self, label: str):
        self.update_account(label, is_valid=False)

    def get_all_accounts(self) -> List[dict]:
        with self.lock:
            return [acc.to_dict() for acc in self.accounts]

    def get_proxy(self) -> str:
        with self.lock:
            return self._proxy_url or ""

    def set_proxy(self, url: str):
        with self.lock:
            self._proxy_url = (url or "").strip()
            self.save()

    def get_passthrough(self) -> bool:
        with self.lock:
            return self._passthrough

    def set_passthrough(self, enabled: bool):
        with self.lock:
            self._passthrough = bool(enabled)
            self.save()

    def get_compression_mode(self) -> str:
        with self.lock:
            return self._compression_mode or DEFAULT_COMPRESSION_MODE

    def set_compression_mode(self, mode: str):
        with self.lock:
            self._compression_mode = "truncation" if mode == "truncation" else DEFAULT_COMPRESSION_MODE
            self.save()

    def get_admin_password(self) -> str:
        with self.lock:
            return self._admin_password

    def set_admin_password(self, password: str):
        with self.lock:
            self._admin_password = password or "admin"
            self.save()

    def get_token(self, label: str) -> str:
        with self.lock:
            for acc in self.accounts:
                if acc.account_label == label:
                    return acc.token
            return ""

    def count(self) -> int:
        with self.lock:
            return len(self.accounts)

    def count_valid(self) -> int:
        with self.lock:
            return sum(1 for a in self.accounts if a.is_valid)

    def get_mailcx_api_key(self) -> str:
        with self.lock:
            return self._mailcx_api_key or ""

    def set_mailcx_api_key(self, key: str):
        with self.lock:
            self._mailcx_api_key = (key or "").strip()
            self.save()

    def get_mailcx_domain(self) -> str:
        with self.lock:
            return self._mailcx_domain or ""

    def set_mailcx_domain(self, domain: str):
        with self.lock:
            self._mailcx_domain = (domain or "").strip()
            self.save()


config_manager = ConfigManager()
