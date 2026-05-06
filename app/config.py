"""配置管理模块 — DeepSeek 多账号管理 + 轮询负载均衡"""

import json
import threading
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, asdict, field


BASE_DIR = Path(__file__).parent.parent
CONFIG_FILE = BASE_DIR / "config.json"
LEGACY_FILE = BASE_DIR / "token.json"


@dataclass
class DsAccount:
    """DeepSeek 账号配置"""
    account_label: str       # "+86 REDACTED" 或 "user@example.com"
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

    def to_dict(self):
        d = asdict(self)
        # 不暴露完整 token （只在前端展示掩码版本）
        if self.token and len(self.token) > 28:
            d["token_masked"] = self.token[:20] + "..." + self.token[-8:]
        else:
            d["token_masked"] = "***"
        return d

    def to_save_dict(self):
        """保存到文件时保留完整字段"""
        return asdict(self)


class ConfigManager:
    """配置管理器 — 线程安全 + 轮询负载均衡"""

    def __init__(self):
        self.config_file = CONFIG_FILE
        self.lock = threading.RLock()
        self.account_idx = 0
        self.accounts: List[DsAccount] = []
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
                # 从凭证推断
                if old.get("_email"):
                    account_label = old["_email"]
                elif old.get("_mobile"):
                    account_label = f"{old.get('_area_code', '+86')} {old['_mobile']}"
                else:
                    account_label = "legacy_account"

            # 检查是否已迁移过
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
            # 重命名旧文件避免重复迁移
            LEGACY_FILE.rename(LEGACY_FILE.with_suffix(".json.bak"))
            print(f"[Config] 已从 token.json 迁移账号: {account_label}")
            return True
        except Exception as e:
            print(f"[Config] 迁移 token.json 失败: {e}")
            return False

    def load(self):
        """加载配置"""
        if not self.config_file.exists():
            # 尝试迁移旧文件
            if not self._migrate_legacy():
                self.save()
            return
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.accounts = [
                    DsAccount(**{k: v for k, v in acc.items()
                                 if k in DsAccount.__dataclass_fields__})
                    for acc in data.get('accounts', [])
                ]
        except Exception as e:
            print(f"[Config] 加载配置失败: {e}")
            self.accounts = []
            self.save()

    def save(self):
        """保存配置"""
        with self.lock:
            try:
                data = {
                    "accounts": [acc.to_save_dict() for acc in self.accounts],
                }
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[Config] 保存配置失败: {e}")

    def get_next_account(self) -> Optional[DsAccount]:
        """获取下一个可用账号（轮询 + 自动跳过无效账号）"""
        with self.lock:
            if not self.accounts:
                return None
            valid_accounts = [a for a in self.accounts if a.is_valid]
            if not valid_accounts:
                return None
            # 只在有效账号中轮询
            account = valid_accounts[self.account_idx % len(valid_accounts)]
            self.account_idx += 1
            return account

    def get_account_by_label(self, label: str) -> Optional[DsAccount]:
        """按标签查找账号"""
        with self.lock:
            for acc in self.accounts:
                if acc.account_label == label:
                    return acc
            return None

    def add_account(self, account: DsAccount) -> bool:
        """添加账号。返回 True 表示新增，False 表示已存在"""
        with self.lock:
            for acc in self.accounts:
                if acc.account_label == account.account_label:
                    # 更新已有账号（如密码变更）
                    acc._password = account._password or acc._password
                    acc._mobile = account._mobile or acc._mobile
                    acc._area_code = account._area_code or acc._area_code
                    acc._email = account._email or acc._email
                    acc.login_type = account.login_type or acc.login_type
                    self.save()
                    return False
            self.accounts.append(account)
            self.save()
            return True

    def remove_account(self, label: str) -> bool:
        """删除账号"""
        with self.lock:
            before = len(self.accounts)
            self.accounts = [a for a in self.accounts if a.account_label != label]
            if len(self.accounts) < before:
                self.save()
                return True
            return False

    def update_account(self, label: str, **kwargs):
        """更新账号字段（如 token/session_id/headers 刷新）"""
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
        """标记账号无效"""
        self.update_account(label, is_valid=False)

    def get_all_accounts(self) -> List[dict]:
        """获取所有账号的摘要信息（无敏感字段）"""
        with self.lock:
            return [acc.to_dict() for acc in self.accounts]

    def count(self) -> int:
        with self.lock:
            return len(self.accounts)

    def count_valid(self) -> int:
        with self.lock:
            return sum(1 for a in self.accounts if a.is_valid)


# 全局配置管理器实例
config_manager = ConfigManager()
