"""Admin 认证模块 — HTTP Basic Auth"""

import secrets
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.config import config_manager

security = HTTPBasic()


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """验证管理员密码。用户名固定为 admin，密码从配置读取。"""
    correct_password = config_manager.get_admin_password()
    
    # 使用 secrets.compare_digest 防时序攻击
    is_correct_username = secrets.compare_digest(
        credentials.username.encode("utf-8"),
        b"admin"
    )
    is_correct_password = secrets.compare_digest(
        credentials.password.encode("utf-8"),
        correct_password.encode("utf-8")
    )
    
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="认证失败",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials
