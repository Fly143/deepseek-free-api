import base64
import os
import random
import threading
from collections import deque
from typing import Dict

_IOS_UAS = [
    "DeepSeek/2 CFNetwork/1568.100.1 Darwin/24.0.0",
    "DeepSeek/2.1 CFNetwork/1568.100.1 Darwin/24.0.0",
    "DeepSeek/2 CFNetwork/1490.0.4 Darwin/23.4.0",
    "DeepSeek/2 CFNetwork/1410.0.3 Darwin/22.4.0",
    "DeepSeek/2.0 CFNetwork/1568.100.1 Darwin/24.1.0",
    "DeepSeek/2 CFNetwork/1474.4 Darwin/23.2.0",
]
_IOS_VERSIONS = ["2.0.4", "2.0.2", "2.1.0", "2.0.1", "1.10.1", "2.0.0"]
_IOS_TZ_OFFSETS = ["28800", "3600", "0", "7200", "32400", "10800", "-14400", "25200"]


def _random_ios_headers(locale: str = "zh_CN") -> Dict[str, str]:
    return {
        "accept": "application/json",
        "content-type": "application/json",
        "user-agent": random.choice(_IOS_UAS),
        "x-client-bundle-id": "com.deepseek.chat",
        "x-client-locale": locale,
        "x-client-platform": "ios",
        "x-client-timezone-offset": random.choice(_IOS_TZ_OFFSETS),
        "x-client-version": random.choice(_IOS_VERSIONS),
        "x-rangers-id": str(random.randint(10 ** 18, 10 ** 19 - 1)),
    }


def random_ios_headers(locale: str = "zh_CN") -> Dict[str, str]:
    return _random_ios_headers(locale)


_recent_device_ids: "deque" = deque(maxlen=400)
_recent_lock = threading.Lock()


def new_device_id() -> str:
    """生成 device_id：优先取仓库内置的真实指纹池（与上游真实客户端格式一致），池不可用时回退随机 base64。

    近期已发放的 id 会被排除，避免短时间内重复；池耗尽时 get_device_id 自行回退随机值。
    """
    try:
        from app.device_ids import get_device_id

        with _recent_lock:  # 选号与登记放在同一临界区，避免并发拿到同一 id
            value = get_device_id(exclude=set(_recent_device_ids))
            if isinstance(value, str) and value:
                _recent_device_ids.append(value)  # 有界队列自动淘汰最旧，避免整批清空后立刻复用
                return value
    except Exception:
        pass
    return base64.b64encode(os.urandom(32)).decode()
