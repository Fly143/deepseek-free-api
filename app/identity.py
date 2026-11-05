import base64
import os
import random
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


def new_device_id() -> str:
    return base64.b64encode(os.urandom(32)).decode()
