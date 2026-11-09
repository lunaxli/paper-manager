import os

DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "700"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()