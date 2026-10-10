import ssl
import warnings

# Ignore pkg_resources warning due to jieba depends on it.
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")

# Temporarily bypass SSL certificate verification to download files from oss.
ssl._create_default_https_context = ssl._create_unverified_context


def load_env():
    import logging
    import os
    from pathlib import Path
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s - %(filename)s: %(funcName)s - %(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    pth = Path(__file__).parent.parent / '.env'
    if not pth.exists():
        logging.error(f'Did not detect the .env file at {pth}, failed to load.')
        return

    from dotenv import dotenv_values
    values = dotenv_values(str(pth))
    for k, v in values.items():
        if v is not None and len(v):
            # setdefault: 已显式 export 的环境变量优先于 .env (python-dotenv 标准行为)。
            # 避免 ./.env 里错误/过期的 key 覆盖调用方明确指定的正确 key。
            os.environ.setdefault(k, v)
    logging.info(f'API Keys successfully loaded from {pth}')


load_env()

try:
    import os as _os
    if _os.environ.get('VLMEVAL_NO_TORCH', '') == '1':
        # 仅做离线打分 (ocrbrnch_v2_eval 等) 时不需要 torch/vllm; 跳过可省去 NFS 上
        # 巨量 torch 文件的慢导入。需要推理/模型时勿设置此变量。
        pass
    else:
        import torch  # noqa: F401
except ImportError:
    pass

# from .api import *  # noqa: F401, F403, E402
# from .config import *  # noqa: F401, F403, E402
# from .dataset import *  # noqa: F401, F403, E402
# from .smp import *  # noqa: F401, F403, E402
from .tools import cli  # noqa: F401, E402

# from .utils import *  # noqa: F401, F403, E402
# from .vlm import *  # noqa: F401, F403, E402

__version__ = '0.2rc1'
