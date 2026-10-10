import json
import math
import os
import re

import numpy as np
import requests
from PIL import Image

from vlmeval.smp import encode_image_to_base64, get_logger
from .base import BaseAPI

logger = get_logger(__name__)

# ---- 全局 RPM 限速门 (针对有 RPM 配额的网关, 如 antchemy ~60 RPM) ----
# 多 worker 共享一把锁: 串行化"请求起始时刻", 把总请求速率封顶在 API_RPM/分钟。
# 高 api-nproc 也不会撞配额(快条目被门挡住、慢条目用足并发)。
# 未设 API_RPM 则不启用, 对其他脚本零影响。
import threading as _threading
import time as _time

_RPM_LOCK = _threading.Lock()
_RPM_LAST = 0.0


def _rpm_gate():
    _limit = os.getenv('API_RPM', '')
    if not _limit:
        return
    try:
        _rpm = float(_limit)
    except (TypeError, ValueError):
        return
    if _rpm <= 0:
        return
    _min_interval = 60.0 / _rpm
    global _RPM_LAST
    with _RPM_LOCK:
        _now = _time.monotonic()
        _wait = _min_interval - (_now - _RPM_LAST)
        if _wait > 0:
            _time.sleep(_wait)
        _RPM_LAST = _time.monotonic()


APIBASES = {
    'OFFICIAL': 'https://api.openai.com/v1/chat/completions',
}


def GPT_context_window(model):
    length_map = {
        'gpt-4': 8192,
        'gpt-4-0613': 8192,
        'gpt-4-turbo-preview': 128000,
        'gpt-4-1106-preview': 128000,
        'gpt-4-0125-preview': 128000,
        'gpt-4-vision-preview': 128000,
        'gpt-4-turbo': 128000,
        'gpt-4-turbo-2024-04-09': 128000,
        'gpt-3.5-turbo': 16385,
        'gpt-3.5-turbo-0125': 16385,
        'gpt-3.5-turbo-1106': 16385,
        'gpt-3.5-turbo-instruct': 4096,
    }
    if model in length_map:
        return length_map[model]
    else:
        return 128000


class OpenAIWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'gpt-3.5-turbo-0613',
                 retry: int = 5,
                 key: str = None,
                 verbose: bool = False,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 300,
                 api_base: str = None,
                 max_tokens: int = 2048,
                 img_size: int = -1,
                 total_img_size: int = -1,
                 max_file_size: int = 1e9,
                 img_detail: str = 'low',
                 use_azure: bool = False,
                 strip_reasoning_content: bool = False,
                 **kwargs):

        self.model = model
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_azure = use_azure
        self.strip_reasoning_content = strip_reasoning_content

        if 'step' in model:
            env_key = os.environ.get('STEPAI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'yi-vision' in model:
            env_key = os.environ.get('YI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'internvl2-pro' in model:
            env_key = os.environ.get('InternVL2_PRO_KEY', '')
            if key is None:
                key = env_key
        elif 'abab' in model:
            env_key = os.environ.get('MiniMax_API_KEY', '')
            if key is None:
                key = env_key
        elif 'moonshot' in model:
            env_key = os.environ.get('MOONSHOT_API_KEY', '')
            if key is None:
                key = env_key
        elif 'grok' in model:
            env_key = os.environ.get('XAI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'gemini' in model and 'preview' in model:
            # Will only handle preview models
            env_key = os.environ.get('GOOGLE_API_KEY', '')
            if key is None:
                key = env_key
            # api_base = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        elif 'ernie' in model:
            env_key = os.environ.get('BAIDU_API_KEY', '')
            if key is None:
                key = env_key
            api_base = 'https://qianfan.baidubce.com/v2/chat/completions'
            self.baidu_appid = os.environ.get('BAIDU_APP_ID', None)
        else:
            if use_azure:
                env_key = os.environ.get('AZURE_OPENAI_API_KEY', None)
                assert env_key is not None, 'Please set the environment variable AZURE_OPENAI_API_KEY. '

                if key is None:
                    key = env_key
                assert isinstance(key, str), (
                    'Please set the environment variable AZURE_OPENAI_API_KEY to your openai key. '
                )
            else:
                env_key = os.environ.get('OPENAI_API_KEY', '')
                if key is None:
                    key = env_key

        self.key = key
        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        assert total_img_size > 0 or total_img_size == -1
        self.total_img_size = total_img_size
        self.max_file_size = max_file_size
        assert img_detail in ['high', 'low']
        self.img_detail = img_detail
        self.timeout = timeout
        self.is_max_completion_tokens = ('o1' in model) or ('o3' in model) or ('o4' in model) or ('gpt-5' in model)
        self.is_o_model = ('o1' in model) or ('o3' in model) or ('o4' in model)
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        if use_azure:
            api_base_template = (
                '{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}'
            )
            endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', None)
            assert endpoint is not None, 'Please set the environment variable AZURE_OPENAI_ENDPOINT. '
            deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', None)
            assert deployment_name is not None, 'Please set the environment variable AZURE_OPENAI_DEPLOYMENT_NAME. '
            api_version = os.getenv('OPENAI_API_VERSION', None)
            assert api_version is not None, 'Please set the environment variable OPENAI_API_VERSION. '

            self.api_base = api_base_template.format(
                endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
                api_version=os.getenv('OPENAI_API_VERSION')
            )
        else:
            if api_base is None:
                if 'OPENAI_API_BASE' in os.environ and os.environ['OPENAI_API_BASE'] != '':
                    logger.info('Environment variable OPENAI_API_BASE is set. Will use it as api_base. ')
                    api_base = os.environ['OPENAI_API_BASE']
                else:
                    api_base = 'OFFICIAL'

            assert api_base is not None

            if api_base in APIBASES:
                self.api_base = APIBASES[api_base]
            elif api_base.startswith('http'):
                self.api_base = api_base
            else:
                logger.error('Unknown API Base. ')
                raise NotImplementedError
            if os.environ.get('BOYUE', None):
                self.api_base = os.environ.get('BOYUE_API_BASE')
                self.key = os.environ.get('BOYUE_API_KEY')

        masked_key = '' if not self.key else (str(self.key)[:4] + '****' + str(self.key)[-4:] if len(str(self.key)) > 8 else '****')
        logger.info(f'Using API Base: {self.api_base}; API Key: {masked_key}')

    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        image_num = len([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text' and msg['value']:  # Skip empty string
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    from PIL import Image
                    img = Image.open(msg['value'])
                    target_size = math.inf
                    if self.img_size > 0:
                        target_size = self.img_size
                    if self.total_img_size > 0:
                        target_size = min(target_size, int(self.img_size / (image_num**0.5)))
                    target_size = -1 if math.isinf(target_size) else target_size
                    b64 = encode_image_to_base64(
                        img, target_size=target_size, max_file_size=self.max_file_size)
                    img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail=self.img_detail)
                    content_list.append(dict(type='image_url', image_url=img_struct))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:
        input_msgs = self.prepare_inputs(inputs)
        _rpm_gate()   # 全局 RPM 限速 (API_RPM env 启用), 在发请求前排队
        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        # Will send request if use Azure, dk how to use openai client for it
        if self.use_azure:
            headers = {'Content-Type': 'application/json', 'api-key': self.key}
        elif 'internvl2-pro' in self.model:
            headers = {'Content-Type': 'application/json', 'Authorization': self.key}
        else:
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        if hasattr(self, 'baidu_appid'):
            headers['appid'] = self.baidu_appid

        payload = dict(
            model=self.model,
            messages=input_msgs,
            n=1,
            temperature=temperature,
            **kwargs)

        if self.is_max_completion_tokens:
            payload['max_completion_tokens'] = max_tokens
            payload.pop('temperature')
        else:
            payload['max_tokens'] = max_tokens

        if 'gemini' in self.model:
            payload.pop('max_tokens')
            payload.pop('n')
            payload['reasoning_effort'] = 'high'

        if 'claude' in self.model and not os.environ.get('CLAUDE_KEEP_TEMPERATURE', ''):
            # Anthropic answers with a 400 when sampling params are sent to a thinking
            # model, and OpenAI-compatible gateways forward temperature verbatim.
            # Set CLAUDE_KEEP_TEMPERATURE=1 for gateways that swallow it themselves.
            payload.pop('temperature', None)

        if 'openai.com' not in self.api_base and payload.get('n') == 1:
            payload.pop('n', None)

        # openai SDK 通道 (env VLMEVAL_OPENAI_SDK=1 启用): 本环境里 requests/curl 走系统 DNS
        # 会 gaierror, 而 openai SDK(httpx) 能连通 —— 推理流量整体切到 SDK 收发。
        if os.environ.get('VLMEVAL_OPENAI_SDK', ''):
            return self._generate_via_openai_sdk(payload)

        proxies = {}
        if os.getenv('http_proxy'):
            proxies['http'] = os.getenv('http_proxy')
        if os.getenv('https_proxy'):
            proxies['https'] = os.getenv('https_proxy')
        proxies = proxies or None

        response = requests.post(
            self.api_base,
            headers=headers,
            data=json.dumps(payload),
            proxies=proxies,
            timeout=self.timeout * 1.1,
        )
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            message = resp_struct['choices'][0]['message']
            content = message.get('content') or ''
            answer = content.strip()
            # Some OpenAI-compatible reasoning models return CoT in a separate
            # `reasoning_content` field, while a few endpoints may leak it into
            # content with <think> tags. Keep only final answer for scoring.
            if self.strip_reasoning_content:
                answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.S).strip()
        except Exception as err:
            logger.error(f'{type(err)}: {err}')
            if self.verbose:
                logger.error(response.text if hasattr(response, 'text') else response)

        return ret_code, answer, response

    def _generate_via_openai_sdk(self, payload):
        """走 openai 官方 SDK 发请求 (httpx 通道)。

        payload 与 requests 路径共用 (model/messages/temperature/max_tokens...,
        gemini/claude 分支的裁剪已在上游完成)。api_base 配置里的 '/chat/completions'
        后缀剥掉即为 SDK 的 base_url。异常时返回非 0 ret_code 走 BaseAPI 的重试。
        """
        from openai import OpenAI

        base = self.api_base.split('/chat/completions')[0].rstrip('/')
        if getattr(self, '_sdk_client', None) is None or self._sdk_client.base_url != base:
            # max_retries=4: 网关链路抖动 (连接级失败) 靠 SDK 自带指数退避扛,
            # 外层还有 BaseAPI.retry (config 里 retry=6) 兜底。
            self._sdk_client = OpenAI(base_url=base, api_key=self.key,
                                      timeout=self.timeout * 1.1, max_retries=4)
        kwargs = {k: v for k, v in payload.items() if v is not None}
        try:
            resp = self._sdk_client.chat.completions.create(**kwargs)
        except Exception as err:
            logger.error(f'{type(err).__name__}: {str(err)[:300]}')
            return 1, self.fail_msg, None

        class _Stub:  # 兼容失败路径里个别调用方读 .status_code/.text 的习惯
            status_code = 0
            text = ''

        answer = ''
        try:
            message = resp.choices[0].message
            answer = (message.content or '').strip()
            if self.strip_reasoning_content:
                # 与 requests 路径同一口径: 剥离 content 里漏出的 çois...thinking 块
                answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.S).strip()
        except Exception as err:
            logger.error(f'{type(err).__name__}: {str(err)[:300]}')
            return 1, self.fail_msg, _Stub()
        return 0, answer, resp

    def get_image_token_len(self, img_path, detail='low'):
        import math
        if detail == 'low':
            return 85

        im = Image.open(img_path)
        height, width = im.size
        if width > 1024 or height > 1024:
            if width > height:
                height = int(height * 1024 / width)
                width = 1024
            else:
                width = int(width * 1024 / height)
                height = 1024

        h = math.ceil(height / 512)
        w = math.ceil(width / 512)
        total = 85 + 170 * h * w
        return total

    def get_token_len(self, inputs) -> int:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(self.model)
        except Exception as err:
            if 'gpt' in self.model.lower():
                if self.verbose:
                    logger.warning(f'{type(err)}: {err}')
                enc = tiktoken.encoding_for_model('gpt-4')
            else:
                return 0
        assert isinstance(inputs, list)
        tot = 0
        for item in inputs:
            if 'role' in item:
                tot += self.get_token_len(item['content'])
            elif item['type'] == 'text':
                tot += len(enc.encode(item['value']))
            elif item['type'] == 'image':
                tot += self.get_image_token_len(item['value'], detail=self.img_detail)
        return tot


class GPT4V(OpenAIWrapper):

    def generate(self, message, dataset=None):
        return super(GPT4V, self).generate(message)
