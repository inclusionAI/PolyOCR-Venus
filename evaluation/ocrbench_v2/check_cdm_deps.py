#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CDM(公式识别 + 文档解析公式子项)依赖自检: 缺啥报啥, 不静默降级。

CDM 需要:
  python: numpy/pandas/nltk(+wordnet)/jieba/tqdm/openpyxl/Pillow/lxml/Levenshtein/
          pylatexenc/scipy/bs4/PyMuPDF
  系统  : TeX Live 的 pdflatex + kpsewhich + CJK.sty + gkai 字体(arphic)

退出码: 0 = 全部就绪;  2 = 有 CDM 阻断性缺失(公式类无法正确评分)。
"""
from pathlib import Path as _EvalPath
import os as _eval_os
_EVAL_ROOT = _EvalPath(_eval_os.environ.get("VLMEVAL_ROOT", str(_EvalPath(__file__).resolve().parents[0]))).resolve()

import os
import sys

for p in (str(_EVAL_ROOT / '.cache/localpkgs'), str(_EVAL_ROOT / '.cache/shadow')):
    if os.path.isdir(p):
        sys.path.insert(0, p)
os.environ.setdefault('CDM_TEXLIVE_ROOT', str(_EVAL_ROOT / '.tools/texlive/2025'))


def _ok(msg):    print(f'  [ OK ] {msg}')
def _miss(msg):  print(f'  [MISS] {msg}')
def _info(msg):  print(f'        {msg}')


def check_python():
    print('== python 依赖 ==')
    py = [
        ('numpy', 'numpy'), ('pandas', 'pandas'), ('nltk', 'nltk'), ('jieba', 'jieba'),
        ('tqdm', 'tqdm'), ('openpyxl', 'openpyxl'), ('PIL', 'pillow'), ('lxml', 'lxml'),
        ('Levenshtein', 'python-Levenshtein'), ('pylatexenc', 'pylatexenc'),
        ('scipy', 'scipy'), ('bs4', 'beautifulsoup4'), ('pymupdf', 'PyMuPDF'),
    ]
    miss = []
    for mod, pip in py:
        try:
            __import__(mod)
            _ok(f'{mod}')
        except Exception as e:
            _miss(f'{mod}  ->  pip install {pip}')
            miss.append(pip)
    # wordnet 语料
    try:
        import nltk
        from nltk.corpus import wordnet
        wordnet.synsets('dog')
        _ok('nltk wordnet 语料')
    except Exception:
        _miss('nltk wordnet 语料  ->  nltk.download(wordnet) + nltk.download(omw-1.4), 放到 NLTK_DATA')
        miss.append('wordnet')
    return miss


def check_texlive():
    print('\n== TeX Live (CDM 公式渲染) ==')
    miss = []
    try:
        from vlmeval.dataset.utils.cdm.modules.texlive_env import (
            resolve_tex_binary, describe_tex_runtime)
    except Exception as e:
        _miss(f'无法导入 texlive_env: {e}')
        return ['texlive_env']
    desc = describe_tex_runtime()
    # pdflatex
    p = resolve_tex_binary('pdflatex')
    if p and os.path.isfile(p) and os.access(p, os.X_OK) and not str(desc.get('pdflatex_version', '')).startswith('unavailable'):
        _ok(f'pdflatex: {p}')
    else:
        _miss('pdflatex  (TeX Live 未安装/不在 PATH/CDM_TEXLIVE_ROOT 指错)')
        miss.append('pdflatex')
    # kpsewhich
    k = resolve_tex_binary('kpsewhich')
    if k and os.path.isfile(k) and os.access(k, os.X_OK):
        _ok(f'kpsewhich: {k}')
    else:
        _miss('kpsewhich')
        miss.append('kpsewhich')
    # CJK.sty
    cjk = str(desc.get('cjk_sty', ''))
    if cjk and not cjk.startswith('unavailable') and not cjk.startswith('exit_code') and os.path.isfile(cjk):
        _ok(f'CJK.sty: {cjk}')
    else:
        _miss('CJK.sty  ->  TeX Live collection-langcjk')
        _info(cjk)
        miss.append('CJK.sty')
    # gkai 字体
    gkai = str(desc.get('cjk_font_fd', ''))
    cjk_font = str(desc.get('cjk_font', ''))
    if gkai and not gkai.startswith('unavailable') and not gkai.startswith('exit_code') and os.path.isfile(gkai):
        _ok(f'gkai 字体 ({cjk_font}): {gkai}')
    else:
        _miss(f'gkai 字体 ({cjk_font})  ->  TeX Live collection-langchinese (含 arphic)')
        _info(gkai)
        miss.append('gkai')
    _info(f'texlive_root = {desc.get("texlive_root","") or "(未检出)"}')
    _info(f'CDM_TEXLIVE_ROOT env = {os.environ.get("CDM_TEXLIVE_ROOT","")}')
    return miss


def cdm_deps_status():
    """跑依赖检查(逐项打印 [OK]/[MISS]), 返回 (py_miss, tex_miss)。供打分脚本 fail-fast 调用。"""
    py_miss = check_python()
    tex_miss = check_texlive()
    return py_miss, tex_miss


def main():
    print('CDM 依赖自检 (公式识别 / 文档解析公式子项)\n')
    py_miss, tex_miss = cdm_deps_status()
    print('\n== 结论 ==')
    if not py_miss and not tex_miss:
        print('  全部就绪: CDM 公式评测可正常运行。')
        return 0
    if py_miss:
        print(f'  缺 python: {py_miss}')
        print('  -> pip install ' + ' '.join(sorted(set(py_miss) - {"wordnet"})))
        if 'wordnet' in py_miss:
            print('  -> nltk wordnet 见上 (放到 NLTK_DATA, 如 .cache/nltk_data 或 .cache/nltk_data)')
    if tex_miss:
        print(f'  缺 TeX Live: {tex_miss}')
        print('  -> 安装 TeX Live 2025 (scheme-basic + collection-latexrecommended')
        print('     + collection-latexextra + collection-langcjk + collection-langchinese),')
        print('     装到 .tools/texlive/2025 或设 CDM_TEXLIVE_ROOT 指向其根目录。')
        print('  -> 缺 TeX Live 时, 公式识别 en/cn 与文档解析的公式子项无法用 CDM 评分。')
    return 2


if __name__ == '__main__':
    sys.exit(main())