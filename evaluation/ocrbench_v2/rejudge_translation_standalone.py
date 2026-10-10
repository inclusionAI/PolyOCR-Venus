#!/usr/bin/env python3
# 外科式复判 ocrbench_v21 的 "text translation cn" 子类 (400 条)。
# 目的: 上次全 0 是 antchemy 网关 429 -> judge 失败回退 0; 本脚本低并发重判翻译,
#       rubric 与官方 vqa_metric 逐字一致, 不碰 pdflatex/TEDS/CDM, 不依赖 NFS 重 import。
# 完成后: 新 translation 均值 + 旧 reasoning VQA cn (确定性, 未受影响) 重算
#         cn_knowledge_reasoning 与 Chinese Overall, 增量 patch 三份官方产物。
import os, sys, re, json, time, hashlib, ast, threading
from collections import Counter, deque

# ---------- judge rubric: 与 vqa_metric.py 逐字一致 ----------
TEXT_TRANSLATION_JUDGE_PROMPT = """You are a translation evaluator for OCRBench v2. The task may be Chinese-to-English or English-to-Chinese. Compare the model's translated text with the reference translation and judge whether they express the same meaning in the intended target language.

Evaluation principles:
- Focus on semantic equivalence, not word-by-word matching.
- Natural paraphrases, synonym substitutions, and word-order differences should receive high scores if the meaning is preserved.
- Do not penalize harmless formatting or punctuation differences.
- Penalize mistranslation, missing key information, wrong entities/numbers/negation/relations, or using the wrong target language.
- If the model output contains extra explanation, score based on whether the reference meaning is still complete and not contradicted.

Model translation:
{predict}

Reference translation:
{answer}

Please give a score between 0 and 1:
- 1.0: Semantically equivalent, fluent, and complete
- 0.7-0.9: Mostly equivalent with only minor issues
- 0.4-0.6: Partially correct but with obvious omissions or mistranslations
- 0.1-0.3: Only a few fragments are relevant; overall meaning is wrong
- 0.0: Completely wrong, unrelated, empty, or unreadable

Output ONLY a single number between 0 and 1 (e.g. 0.8). Do not output anything else."""

def _parse_score(response):
    if not response:
        return 0.0
    # Strip Gemini thinking tokens (repeated … or … chars) that appear before the actual score
    response = re.sub(r'^[\.…\s]+', '', response.strip())
    # If after stripping thinking tokens only whitespace/cruft remains, try tail for a number
    if not response or not re.search(r'\d', response):
        # Scan entire original response for a plausible float 0-1
        m = re.search(r'\b([01]?\.?\d+)\b', response or '')
        if m:
            try:
                return max(0.0, min(1.0, float(m.group(1))))
            except ValueError:
                pass
        return 0.0
    try:
        return max(0.0, min(1.0, float(response)))
    except ValueError:
        m = re.search(r'(\d+\.?\d*)', response)
        if m:
            try:
                return max(0.0, min(1.0, float(m.group(1))))
            except ValueError:
                pass
        return 0.0

def _safe_literal_eval(value):
    raw = value if isinstance(value, str) else str(value)
    try:
        result = ast.literal_eval(raw)
    except Exception:
        result = raw
    if isinstance(result, (list, dict, tuple)):
        return result
    s = raw.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        inner = s[1:-1].replace('""', '"')
        try:
            r = ast.literal_eval(inner)
            if isinstance(r, (list, dict, tuple)):
                return r
        except Exception:
            pass
    return result

def _sleep(s): time.sleep(s)

import requests

class Judge:
    FAIL = 'Failed to obtain answer via API. '
    def __init__(self, model, api_base, key, retry=6, timeout=120):
        self.model=model; self.api_base=api_base.rstrip('/'); self.key=key
        self.retry=retry; self.timeout=timeout; self.temperature=0; self.max_tokens=1024
    def generate(self, message):
        prompt=message if isinstance(message,str) else json.dumps(message,ensure_ascii=False)
        k=hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        c=_cache_get(k)
        if c is not None: return c
        payload={'model':self.model,'messages':[{'role':'user','content':prompt}],'temperature':self.temperature}
        if 'gemini' in self.model:
            payload.pop('max_tokens',None); payload['reasoning_effort']='high'
        else:
            payload['max_tokens']=self.max_tokens
        headers={'Content-Type':'application/json','Authorization':f'Bearer {self.key}'}
        last=None
        for attempt in range(max(1,self.retry)):
            _acquire_slot()
            try:
                r=requests.post(self.api_base+'/chat/completions',headers=headers,
                    data=json.dumps(payload,ensure_ascii=False),
                    timeout=self.timeout*1.1,proxies=None)
                if 200<=r.status_code<300:
                    ans=json.loads(r.text)['choices'][0]['message']['content']
                    _cache_put(k,ans); return ans
                last=f'{r.status_code}:{r.text[:300]}'
                if r.status_code==429:
                    _sleep(min(30.0,5.0*(attempt+1))); continue
            except Exception as e:
                last=f'{type(e).__name__}:{e}'
            _sleep(0.5)
        print(f'[WARN] judge fail model={self.model} last={last}',flush=True)
        return self.FAIL

# ---------- 单进程限流: 滑窗 60s <= RPM ----------
_s_lock=threading.Lock(); _stamps=deque()
RPM=int(os.environ.get('OCRBENCH_JUDGE_RPM','40'))
def _acquire_slot():
    while True:
        with _s_lock:
            now=time.time()
            while _stamps and _stamps[0] < now-60.0:
                _stamps.popleft()
            if len(_stamps) < RPM:
                _stamps.append(now); return
            wait=_stamps[0]+60.02-now
        _sleep(max(0.05,min(wait,1.0)))

# ---------- 断点缓存: 与官方 judge_cache.json 同格式 {sha256: ans} ----------
CACHE_DIR=os.environ.get('OCRBENCH_JUDGE_CACHE_DIR','')
CACHE_FILE=os.path.join(CACHE_DIR,'judge_cache.json') if CACHE_DIR else ''
_cache_buf={}; _cache_dirty=False; _wlock=threading.Lock()
def _cache_load():
    global _cache_buf
    if not CACHE_FILE: return
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE) as f: _cache_buf=json.load(f) or {}
            print(f'[cache] loaded {len(_cache_buf)} entries',flush=True)
    except Exception as e:
        print(f'[cache] load failed {e}',flush=True)
def _cache_get(k): return _cache_buf.get(k)
def _cache_put(k,ans):
    global _cache_dirty
    with _wlock: _cache_buf[k]=ans; _cache_dirty=True
def _cache_flush(force=False):
    global _cache_dirty
    if not CACHE_FILE or not _cache_dirty: return
    with _wlock:
        if not _cache_dirty: return
        os.makedirs(os.path.dirname(CACHE_FILE) or '.',exist_ok=True)
        tmp=CACHE_FILE+'.tmp'
        with open(tmp,'w') as f: json.dump(_cache_buf,f,ensure_ascii=False)
        os.replace(tmp,CACHE_FILE); _cache_dirty=False

# ---------- xlsx 读取 (openpyxl, 绕开 pandas) ----------
def read_translation_rows(xlsx_path):
    import openpyxl
    wb=openpyxl.load_workbook(xlsx_path,read_only=True,data_only=True)
    ws=wb.active
    rows=ws.iter_rows(values_only=True)
    header=next(rows)
    idx={name:i for i,name in enumerate(header)}
    need=['category','prediction','answer','orig_idx']
    miss=[c for c in need if c not in idx]
    if miss: raise SystemExit(f'[FATAL] xlsx 缺列 {miss}; 实有 {list(header)}')
    out=[]
    for r in rows:
        if r[idx['category']]!='text translation cn': continue
        predict=str(r[idx['prediction']]) if r[idx['prediction']] is not None else ''
        ans_raw=r[idx['answer']]
        answers=_safe_literal_eval(ans_raw) if (ans_raw not in (None,'')) else []
        if not isinstance(answers,list): answers=[answers]
        orig=r[idx['orig_idx']]
        out.append({'orig_idx': int(orig) if (orig is not None and str(orig).strip()!='') else None,
                    'predict': predict, 'answers': answers})
    wb.close()
    return out

def judge_one_row(j, row):
    # 逐字复刻 text_translation_vlm_judge_evaluation: 多答案取 max; 返回 (score, n_calls, n_ok)
    predict=row['predict']
    if not isinstance(predict,str) or len(predict.strip())==0:
        return 0.0,0,0
    answers=row['answers'] if isinstance(row['answers'],list) else [row['answers']]
    max_score=0.0; n=0; nok=0
    for ans in answers:
        if not isinstance(ans,str): ans=str(ans)
        if len(ans.strip())==0: continue
        prompt=TEXT_TRANSLATION_JUDGE_PROMPT.format(predict=predict.strip(),answer=ans.strip())
        resp=j.generate(prompt); n+=1
        if resp!=j.FAIL: nok+=1
        s=_parse_score(resp)
        if s>max_score: max_score=s
    return max_score,n,nok

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--pred',default=os.environ.get('PREDFILE'))
    ap.add_argument('--xlsx-out-dir',default=os.environ.get('SCOREDIR'))
    ap.add_argument('--nproc',type=int,default=int(os.environ.get('OCRBENCH_REJUDGE_NPROC','8')))
    ap.add_argument('--dry-run',action='store_true',help='只读 xlsx + 1 条真 judge canary + 打印, 不批量')
    ap.add_argument('--apply',action='store_true',help='判分完成且 ok_rate>=门槛 时 patch 官方三份产物')
    ap.add_argument('--min-ok-rate',type=float,default=0.90)
    args=ap.parse_args()

    model=os.environ.get('JUDGE_MODEL','gemini-3.5-flash-lite')
    base=os.environ.get('JUDGE_BASE_URL','https://antalchemy.alipay.com/v1')
    key=os.environ['JUDGE_KEY']
    retry=int(os.environ.get('JUDGE_RETRY','6')); timeout=int(os.environ.get('JUDGE_TIMEOUT','120'))
    base_out=args.xlsx_out_dir or os.path.dirname(args.pred)
    print(f'[cfg] pred={args.pred}',flush=True)
    print(f'[cfg] judge={model}@{base} rpm={RPM} nproc={args.nproc} retry={retry} timeout={timeout} cache={CACHE_FILE or "(off)"} apply={args.apply}',flush=True)

    _cache_load()
    rows=read_translation_rows(args.pred)
    answer_dist=Counter(len(r['answers']) for r in rows)
    emp=sum(1 for r in rows if not r['predict'].strip())
    print(f'[parse] translation rows={len(rows)} 预测空={emp}(应0) answers/row分布={dict(sorted(answer_dist.items()))}',flush=True)

    j=Judge(model,base,key,retry=retry,timeout=timeout)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if args.dry_run:
        sample=[r for r in rows if r['predict'].strip()][0]
        print(f'[canary] predict={sample["predict"][:80]!r} answers={sample["answers"]}',flush=True)
        t=time.time(); score,n,nok=judge_one_row(j,sample)
        print(f'[canary] score={score} n_calls={n} ok={nok} latency={time.time()-t:.1f}s',flush=True)
        _cache_flush(force=True); return

    scores=[None]*len(rows); meta=[None]*len(rows)
    done=[0]; t0=time.time(); lock=threading.Lock(); fail_rows=[0]
    def work(i): return i, *judge_one_row(j,rows[i])
    with ThreadPoolExecutor(max_workers=args.nproc) as ex:
        futs=[ex.submit(work,i) for i in range(len(rows))]
        for fut in as_completed(futs):
            i,s,n,nok=fut.result()
            scores[i]=s; meta[i]=(n,nok)
            with lock:
                done[0]+=1
                if n>0 and nok==0: fail_rows[0]+=1
                dn=done[0]
                if dn%10==0 or dn==len(rows):
                    print(f'[run] {dn}/{len(rows)} ({100*dn//len(rows)}%) t={time.time()-t0:.0f}s cache={len(_cache_buf)} failrows={fail_rows[0]}',flush=True)
                if dn%50==0: _cache_flush()
    _cache_flush(force=True)
    n_calls=sum(m[0] for m in meta if m); n_ok=sum(m[1] for m in meta if m)
    n_bad=sum(1 for m in meta if m and m[0]>0 and m[1]==0)
    ok_rate=n_ok/n_calls if n_calls else 0.0
    tt_mean=sum(scores)/len(scores) if scores else 0.0
    print(f'\n[done] rows={len(scores)} calls={n_calls} ok_calls={n_ok} ok_rate={ok_rate:.3f} fail_rows={n_bad} tt_mean={tt_mean:.5f} t={time.time()-t0:.0f}s',flush=True)
    nz=[s for s in scores if s>0]; z=sum(1 for s in scores if s==0.0)
    import statistics
    print(f'[hist] 非零 {len(nz)}/{len(scores)}; 零(含坏0) {z}; 非零均值={statistics.mean(nz):.4f}' if nz else '[hist] 全0',flush=True)

    audit=os.path.join(base_out,'rejudge_translation_peritem.jsonl')
    with open(audit,'w') as f:
        for i,r in enumerate(rows):
            f.write(json.dumps({'orig_idx':r['orig_idx'],'predict':r['predict'][:200],'answers':r['answers'],
                                'score':scores[i],'n_calls':meta[i][0],'n_ok':meta[i][1]},ensure_ascii=False)+'\n')
    print(f'[audit] -> {audit}',flush=True)

    # 旧 reasoning VQA cn 逐条分(确定性, 未受影响), 重算 cn_knowledge_reasoning
    old_jsonl=os.path.join(base_out,'eval_qwen36_35ba3b_ocrbench_v2_org_0819_all_think_combined_OCRBench_v2_org_scored.jsonl.pre_rejudge')
    old_score_p=os.path.join(base_out,'eval_qwen36_35ba3b_ocrbench_v2_org_0819_all_think_combined_OCRBench_v2_org_score.json.pre_rejudge')
    old_sub_p=os.path.join(base_out,'eval_qwen36_35ba3b_ocrbench_v2_org_0819_all_think_combined_OCRBench_v2_org_subcategory_score.json.pre_rejudge')
    rv=[]
    if os.path.exists(old_jsonl):
        for ln in open(old_jsonl):
            d=json.loads(ln)
            if d.get('category')=='reasoning VQA cn' and d.get('score') is not None:
                rv.append(d['score'])
    sum_rv=sum(rv); n_rv=len(rv); sum_tt=sum(scores); n_tt=len(scores)
    new_cn_kr=(sum_rv+sum_tt)/(n_rv+n_tt) if (n_rv and n_tt) else None
    old_sc=json.load(open(old_score_p)) if os.path.exists(old_score_p) else {}
    old_sub_d=json.load(open(old_sub_p)) if os.path.exists(old_sub_p) else {}
    cn_buckets=[old_sc.get('cn_text_recognition'),old_sc.get('cn_relationship_extraction'),
                old_sc.get('cn_element_parsing'),old_sc.get('cn_visual_text_understanding'),new_cn_kr]
    kept=[x for x in cn_buckets if x is not None]
    new_cn_overall=sum(kept)/len(kept) if kept else None
    old_tt=old_sub_d.get('text translation cn'); old_cn_kr=old_sc.get('cn_knowledge_reasoning')
    print('\n================ 翻译补判结果对照 ================')
    print(f'text translation cn      : 旧={old_tt} -> 新={round(tt_mean,6)}')
    print(f'cn_knowledge_reasoning   : 旧={old_cn_kr} -> 新={round(new_cn_kr,6) if new_cn_kr is not None else None}')
    print(f'Chinese Overall Score     : 旧={old_sc.get("Chinese Overall Score")} -> 新={round(new_cn_overall,6) if new_cn_overall is not None else None}')
    print(f'(其它 29 子类 + EN 侧不变, 均确定性指标)')
    print(f'\n[gateway] ok_rate={ok_rate:.3f} (门槛 {args.min_ok_rate}) fail_rows={n_bad}')

    diff={'text_translation_cn':{'old':old_tt,'new':round(tt_mean,6)},
          'cn_knowledge_reasoning':{'old':old_cn_kr,'new':round(new_cn_kr,6) if new_cn_kr is not None else None},
          'Chinese_Overall_Score':{'old':old_sc.get('Chinese Overall Score'),'new':round(new_cn_overall,6) if new_cn_overall is not None else None},
          'judge_ok_rate':round(ok_rate,4),'total_rows':n_tt,'n_calls':n_calls,'n_ok_calls':n_ok,
          'reasoning_vqa_cn_n':n_rv,'reasoning_vqa_cn_mean':round(sum_rv/n_rv,6) if n_rv else None}
    with open(os.path.join(base_out,'rejudge_translation_diff.json'),'w') as f:
        json.dump(diff,f,ensure_ascii=False,indent=2)
    print(f'[diff] -> {os.path.join(base_out,"rejudge_translation_diff.json")}')

    # patch 官方三份(打分正确的前提下)
    if args.apply and ok_rate>=args.min_ok_rate:
        new_sub=dict(old_sub_d); new_sub['text translation cn']=round(tt_mean,6)
        live_sub=os.path.splitext(old_sub_p)[0] if old_sub_p.endswith('.pre_rejudge') else old_sub_p.replace('.pre_rejudge','')
        live_sub=old_sub_p[:-len('.pre_rejudge')]
        with open(live_sub,'w') as f: json.dump(new_sub,f,ensure_ascii=False,indent=2)
        new_sc=dict(old_sc); new_sc['cn_knowledge_reasoning']=new_cn_kr; new_sc['Chinese Overall Score']=new_cn_overall
        live_score=old_score_p[:-len('.pre_rejudge')]
        with open(live_score,'w') as f: json.dump(new_sc,f,ensure_ascii=False,indent=2)
        # scored.jsonl: translation 行在文件中按 xlsx 行序连续出现 -> 顺序回填
        live_jsonl=old_jsonl[:-len('.pre_rejudge')]
        with open(live_jsonl) as f: lines=f.readlines()
        si=iter(scores); n_written=0
        with open(live_jsonl,'w') as f:
            for ln in lines:
                d=json.loads(ln)
                if d.get('category')=='text translation cn':
                    s=next(si,d.get('score')); d['score']=s; n_written+=1
                    f.write(json.dumps(d,ensure_ascii=False)+'\n')
                else:
                    f.write(ln)
        print(f'[apply] 已 patch: subcategory_score.json / score.json; scored.jsonl 翻译行回填 {n_written} 个')
    elif args.apply:
        print(f'[skip] ok_rate {ok_rate:.3f} < 门槛 {args.min_ok_rate}, 不 patch; 可重跑填 cache 缺口')
    else:
        print('[noop] 未传 --apply (仅 audit/diff); 满意后加 --apply 重跑(默认复用 cache)')

if __name__=='__main__':
    main()
