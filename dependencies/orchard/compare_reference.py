#!/usr/bin/env python3
"""CPU differential checks. Usage: compare_reference.py PATCHED_SLIME REFERENCE_SLIME

The reference checkout is read through Git HEAD; no reference content is saved.
GPU imports are excluded. Real pure/async functions execute with mocked I/O.
"""
import ast
import asyncio
import copy
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import typing

logging.disable(logging.CRITICAL)
PATCHED, REFERENCE = map(Path, sys.argv[1:3])
TRAINING = Path(__file__).resolve().parents[2] / 'training'


def reference(path):
    return subprocess.check_output(['git','-C',str(REFERENCE),'show','HEAD:'+path],text=True)


def functions(text, names=None):
    tree=ast.parse(text)
    body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))
          and (names is None or n.name in names)]
    ns={'asyncio':asyncio,'os':os,'re':re,'Sample':SimpleNamespace,
        'logger':logging.getLogger('test'),'Any':typing.Any,'_OUTPUT_LIMIT':8000}
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,*body],type_ignores=[])),
                 '<source-under-test>','exec'),ns)
    return ns


class Tokenizer:
    def __len__(self): return 256
    def get_vocab(self): return {str(i):i for i in range(256)}
    def encode(self,text,add_special_tokens=False): return [90,91,92]


async def teacher_case(text,hinted,limit,side,short):
    names={'reward_func','_build_hinted_teacher_input_ids','_is_same_vocab',
           '_truncate_to_teacher_max_len','_extract_token_logprobs'}
    ns=functions(text,names)
    cfg=SimpleNamespace(teacher_max_len=limit,teacher_truncation_side=side,
                        hint_metadata_key='hint',hint_template='{hint}',url='mock',timeout=1)
    requests=[]
    class Response:
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        def raise_for_status(self):pass
        async def json(self):
            payload=requests[-1]
            tokens=payload['input_ids'][payload['logprob_start_len']:]
            rows=[[None, tokens[0]]]+[[-token/100,token] for token in tokens[1:]]
            if short: rows=rows[:max(1,len(rows)-1)]
            return {'meta_info':{'input_token_logprobs':rows}}
    class Session:
        def __init__(self,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        def post(self,url,json):requests.append(copy.deepcopy(json));return Response()
    ns.update(_load_config=lambda args:cfg,_teacher_max_len=None,
              _get_hint_tokenizers=lambda *args:(Tokenizer(),Tokenizer()),
              _DEFAULT_TEACHER_TIMEOUT=1,
              aiohttp=SimpleNamespace(ClientSession=Session,ClientTimeout=lambda **kwargs:None))
    sample=SimpleNamespace(tokens=[1,2,3,4,5,6],response_length=3,
                           metadata={'hint':'secret'} if hinted else {},prompt='student')
    before=copy.deepcopy(sample.__dict__)
    result=await ns['reward_func'](None,sample)
    assert sample.__dict__==before, 'Student sample mutated'
    assert len(result['valid_mask'])==3
    return requests,result


async def grade_case(text,functional,security,apply_ok=True,valid=True):
    ns=functions(text)
    calls=[]
    async def create(*args,**kwargs):calls.append('create');return object()
    async def stop(env):calls.append('stop')
    async def apply(env,workdir,patch,name):
        calls.append(name);return (apply_ok if 'model' in name else True),'mock'
    async def write(*args):pass
    async def execute(env,command):
        return {'returncode':functional if 'functional_eval' in command else security,'output':'mock'}
    ns.update(create_environment=create,stop_environment=stop,_apply_repo_patch=apply,
              _valid_patch=lambda p:bool(p),_filter_diff=lambda p,paths:p,
              _test_file_paths=lambda p:set(),_write_file=write,_execute=execute,
              resolve_swe_config_path=lambda *a,**kw:'config',
              _resolve_swe_config_path=lambda *a,**kw:'config',load_config=lambda p:{})
    sample=SimpleNamespace(index=0,metadata={'final_output':'patch' if valid else '',
        'mask_patch':'mask','test_patch':'tests','grade_image_name':'mock-image',
        'functional_eval_cmd':'functional','security_eval_cmd':'security'})
    value=await ns['_evaluate'](None,sample)
    assert ('stop' in calls)==('create' in calls), 'Sandbox cleanup missing'
    return value,sample.metadata.get('reward_components'),calls


async def main():
    core=('slime/rollout/on_policy_distillation.py','slime/rollout/opd_config.py',
          'slime/rollout/sglang_rollout.py','slime/utils/arguments.py')
    for path in core:
        assert (PATCHED/path).read_text()==reference(path), f'Core differs: {path}'
    print('All four patched core files match reference HEAD byte-for-byte.')
    public=(PATCHED/core[0]).read_text(); ref=reference(core[0]);count=0
    for hint in (False,True):
        for limit in (2,4,7,20):
            for side in ('prefix','suffix'):
                for short in (False,True):
                    assert await teacher_case(public,hint,limit,side,short)==await teacher_case(ref,hint,limit,side,short)
                    count+=1
    print(f'{count} teacher request/log-probability/mask comparisons passed; student tokens unchanged.')
    local=(TRAINING/'slime_opd/patcheval_reward.py').read_text()
    ref=reference('examples/security_coding_agent_opd/reward.py')
    for args in ((0,0),(0,1),(1,0),(1,1),(0,0,False),(0,0,True,False)):
        assert await grade_case(local,*args)==await grade_case(ref,*args),args
    print('6 PatchEval reward/patch-failure/cleanup comparisons passed.')


if __name__=='__main__':asyncio.run(main())
