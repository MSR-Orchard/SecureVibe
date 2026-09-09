"""CPU-only contract tests against the actual patched Slime source.

Run: python test_compatibility.py /path/to/Orchard/trainer/slime
Pure functions are compiled from source to avoid importing the GPU stack.
"""
import ast
import asyncio
import copy
import importlib.util
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

SLIME = Path(sys.argv.pop(1)).resolve()
TRAINING = Path(__file__).resolve().parents[2] / 'training'
sys.path.insert(0, str(TRAINING))
import orchard_compat

source = ast.parse((SLIME / 'slime/rollout/on_policy_distillation.py').read_text())
names = {'_is_same_vocab', '_build_hinted_teacher_input_ids', '_truncate_to_teacher_max_len'}
selected = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
namespace = {'Sample': SimpleNamespace, 'logger': logging.getLogger('test')}
exec(compile(ast.Module(body=selected, type_ignores=[]), '<patched-opd>', 'exec'), namespace)


class Tokenizer:
    def __init__(self, mismatch=False):
        self.vocab = {str(i): i for i in range(200)}
        if mismatch:
            self.vocab['199'] = 300
    def __len__(self): return len(self.vocab)
    def get_vocab(self): return self.vocab
    def encode(self, text, add_special_tokens=False): return [100, 101]


class CompatibilityTests(unittest.TestCase):
    def test_hint_preserves_student_and_response(self):
        sample = SimpleNamespace(tokens=[1, 2, 3, 4], response_length=2,
                                 metadata={'teacher_hint': 'private hint'}, prompt='student')
        before = copy.deepcopy(sample.__dict__)
        cfg = SimpleNamespace(hint_metadata_key='teacher_hint', hint_template='{hint}')
        ids = namespace['_build_hinted_teacher_input_ids'](sample, cfg, Tokenizer(), Tokenizer())
        self.assertEqual(ids, [1, 2, 100, 101, 3, 4])
        self.assertEqual(sample.__dict__, before)
        trimmed, prompt_drop, response_drop = namespace['_truncate_to_teacher_max_len'](ids, 4, 2, 'prefix')
        self.assertEqual(trimmed, [100, 101, 3, 4])
        self.assertEqual((prompt_drop, response_drop), (2, 0))

    def test_full_vocab_mismatch_rejected(self):
        sample = SimpleNamespace(tokens=[1, 2], response_length=1, metadata={'hint': 'x'})
        cfg = SimpleNamespace(hint_metadata_key='hint', hint_template='{hint}')
        with self.assertRaises(ValueError):
            namespace['_build_hinted_teacher_input_ids'](sample, cfg, Tokenizer(), Tokenizer(True))

    def test_unhinted_and_invalid_hint(self):
        cfg = SimpleNamespace(hint_metadata_key='hint', hint_template='{hint}')
        sample = SimpleNamespace(tokens=[1, 2], response_length=1, metadata={})
        self.assertEqual(namespace['_build_hinted_teacher_input_ids'](sample, cfg, None, None), [1, 2])
        sample.metadata['hint'] = {'invalid': True}
        with self.assertRaises(ValueError):
            namespace['_build_hinted_teacher_input_ids'](sample, cfg, None, None)

    def test_response_truncation_accounting(self):
        truncate = namespace['_truncate_to_teacher_max_len']
        self.assertEqual(truncate([1,2,3,4,5], 2, 3, 'prefix'), ([4,5], 2, 1))
        self.assertEqual(truncate([1,2,3,4,5], 4, 3, 'suffix'), ([1,2,3,4], 0, 1))

    def test_config_precedence_and_missing_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp)/'first.yaml', Path(tmp)/'second.yaml'
            first.touch(); second.touch()
            sample = SimpleNamespace(metadata={'task_type':'autobax','swe_config_path':str(first)})
            args = SimpleNamespace(swe_config_path=str(second))
            self.assertEqual(orchard_compat.resolve_swe_config_path(args,sample), str(first))
            del sample.metadata['swe_config_path']
            with patch.dict(os.environ, {'AUTOBAX_CONFIG_PATH':str(first)}, clear=True):
                self.assertEqual(orchard_compat.resolve_swe_config_path(None,sample), str(first))
            sample.metadata['swe_config_path'] = str(first)+'missing'
            with self.assertRaises(FileNotFoundError): orchard_compat.resolve_swe_config_path(args,sample)

    def test_concurrent_routing_does_not_mutate_args(self):
        module = ModuleType('examples.orchard_swe.swe_generate_v2')
        async def fake_generate(args,sample,params):
            await asyncio.sleep(0)
            return args.swe_config_path
        module.generate=fake_generate
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {module.__name__:module}):
            paths=[Path(tmp)/'a.yaml',Path(tmp)/'b.yaml']
            for p in paths:p.touch()
            args=SimpleNamespace(swe_config_path='unchanged')
            samples=[SimpleNamespace(metadata={'swe_config_path':str(p)}) for p in paths]
            async def run():
                return await asyncio.gather(*(orchard_compat.generate(args,s,{}) for s in samples))
            self.assertEqual(asyncio.run(run()),list(map(str,paths)))
            self.assertEqual(args.swe_config_path,'unchanged')

    def test_combined_reward_evaluation_and_teacher_failure(self):
        import traceback
        tree=ast.parse((TRAINING/'slime_opd/combined_reward.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='reward_func')
        calls=[]
        async def task(*args,**kwargs):return 0.5
        async def teacher(*args,**kwargs):
            calls.append(True)
            raise asyncio.TimeoutError()
        ns={'Sample':SimpleNamespace,'asyncio':asyncio,'traceback':traceback,
            'logger':logging.getLogger('test'),'_task_reward_func':task,
            '_get_opd_timeout':lambda args:1,'opd_reward_func':teacher}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<combined-reward>','exec'),ns)
        sample=SimpleNamespace(metadata={},teacher_log_probs=None)
        self.assertEqual(asyncio.run(ns['reward_func'](None,sample,evaluation=True)),0.5)
        self.assertEqual(calls,[])
        self.assertEqual(asyncio.run(ns['reward_func'](None,sample)),0.5)
        self.assertEqual(calls,[True])
        self.assertIsNone(sample.teacher_log_probs)

    def test_core_calls_hint_builder_and_propagates_evaluation(self):
        reward=next(n for n in source.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='reward_func')
        self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                            and n.func.id=='_build_hinted_teacher_input_ids' for n in ast.walk(reward)))
        rollout=ast.parse((SLIME/'slime/rollout/sglang_rollout.py').read_text())
        calls=[n for n in ast.walk(rollout) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
               and n.func.id in ('async_rm','batched_async_rm')]
        self.assertTrue(calls)
        for call in calls:self.assertIn('evaluation',[kw.arg for kw in call.keywords])


if __name__ == '__main__': unittest.main()
