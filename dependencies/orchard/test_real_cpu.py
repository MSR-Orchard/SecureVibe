import sys, tempfile,json,copy,os,argparse
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
parser=argparse.ArgumentParser(description="Real CPU tokenizer and sampler checkpoint checks")
parser.add_argument('--slime-dir',required=True)
parser.add_argument('--student-tokenizer',required=True)
parser.add_argument('--teacher-tokenizer',required=True)
cli=parser.parse_args()
sys.path[:0]=[cli.slime_dir,str(Path(__file__).resolve().parents[2]/'training')]
from transformers import AutoTokenizer
from slime_rl.joint_data_source import JointRolloutDataSource
from slime.rollout.on_policy_distillation import _build_hinted_teacher_input_ids,_is_same_vocab
student=AutoTokenizer.from_pretrained(cli.student_tokenizer,local_files_only=True)
teacher=AutoTokenizer.from_pretrained(cli.teacher_tokenizer,local_files_only=True)
assert _is_same_vocab(student,teacher)
for text in ['Check input types.','检查路径，拒绝 ../。','Unicode: café 🔒','Literal {hint} and <|im_end|>']:
 prompt=student.encode('Implement safe input handling.',add_special_tokens=False)
 response=student.encode('I will validate the input.',add_special_tokens=False)
 sample=SimpleNamespace(tokens=prompt+response,response_length=len(response),metadata={'hint':text})
 before=copy.deepcopy(sample.__dict__)
 cfg=SimpleNamespace(hint_metadata_key='hint',hint_template='\nTeacher: {hint}\n')
 ids=_build_hinted_teacher_input_ids(sample,cfg,student,teacher)
 assert ids==prompt+student.encode(cfg.hint_template.format(hint=text),add_special_tokens=False)+response
 assert sample.__dict__==before
print('Real tokenizer: exact vocab match; four hint/token-boundary cases passed.')
with tempfile.TemporaryDirectory(prefix='safevibe-sampler-') as tmp:
 root=Path(tmp); cfg=root/'config.yaml';cfg.write_text('agent: {}\n')
 for kind,profile in [('patcheval','repo_patch'),('autobax','app_builder')]:
  rows=[{'prompt':f'Synthetic {kind} task {i}','label':str(i),'metadata':{'task_type':kind,'reward_type':kind,'agent_profile':profile,'swe_config_path':str(cfg)}} for i in range(7)]
  (root/f'{kind}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
 args=SimpleNamespace(hf_checkpoint=cli.student_tokenizer,joint_patcheval_data=str(root/'patcheval.jsonl'),joint_autobax_data=str(root/'autobax.jsonl'),joint_patcheval_weight=.5,joint_autobax_weight=.5,n_samples_per_prompt=2,n_samples_per_prompt_max=4,rollout_seed=42,rollout_shuffle=True,save=tmp,load=tmp)
 a=JointRolloutDataSource(args); used=a.get_samples(17);a.add_samples(used[-3:]);a.save(7)
 expected=[[asdict(s) for s in g] for g in a.get_samples(75)]
 b=JointRolloutDataSource(args);b.load(7)
 actual=[[asdict(s) for s in g] for g in b.get_samples(75)]
 assert actual==expected
 assert b._state_dict()==a._state_dict()
 print('Real torch checkpoint: 75 groups identical after resume, including buffered groups, indices, RNG and epoch rollover.')
 original=args.joint_patcheval_weight;args.joint_patcheval_weight=.25;args.joint_autobax_weight=.75
 try:JointRolloutDataSource(args).load(7)
 except ValueError:print('Changed sampling weights correctly rejected.')
 else:raise AssertionError('weights mismatch not rejected')
 args.joint_patcheval_weight=.5;args.joint_autobax_weight=.5
 with (root/'patcheval.jsonl').open('a') as f:f.write('\n')
 try:JointRolloutDataSource(args).load(7)
 except ValueError:print('Changed dataset fingerprint correctly rejected.')
 else:raise AssertionError('data mismatch not rejected')
