"""R3-4 (sparsity levels): evaluate trained VoD checkpoints with a fraction of the radar points removed.
Read-only: no repository file is modified, no training. Results are appended to ~/rev1_tmp/sparsity_results.csv.

The same points are removed for every model (random choice seeded by frame id and drop rate),
so the comparison between models is paired. Points are removed from every point array of a sample
(the current five-scan input and the previous inputs used by the temporal module) before voxelization.

Usage (from tools/, GPU 1):
  cd tools/
  CUDA_VISIBLE_DEVICES=1 python tools/analysis/eval_sparsity.py --model B --seed 0 --drop 0
  (sanity check: drop 0 must give about 56.2 entire-area mAP for B seed 0)
  bash tools/scripts/run_sparsity.sh  # optional wrapper       # all models, seeds and drop rates
"""
import os, sys, argparse, json, zlib, time, math
import numpy as np, torch, yaml
from easydict import EasyDict
sys.path.insert(0, os.path.abspath('..'))
from pcdet.config import cfg
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# Paper main-table models (disconnected full RCS model excluded from the public zoo).
# Checkpoints may live under CHECKPOINT_ROOT (default: checkpoints/) or OUTPUT_ROOT.
RUNS = {
    'repro': ('cfgs/vod/m3net_repro.yaml',
              ['m3net_repro/seed0/checkpoint_epoch_78.pth',
               'm3net_repro/seed1/checkpoint_epoch_61.pth',
               'm3net_repro/seed2/checkpoint_epoch_73.pth',
               'm3net_repro/seed3/checkpoint_epoch_73.pth']),
    'B': ('cfgs/vod/4d_mambanet.yaml',
          ['4d_mambanet/seed0/checkpoint_epoch_78.pth',
           '4d_mambanet/seed1/checkpoint_epoch_79.pth',
           '4d_mambanet/seed2/checkpoint_epoch_80.pth',
           '4d_mambanet/seed3/checkpoint_epoch_71.pth']),
}
OUT = os.environ.get('SPARSITY_OUT', 'output/sparsity')

ap = argparse.ArgumentParser()
ap.add_argument('--model', choices=list(RUNS), required=True)
ap.add_argument('--ckpt_root', default=os.environ.get('CHECKPOINT_ROOT', 'checkpoints'))
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--drop', type=float, required=True, help='fraction of radar points removed, e.g. 0.25')
ap.add_argument('--workers', type=int, default=4)
ap.add_argument('--bs', type=int, default=1)
ap.add_argument('--tag', default='')
args = ap.parse_args()

# ---------------- config loading (same as make_fig_order.py: _BASE_CONFIG_ chain + !python/eval)
def _py_eval(loader, node):
    return eval(loader.construct_scalar(node), {'np': np, 'math': math, '__builtins__': __builtins__})
yaml.SafeLoader.add_constructor('!python/eval', _py_eval)
def _merge(b, n):
    for k, v in n.items():
        if isinstance(v, dict) and isinstance(b.get(k), dict): _merge(b[k], v)
        else: b[k] = v
    return b
def _resolve(d):
    if isinstance(d, dict):
        if '_BASE_CONFIG_' in d:
            p = d.pop('_BASE_CONFIG_')
            with open(p) as f: d = _merge(_resolve(yaml.safe_load(f)), d)
        return {k: _resolve(v) for k, v in d.items()}
    return d
CFG, CKPTS = RUNS[args.model]
CFG = CFG if os.path.isabs(CFG) else os.path.join(os.path.dirname(__file__), '..', CFG)
with open(CFG) as f:
    cfg.update(EasyDict(_resolve(yaml.safe_load(f))))
cfg.TAG = os.path.splitext(os.path.basename(CFG))[0]; cfg.EXP_GROUP_PATH = 'vod_models/last_model'
ckpt = os.path.join(args.ckpt_root, CKPTS[args.seed])
if not os.path.exists(ckpt):
    ckpt = os.path.join(os.environ.get('OUTPUT_ROOT', '../output'), CKPTS[args.seed])
assert os.path.exists(ckpt), ckpt

logger = common_utils.create_logger()
ds, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=args.bs,
                                 dist=False, workers=args.workers, training=False, logger=logger)

# ---------------- point removal before voxelization
_orig_prepare = ds.prepare_data
_shown = {'done': False}
def prepare_with_drop(data_dict):
    if args.drop > 0:
        fid = str(data_dict.get('frame_id', ''))
        for k in list(data_dict.keys()):
            v = data_dict[k]
            if 'points' in k and isinstance(v, np.ndarray) and v.ndim == 2 and len(v) > 1:
                rng = np.random.default_rng(zlib.crc32(f'{fid}|{k}|{args.drop}'.encode()))
                keep = rng.random(len(v)) >= args.drop
                if keep.sum() == 0: keep[rng.integers(len(v))] = True
                data_dict[k] = v[keep]
                if not _shown['done']:
                    print(f'[drop] key {k}: {len(v)} -> {keep.sum()} points')
        _shown['done'] = True
    return _orig_prepare(data_dict=data_dict)
ds.prepare_data = prepare_with_drop

model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
model.cuda().eval()

det_annos, t0 = [], time.time()
with torch.no_grad():
    for i, batch in enumerate(loader):
        load_data_to_gpu(batch)
        pred_dicts, _ = model(batch)
        det_annos += ds.generate_prediction_dicts(batch, pred_dicts, cfg.CLASS_NAMES, output_path=None)
        if i % 300 == 0: print(f'{i}/{len(loader)}  {time.time() - t0:.0f}s', flush=True)

run_dir = os.path.join(OUT, 'sparsity_runs', f'{args.model}_s{args.seed}_d{args.drop}{args.tag}')
os.makedirs(run_dir, exist_ok=True)   # the VoD evaluation writes map.txt here
import pickle, inspect, re
with open(os.path.join(run_dir, 'det_annos.pkl'), 'wb') as f:   # keep predictions: evaluation can be re-run without inference
    pickle.dump(det_annos, f)
# supply every kwargs['...'] key that the repo's VoD evaluation reads
src = inspect.getsource(type(ds).evaluation)
need = sorted(set(re.findall(r"kwargs\[['\"](\w+)['\"]\]", src)))
defaults = {'result_dir': run_dir, 'epoch_id': f'sparsity_{args.model}_s{args.seed}_d{args.drop}',
            'eval_metric': cfg.MODEL.POST_PROCESSING.get('EVAL_METRIC', 'kitti')}
kw = {k: defaults.get(k, None) for k in need}
kw.setdefault('eval_metric', defaults['eval_metric'])
print('evaluation kwargs:', {k: str(v) for k, v in kw.items()})
res = ds.evaluation(det_annos, cfg.CLASS_NAMES, **kw)
result_str, result_dict = (res if isinstance(res, tuple) else (str(res), res))
print(result_str)
rec = dict(model=args.model, seed=args.seed, drop=args.drop, bs=args.bs, tag=args.tag, ckpt=ckpt,
           result={k: float(v) if np.isscalar(v) else str(v) for k, v in dict(result_dict).items()})
with open(os.path.join(OUT, f'sparsity_results{args.tag}.jsonl'), 'a') as f:
    f.write(json.dumps(rec) + '\n')
with open(os.path.join(OUT, f'sparsity_{args.model}_s{args.seed}_d{args.drop}{args.tag}.txt'), 'w') as f:
    f.write(str(result_str))
print('saved', args.model, args.seed, args.drop)
