"""Frozen, pinned IBM MoLFormer inference on an authorized CUDA host."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel,AutoTokenizer

MODEL='ibm-research/MoLFormer-XL-both-10pct'
REVISION='361063d0ad524ef77cf39b08469f6be770dc550f'
CODE_HASHES={'configuration_molformer.py':'b88ea8d4b7b5e54f4f186cc7a230eff308928020a030f153a038fd12c05e3bed',
             'modeling_molformer.py':'6f1ef72022de2c69e95661899422a7bb39a40a2cc5a6cb6216f14e9b7d84559c',
             'config.json':'3ef9eaac8c7ca6282fd6256ed038d151bd4ff42a4ff855367e0d7197bbc1c284'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(args):
    start=time.monotonic();args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    data=json.loads(args.input.read_text());smiles=data['canonical_inputs']
    assert torch.cuda.is_available()
    folder=Path(snapshot_download(MODEL,revision=REVISION,
                                 allow_patterns=['*.json','*.py','model.safetensors','README.md']))
    assert all(sha(folder/name)==digest for name,digest in CODE_HASHES.items())
    # The pinned publisher code was inspected before use; no unpinned code is loaded.
    tokenizer=AutoTokenizer.from_pretrained(folder,trust_remote_code=False)
    model=AutoModel.from_pretrained(MODEL,revision=REVISION,code_revision=REVISION,
                                   trust_remote_code=True,use_safetensors=True,
                                   deterministic_eval=True).eval().requires_grad_(False).cuda()
    vectors=[];lengths=[];unknown=0;repeat_delta=None
    unique=list(dict.fromkeys(smiles));index={s:i for i,s in enumerate(unique)}
    with torch.inference_mode():
        for offset in range(0,len(unique),16):
            batch=tokenizer(unique[offset:offset+16],padding=True,truncation=False,return_tensors='pt')
            lengths.extend(batch.attention_mask.sum(1).tolist())
            if tokenizer.unk_token_id is not None:unknown+=int((batch.input_ids==tokenizer.unk_token_id).sum())
            inputs={k:v.cuda() for k,v in batch.items()}
            out=model(**inputs).pooler_output
            if repeat_delta is None:
                repeat_delta=float((out-model(**inputs).pooler_output).abs().max())
                assert repeat_delta<1e-5
            vectors.append(out.cpu().numpy())
            print('encoded',min(offset+16,len(unique)),'of',len(unique),flush=True)
    x=np.concatenate(vectors)[[index[s] for s in smiles]]
    assert np.isfinite(x).all()
    np.savez_compressed(args.output,vectors=x)
    meta={**data,'model':MODEL,'revision':REVISION,'source_code_sha256':CODE_HASHES,
          'weights_sha256':sha(folder/'model.safetensors'),'extractor_sha256':sha(__file__),
          'input_sha256':sha(args.input),'embedding_sha256':sha(args.output),
          'dimension':int(x.shape[1]),'rows':len(x),'precision':'float32',
          'pooling':'publisher pooler_output','unknown_token_occurrences':unknown,
          'max_tokens':max(lengths),'inputs_exceeding_pretraining_202_tokens':sum(n>202 for n in lengths),
          'truncation':False,'repeated_inference_delta':repeat_delta,
          'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'seconds':time.monotonic()-start}
    args.output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({k:meta[k] for k in ['rows','dimension','seconds','unknown_token_occurrences','max_tokens']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
