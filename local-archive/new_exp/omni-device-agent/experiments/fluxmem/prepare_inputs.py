"""Use the author's actual video reader, prompt and processor before model load."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import resource
import sys
import traceback

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(args.upstream/'models/qwen2-5-vl/src'),
                   str(args.upstream/'models/qwen-vl-utils/src')]
    import torch
    from qwen2_5_vl_fluxmem import Qwen2_5_VLProcessor
    from qwen_vl_utils_fluxmem import process_vision_info
    from qwen_vl_utils_fluxmem import vision_process
    from bounded_reader import read_video_bounded
    vision_process.VIDEO_READER_BACKENDS["torchvision"] = read_video_bounded
    from measure import Meter
    meter = Meter(torch, device='cpu')
    manifest = json.loads(args.data_manifest.read_text())
    code = args.upstream/'evaluation/streamingbench/streamingbench.py'
    tree = ast.parse(code.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StreamingBenchEvaluator')
    prompt_node = next(n for n in cls.body if isinstance(n, ast.Assign) and
                       any(isinstance(t, ast.Name) and t.id == 'PROMPT_TEMPLATE' for t in n.targets))
    template = ast.literal_eval(prompt_node.value)
    result = {'status': 'running', 'dataset': manifest, 'records': [],
              'preprocess_scheduling': 'separate process before model load',
              'template_source_sha256': hashlib.sha256(code.read_bytes()).hexdigest()}
    def save():
        (args.output/'prepared.json').write_text(json.dumps(result, indent=2)+'\n')
    save()
    try:
        processor = Qwen2_5_VLProcessor.from_pretrained(args.model, local_files_only=True,
                                                      min_pixels=16*28*28, max_pixels=128*28*28)
        for idx, row in enumerate(manifest['queries']):
            parts = list(map(int, row['time_stamp'].split(':')))
            seconds = parts[0]*3600 + parts[1]*60 + parts[2]
            prompt = template.format(question=row['question'], options='\n'.join(ast.literal_eval(row['options'])))
            messages = [{'role': 'user', 'content': [
                {'type': 'video', 'video': manifest['video'], 'min_pixels':16*28*28,
                 'max_pixels':128*28*28, 'max_frames':96, 'min_frames':2,
                 'fps':1.0, 'video_start':0, 'video_end':seconds, 'anchor_end':False},
                {'type':'text','text':prompt}]}]
            with meter.phase('video_read_and_processor', query=idx) as event:
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images, videos, kwargs = process_vision_info(messages, return_video_kwargs=True)
                inputs = processor(text=[text], images=images, videos=videos,
                                   padding=True, return_tensors='pt', **kwargs)
            target = args.output/f'query_{idx}.pt'
            torch.save(dict(inputs), target)
            result['records'].append({'index':idx, 'query':row, 'cutoff_seconds':seconds,
                'input_path':str(target.resolve()), 'input_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                'input_tokens':inputs['input_ids'].shape[1],
                'video_grid_thw':inputs['video_grid_thw'].tolist(),
                'sampled_video_shapes':[list(v.shape) for v in videos],
                'second_per_grid_ts':inputs.get('second_per_grid_ts'), 'measurement':event,
                'sampling_trace':read_video_bounded.last_trace})
            del inputs, videos, images
            save()
            print('PREPARED',idx,seconds,flush=True)
        result['status']='completed'
    except Exception:
        result.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        result['process_maxrss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        meter.close()
        save()

if __name__ == '__main__':
    main()
