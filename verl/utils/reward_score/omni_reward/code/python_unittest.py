import re
import json
import random
import asyncio
from typing import Union, List, Optional
from sandbox_fusion import run_jupyter_async, RunJupyterRequest, RunCodeRequest, run_code_async

from verl.utils.reward_score.utils import ErrCode



async def sandbox_run_jupyter(
        cells: List, 
        service_url: Union[str, List],
        max_attempts: int=2, 
        timeout: int=5,
    ):
    """Execute python code in sandbox."""
    if isinstance(service_url, list):
        sandbox_address = random.choice(service_url)
    elif isinstance(service_url, str):
        sandbox_address = service_url
    else:
        raise ValueError(f"Invalid sandbox_address: {service_url}")
    
    try:
        resp = await run_jupyter_async(
            RunJupyterRequest(cells=cells, total_timeout=timeout),
            endpoint=f"http://{sandbox_address}",
            max_attempts=max_attempts,
            client_timeout=(timeout * max_attempts + 5),
        )
    except Exception as e:
        print(f"sandbox exception: {e}, {type(e)=}")
        return

    if resp.cells:
        return resp.cells[-1]
    else:
        print("[python_unittest] Warning: no cells in response")
        return
    
async def sandbox_run_code(
        code: str, 
        service_url: Union[str, List],
        stdin: Optional[str] = None,
        language: str = "python",
        max_attempts: int=2, 
        timeout: int=5,
    ):
    """Execute python code in sandbox."""
    if isinstance(service_url, list):
        sandbox_address = random.choice(service_url)
    elif isinstance(service_url, str):
        sandbox_address = service_url
    else:
        raise ValueError(f"Invalid sandbox_address: {service_url}")
    
    try:
        resp = await run_code_async(
            RunCodeRequest(
                code=code, 
                stdin=stdin,
                language=language,
                total_timeout=timeout),
            endpoint=f"http://{sandbox_address}",
            max_attempts=max_attempts,
            client_timeout=(timeout * max_attempts + 5),
        )
        return resp.run_result
    except Exception as e:
        print(f"sandbox exception: {e}, {type(e)=}")
        return
    

def extract_answer(pred):
    python_codes = re.findall(r"```python\s*(.*?)```", pred, re.DOTALL)
    if len(python_codes):
        return python_codes[-1]
    else:
        return None


async def compute_score_entry_point_import_prefix_test_code(code, gt, service_url, is_val):
    code_list = [
        gt['import_prefix'], 
        code,
        gt['test_code'], 
        f"check({gt['entry_point']})"
    ]

    result = await sandbox_run_jupyter(code_list, service_url)
    if result is None or getattr(result, 'error', None):
        return {"score": 0, "error_code": ErrCode.WRONG_ANSWER}
    else:
        return {"score": 1, "error_code": ErrCode.SUCCESS}


async def compute_score_fn_name_inputs_outputs(code, gt, service_url, is_val):
    results = []
    input_outputs = list(zip(gt['inputs'], gt['outputs']))
    if not is_val:
        input_outputs = random.sample(input_outputs, min(8, len(input_outputs)))
    for inp, outp in input_outputs:
        code_list = [
            code,
            f"Solution.{gt['fn_name']}({inp})"
        ]
        res = await sandbox_run_jupyter(code_list, service_url)

        # parse
        if res is None:
            results.append(0)
        elif res.stdout.strip() == outp.strip():
            results.append(1)
        elif getattr(res, "display", None) and res.display and \
            res.display[0].get("text/plain", "").strip() == outp.strip():
            results.append(1)
        else:
            results.append(0)

    score = sum(results) / len(results)
    if score > 0:
        return {"score": score, "error_code": ErrCode.SUCCESS}
    else:
        return {"score": 0, "error_code": ErrCode.WRONG_ANSWER} 

async def compute_score_inputs_outputs(code, gt, service_url, is_val):
    results = []

    input_outputs = list(zip(gt['inputs'], gt['outputs']))
    if not is_val:
        input_outputs = random.sample(input_outputs, min(8, len(input_outputs)))
    for inp, outp in input_outputs:
        
        res = await sandbox_run_code(code, stdin=str(inp), service_url=service_url)

        # parse
        if res is None:
            results.append(0)
        elif str(res.stdout).strip() == str(outp).strip():
            results.append(1)
        else:
            results.append(0)

    score = sum(results) / len(results)
    if score > 0:
        return {"score": score, "error_code": ErrCode.SUCCESS}
    else:
        return {"score": 0, "error_code": ErrCode.WRONG_ANSWER} 


async def _compute_score(code, gt, service_url, is_val):
    if 'entry_point' in gt and 'import_prefix' in gt and 'test_code' in gt:
        return await compute_score_entry_point_import_prefix_test_code(code, gt, service_url, is_val)
    elif 'fn_name' in gt and 'inputs' in gt and 'outputs' in gt:
        return await compute_score_fn_name_inputs_outputs(code, gt, service_url, is_val)
    elif 'inputs' in gt and 'outputs' in gt:
        return await compute_score_inputs_outputs(code, gt, service_url, is_val)
    else:
        return {"score": 0, "error_code": ErrCode.GROUND_TRUTH_ERR}


def compute_score(pred, gt, **kwargs):
    eos_token = kwargs.get("eos_token", "<|im_end|>")
    service_url = kwargs.get("service_url", None)
    is_val = kwargs.get("is_val", False)
    assert service_url is not None, f"service_url not set for code reward!"

    # # format check
    if kwargs.get('check_eos_flag', False):
        if not pred.rstrip().endswith(eos_token):
            return {"score": 0, "error_code": ErrCode.FMT_ERR}
    
    code = extract_answer(pred)
    if code is None:
        return {"score": 0, "error_code": ErrCode.FMT_ERR}
    
    # process gt
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except json.JSONDecodeError:
            return {"score": 0, "error_code": ErrCode.GROUND_TRUTH_ERR}
    elif isinstance(gt, dict):
        pass
    else:
        return {"score": 0, "error_code": ErrCode.GROUND_TRUTH_ERR}
    
    return asyncio.run(_compute_score(code, gt, service_url, is_val))


