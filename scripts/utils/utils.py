import gzip
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Set, Union

import tiktoken


def print_(message: str, fun: str = "*", count: int = 1) -> None:
    """
    Helper to print a fun message.
    """
    banner = (fun + " ") * count
    print(f"{banner.strip()} {message} {banner.strip()}")


def merge_dicts(a: Dict[Any, Any], b: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Helper to recursively merge two dictionaries.
    """
    result = a.copy()
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def count_tokens_tiktoken(text: str, model: str = "gpt-4o") -> int:
    """
    Helper that returns token counts of a string using tiktoken.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    text = text or ""
    return len(encoding.encode(text))


def ungzip_file(gz_path, output_path):
    """
    Helper to unzip a .gz file to the specified output path.
    """
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print_(f"Unzipped {gz_path} to {output_path}")


def load_json(path: Union[str, Path]) -> Any:
    """
    Helper to load a JSON (.json) file and return its contents as a dictionary.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Helper to load a JSON Lines (.jsonl) file and return its contents as a list of dictionaries.
    """
    path = Path(path)
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_completed_task_ids(results_file: Union[str, Path]) -> Set[int]:
    """
    Helper that loads a set of task IDs already completed from a results JSON file.
    """
    results_file = Path(results_file)
    completed = set()
    if results_file.exists():
        try:
            items = json.loads(results_file.read_text(encoding="utf-8"))
            for res in items:
                tid = res.get("task_id")
                if isinstance(tid, int):
                    completed.add(tid)
        except Exception as e:
            print_(f"Warning: could not load completed task IDs: {e}")
    return completed


def append_result(result: Dict[str, Any], results_file: Union[str, Path]) -> None:
    """
    Helper to append a new result to a results JSON file.
    """
    results_file = Path(results_file)
    results: List[Dict[str, Any]] = []
    if results_file.exists():
        try:
            results = json.loads(results_file.read_text(encoding="utf-8"))
        except Exception as e:
            print_(f"Warning: could not read existing results: {e}")
    results.append(result)
    results_file.write_text(json.dumps(results, indent=4), encoding="utf-8")