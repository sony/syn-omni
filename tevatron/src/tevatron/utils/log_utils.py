import json
import logging
import os
import sys

from dataclasses import asdict


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


class _Tee:
    """Duplicate writes to (original stream) + (file)."""

    def __init__(self, stream, f):
        self.stream = stream
        self.f = f

    def write(self, s):
        self.stream.write(s)
        self.f.write(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def setup_rank_logging(training_args):
    # rank id
    rank = (
        training_args.local_rank
        if training_args.local_rank is not None and training_args.local_rank != -1 else 0  # noqa
    )

    logs_dir = os.path.join(training_args.output_dir, "logs")
    _ensure_dir(logs_dir)

    # log file path
    log_path = os.path.join(logs_dir, f"train_rank{rank}.log")
    f = open(log_path, "a", buffering=1, encoding="utf-8")

    # 1) logging -> stream + file
    root = logging.getLogger()
    root.setLevel(logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN)

    fmt = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    # clear handlers set by basicConfig (avoid duplicates)
    for h in list(root.handlers):
        root.removeHandler(h)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # 1) also tee stdout/stderr to file (captures prints / tqdm / traceback, etc.)
    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)

    return log_path


def save_parameters(output_dir: str, model_args, data_args, training_args):
    params_dir = os.path.join(output_dir, "parameters")
    _ensure_dir(params_dir)

    def _dump(obj, name):
        path = os.path.join(params_dir, name)
        with open(path, "w", encoding="utf-8") as wf:
            json.dump(obj, wf, ensure_ascii=False, indent=2)

    _dump(asdict(model_args), "model_args.json")
    _dump(asdict(data_args), "data_args.json")
    _dump(asdict(training_args), "training_args.json")

    all_params = {}
    all_params.update(asdict(model_args))
    all_params.update(asdict(data_args))
    all_params.update(asdict(training_args))
    _dump(all_params, "all_parameters.json")
