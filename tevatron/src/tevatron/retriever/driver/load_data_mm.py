# Download and build all training datasets in a config, then print their sizes.
# python -m tevatron.retriever.driver.load_data_mm \
#     --data_root <path> --train_yaml configs/syn_omni.yaml --dataset_download True
import logging

from collections import defaultdict

from tabulate import tabulate
from transformers import HfArgumentParser, set_seed

from tevatron.retriever.arguments import DataArguments
from tevatron.retriever.datasets import build_train_datasets

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(DataArguments)
    data_args, = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.info("DATA parameters %s", data_args)
    set_seed(42)

    datasets = build_train_datasets(data_args)

    metas = [{k: v for k, v in m.items() if k not in ("name", "data")} for m in datasets.sub_meta]
    total = {
        "view": "Sum:",
        "original size": sum(m["original size"] for m in metas),
        "subset size": sum(m["subset size"] for m in metas),
    }
    print()
    print(tabulate(metas + [total], headers="keys", tablefmt="github"))
    print()

    composition_sum = defaultdict(int)
    for m in metas:
        composition_sum[m["modality_composition"]] += m["subset size"]
    print("samples per modality composition:")
    for comp, n in sorted(composition_sum.items(), key=lambda x: x[1], reverse=True):
        print(f"  {comp}: {n}")


if __name__ == "__main__":
    main()
