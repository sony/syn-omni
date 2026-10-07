"""
from datasets import (
    load_dataset,
    Features,
    Value,
)

features = Features({
    "id": Value("string"),
    "modality": Value("string"),
    "corpus-id": Value("string"),
    "audio_path": Value("string"),
    "audio_bytes": Value("binary"),
})

corpus = load_dataset(
    "parquet",
    data_files=corpus_path,
    features=features,
)
"""
import os

from collections import defaultdict

from datasets import load_dataset

from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from src.model.processor import process_input_text
from src.utils.basic_utils import print_master
from src.utils.hf_utils import download_sounddescs

TASK_INST_QRY = "Find an audio that contains the following content:"
TASK_INST_TGT = "Understand the content of the provided audio."


@add_metainfo_hook
def _prepare_queries(batch_dict, text2audio, **kwargs):
    model_backbone = kwargs['model_backbone']

    query_texts, query_images, query_audio = [], [], []
    dataset_infos = []

    for ind in range(len(batch_dict["id"])):
        qid = batch_dict["query-id"][ind]
        caption = batch_dict["text"][ind].strip()
        assert qid in text2audio
        corpus_ids = text2audio[qid]

        # text to audio
        query_images.append([None])
        query_texts.append([process_input_text(TASK_INST_QRY, model_backbone, text=caption)])
        query_audio.append([None])

        info = {"query_id": qid, "cand_names": corpus_ids, "label_name": corpus_ids}
        dataset_infos.append(info)

    return {
        "query_text": query_texts,
        "query_image": query_images,
        "query_audio": query_audio,
        "dataset_infos": dataset_infos
    }


@add_metainfo_hook
def _prepare_corpus(batch_dict, text2audio=None, **kwargs):
    model_backbone = kwargs['model_backbone']
    corpus_ids_all = kwargs["corpus_ids_all"]
    audio2text = kwargs["audio2text"]
    filter_corpus = kwargs["filter_corpus"]

    cand_texts, cand_images, cand_audio = [], [], []
    dataset_infos = []

    for i in range(len(batch_dict["id"])):
        cid = batch_dict["corpus-id"][i]
        if (filter_corpus) and (cid not in corpus_ids_all):
            continue
        qid = audio2text.get(cid, -1)

        audio_bytes = batch_dict["audio_bytes"][i]

        cand_images.append([None])
        cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_audio_token=True)])
        cand_audio.append([{"bytes": audio_bytes}])
        dataset_infos.append({"query_id": [qid], "cand_names": [cid], "label_name": [cid]})

    return {
        "cand_text": cand_texts,
        "cand_image": cand_images,
        "cand_audio": cand_audio,
        "dataset_infos": dataset_infos
    }


@AutoEvalPairDataset.register("sounddescs")
def load_sounddescs_dataset(model_args, data_args, **kwargs):
    repo, _, _ = EVAL_DATASET_HF_PATH[kwargs['dataset_name']]
    data_root = kwargs.get("data_root", None)
    if data_root is None or data_root == "None":
        data_root = os.path.join(data_args.data_basedir, "audio-tasks")

    eval_path = download_sounddescs(repo, data_root)

    # def _read_parquet(filename):
    #     return load_dataset("parquet", data_files=os.path.join(eval_path, filename))["train"]

    query_path = os.path.join(eval_path, "query.parquet")
    query = load_dataset("parquet", data_files=query_path)["train"]

    qrels_path = os.path.join(eval_path, "qrels.parquet")
    qrels = load_dataset("parquet", data_files=qrels_path)["train"]

    corpus_path = os.path.join(eval_path, "corpus_16khz.parquet")
    corpus = load_dataset("parquet", data_files=corpus_path)["train"]

    print_master(f"original corpus size: {len(corpus)}")

    text2audio = defaultdict(list)
    for row in qrels:
        # 100% sure one to one mapping
        text2audio[row["query-id"]].append(row["corpus-id"])

    corpus_ids_all = [v[0] for v in text2audio.values()]
    audio2text = {v[0]: k for k, v in text2audio.items()}
    kwargs["corpus_ids_all"] = corpus_ids_all
    kwargs["audio2text"] = audio2text
    kwargs['model_backbone'] = model_args.model_backbone

    query = query.map(
        lambda x: _prepare_queries(x, text2audio=text2audio, **kwargs),
        batched=True,
        batch_size=4,
        num_proc=2,
        remove_columns=query.column_names,
        drop_last_batch=False,
        load_from_cache_file=False
    ).select_columns(["query_text", "query_image", "query_audio", "dataset_infos"])

    corpus = corpus.map(
        lambda x: _prepare_corpus(x, text2audio=text2audio, **kwargs),
        batched=True,
        batch_size=8,
        num_proc=2,
        remove_columns=corpus.column_names,
        drop_last_batch=False,
        load_from_cache_file=False
    ).select_columns(["cand_text", "cand_image", "cand_audio", "dataset_infos"])

    print_master(f"T2A: query: {len(query)}, corpus: {len(corpus)}")

    return query, corpus
