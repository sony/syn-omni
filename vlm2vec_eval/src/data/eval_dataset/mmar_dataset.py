import os

from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from src.utils.hf_utils import download_mmar


def process_query(query, prompt, audio_token=''):
    if prompt:
        query = f'{audio_token}{prompt} {query}'
    else:
        query = f'{query} {audio_token}'
    return query


TASK_PROMPT = "Given an audio clip and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question:"  # noqa
OPTIONS = ['A', 'B', 'C', 'D', "E", "F"]


def _convert_options(options):
    return [f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    local_dir = kwargs["local_dir"]

    query_texts, query_images, query_audio = [], [], []
    cand_texts, cand_images, cand_audio = [], [], []
    dataset_infos = []

    for audio_path, query, options, answer, qid, timestamp in zip(
        batch_dict['audio_path'], batch_dict['question'], batch_dict["choices"],
        batch_dict['answer'], batch_dict['id'], batch_dict["timestamp"]
    ):
        options = [o.strip() for o in options]
        answer = answer.strip()

        answer_idx = options.index(answer)
        options = _convert_options(options)  # 'answer' -> 'A. answer'

        audio_file = os.path.join(local_dir, audio_path)
        assert os.path.exists(audio_file), audio_file

        query = process_query(query + '\n' + '\n'.join(options), prompt=TASK_PROMPT)
        query_texts.append([query])
        query_images.append([None])
        query_audio.append([audio_file])

        cand_texts.append([o[o.find('. '):].strip('. ') for o in options])
        cand_images.append([None] * len(options))
        cand_audio.append([None] * len(options))

        dataset_info = {
            "question_id": qid,
            "audio_id": qid,
            "query": query,
            "cand_names": options,
            # "answer": answer,
            "label_name": options[answer_idx],
            "answer_idx": answer_idx,
        }
        dataset_infos.append(dataset_info)

    return {
        "query_text": query_texts,
        "query_image": query_images,
        "query_audio": query_audio,
        "cand_text": cand_texts,
        "cand_image": cand_images,
        "cand_audio": cand_audio,
        "dataset_infos": dataset_infos
    }


DATASET_PARSER_NAME = "mmar"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_mmar_dataset(model_args, data_args, *args, **kwargs):
    repo, _, _ = EVAL_DATASET_HF_PATH[kwargs['dataset_name']]
    data_root = kwargs.get("data_root", None)
    if data_root is None or data_root == "None":
        data_root = os.path.join(data_args.data_basedir, "audio-tasks")

    # filter invalid rows in advance
    def _keep_mask_fn(batch):
        keep_mask = []
        for answer, options in zip(batch["answer"], batch["choices"]):
            options = [o.strip() for o in options]
            answer = answer.strip()
            keep_mask.append(answer in options)
        return keep_mask

    dataset, local_dir = download_mmar(repo, data_root)
    len_orig = len(dataset)

    dataset = dataset.filter(_keep_mask_fn, batched=True, num_proc=4)
    print(
        f"size: {len(dataset)}, where filtered {len_orig - len(dataset)} from original {len_orig} samples"  # noqa
    )

    kwargs['dataset_name'] = DATASET_PARSER_NAME
    kwargs['model_backbone'] = model_args.model_backbone
    kwargs["local_dir"] = local_dir

    dataset = dataset.map(
        lambda x: data_prepare(x, **kwargs),
        batched=True,
        batch_size=16,
        num_proc=4,
        drop_last_batch=False,
        load_from_cache_file=False
    )

    return dataset, None
