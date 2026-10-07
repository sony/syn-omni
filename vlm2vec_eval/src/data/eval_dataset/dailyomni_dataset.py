import os

from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from src.utils.hf_utils import load_hf_dataset_local


def process_query(query, prompt, video_token="", audio_token=''):
    if prompt:
        query = f'{video_token}{audio_token}{prompt} {query}'
    else:
        query = f'{query} {audio_token}'
    return query


TASK_PROMPT = "Given a video with audio and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question:"  # noqa
OPTIONS = ['A', 'B', 'C', 'D']


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    query_texts, query_images, query_audio = [], [], []
    cand_texts, cand_images, cand_audio = [], [], []
    dataset_infos = []

    for video_id, video, audio, query, options, answer in zip(
        batch_dict["video_id"], batch_dict["video"], batch_dict["audio"], batch_dict["question"],
        batch_dict["candidates"], batch_dict["answer"]
    ):
        answer = answer[0]  # only alphabet
        query = process_query(query + '\n' + '\n'.join(options), prompt=TASK_PROMPT, audio_token="")
        query_texts.append([query])
        query_images.append([video])
        query_audio.append([audio])

        cand_texts.append([o[o.find('. '):].strip('. ') for o in options])
        cand_images.append([None] * len(OPTIONS))
        cand_audio.append([None] * len(OPTIONS))

        dataset_info = {
            "question_id": video_id,
            "audio_id": video_id,
            "query": query,
            "cand_names": options,
            "answer": answer,
            "label_name": options[OPTIONS.index(answer)],
            "answer_idx": OPTIONS.index(answer),
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


DATASET_PARSER_NAME = "dailyomni"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_dailyomni_dataset(model_args, data_args, *args, **kwargs):
    repo, _, _ = EVAL_DATASET_HF_PATH[kwargs['dataset_name']]
    data_root = kwargs.get("data_root", None)
    if data_root is None or data_root == "None":
        data_root = os.path.join(data_args.data_basedir, "audiovisual-tasks")

    dataset = load_hf_dataset_local(repo, data_root, split_globs="data/test*.parquet")
    kwargs['dataset_name'] = DATASET_PARSER_NAME
    kwargs['model_backbone'] = model_args.model_backbone

    dataset = dataset.map(
        lambda x: data_prepare(x, **kwargs),
        batched=True,
        batch_size=16,
        num_proc=4,
        drop_last_batch=False,
        load_from_cache_file=False
    )
    dataset = dataset.select_columns(
        [
            "query_text", "query_image", "query_audio", "cand_text", "cand_image", "cand_audio",
            "dataset_infos"
        ]
    )
    return dataset, None
