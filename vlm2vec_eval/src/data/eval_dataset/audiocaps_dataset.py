import os

from collections import defaultdict

from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from src.model.processor import process_input_text
from src.utils.basic_utils import print_master
from src.utils.hf_utils import load_hf_dataset_local

TASK_INST_QRY = "Find an audio that contains the following content:"
TASK_INST_TGT = "Understand the content of the provided audio."
A2T_TASK_INST_QRY = "Find a caption that corresponds to the provided audio."


@add_metainfo_hook
def data_prepare(batch_dict, audio2text=None, text2audio=None, **kwargs):
    model_backbone = kwargs['model_backbone']
    eval_mode = kwargs.get("eval_mode", "T2A")

    query_texts, query_images, query_audio = [], [], []
    cand_texts, cand_images, cand_audio = [], [], []
    dataset_infos = []

    for i in range(len(batch_dict["youtube_id"])):
        yid = batch_dict["youtube_id"][i]
        audio = batch_dict["audio"][i]
        caption = batch_dict["caption"][i]

        if eval_mode == "T2A":
            # text to audio
            query_images.append([None])
            query_texts.append([process_input_text(TASK_INST_QRY, model_backbone, text=caption)])
            query_audio.append([None])

            cand_images.append([None])
            cand_texts.append(
                [process_input_text(TASK_INST_TGT, model_backbone, add_audio_token=True)]
            )
            cand_audio.append([audio])
            # there exist multiple audio that shares the same caption
            dataset_infos.append({"cand_names": [yid], "label_name": text2audio[caption]})
        else:
            query_images.append([None])
            query_texts.append(
                [process_input_text(A2T_TASK_INST_QRY, model_backbone, add_audio_token=True)]
            )
            query_audio.append([audio])

            candidate_captions = audio2text[yid]  # [Cap1, Cap2, Cap3, Cap4, Cap5]
            cand_images.append([None] * len(candidate_captions))
            cand_texts.append(candidate_captions)
            cand_audio.append([None] * len(candidate_captions))
            dataset_infos.append(
                {
                    "cand_names": candidate_captions,
                    "label_name": candidate_captions
                }
            )

    return {
        "query_text": query_texts,
        "query_image": query_images,
        "query_audio": query_audio,
        "cand_text": cand_texts,
        "cand_image": cand_images,
        "cand_audio": cand_audio,
        "dataset_infos": dataset_infos
    }


@AutoEvalPairDataset.register("audiocaps")
def load_audiocaps_dataset(model_args, data_args, **kwargs):
    repo, _, _ = EVAL_DATASET_HF_PATH[kwargs['dataset_name']]
    data_root = kwargs.get("data_root", None)
    if data_root is None or data_root == "None":
        data_root = os.path.join(data_args.data_basedir, "audio-tasks")

    dataset = load_hf_dataset_local(
        repo, data_root, split_globs="data/test*.parquet", split="train"
    )
    eval_mode = kwargs.get("eval_mode", "T2A")

    audio2text = defaultdict(list)
    text2audio = defaultdict(list)
    unique_indices = []
    seen_yids = set()

    for idx, row in enumerate(dataset):
        yid = row["youtube_id"]
        cap = row["caption"]
        audio2text[yid].append(cap)
        text2audio[cap].append(yid)

        if yid not in seen_yids:
            unique_indices.append(idx)
            seen_yids.add(yid)

    if eval_mode == "A2T":
        print_master(
            f"Filtering dataset for A2T: {len(dataset)} -> {len(unique_indices)} unique audios"
        )
        dataset = dataset.select(unique_indices)
    else:
        print_master(
            f"Using full dataset for T2A: {len(dataset)} queries, {len(text2audio)} unique captions"
        )

    kwargs['model_backbone'] = model_args.model_backbone

    dataset = dataset.map(
        lambda x: data_prepare(x, audio2text=audio2text, text2audio=text2audio, **kwargs),
        batched=True,
        batch_size=8,
        num_proc=1,
        remove_columns=dataset.column_names
    )

    dataset = dataset.select_columns(
        [
            "query_text", "query_image", "query_audio", "cand_text", "cand_image", "cand_audio",
            "dataset_infos"
        ]
    )

    return dataset, None
