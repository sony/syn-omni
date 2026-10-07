import os

from collections import defaultdict

from tqdm import tqdm

from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from src.model.processor import process_input_text
from src.utils.basic_utils import print_master
from src.utils.hf_utils import load_hf_dataset_local

TASK_INST_QRY = "Find a audio that contains the following content:"
TASK_INST_TGT = "Understand the content of the provided audio."
A2T_TASK_INST_QRY = "Find a caption that corresponds to the provided audio."


@add_metainfo_hook
def data_prepare(batch_dict, cap_to_idx=None, **kwargs):
    assert cap_to_idx is not None
    model_backbone = kwargs['model_backbone']
    eval_mode = kwargs.get("eval_mode", "T2A")

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    query_audio, cand_audio = [], []

    for audio_id, audio, caption in (
        zip(batch_dict["idx"], batch_dict['audio'], batch_dict['caption'])
    ):
        query_images.append([None])
        cand_images.append([None])

        if eval_mode == "T2A":
            query_texts.append([process_input_text(TASK_INST_QRY, model_backbone, text=caption)])
            query_audio.append([None])

            cand_texts.append(
                [process_input_text(TASK_INST_TGT, model_backbone, add_audio_token=True)]
            )
            cand_audio.append([audio])  # pass the audio as is

            labels = cap_to_idx[caption]
            dataset_infos.append({"cand_names": [audio_id], "label_name": labels})
        else:
            assert eval_mode == "A2T"
            query_texts.append(
                [process_input_text(A2T_TASK_INST_QRY, model_backbone, add_audio_token=True)]
            )
            query_audio.append([audio])

            cand_texts.append([caption])
            cand_audio.append([None])

            dataset_infos.append({"cand_names": [caption], "label_name": caption})

    return {
        "query_text": query_texts,
        "query_image": query_images,
        "query_audio": query_audio,
        "cand_text": cand_texts,
        "cand_image": cand_images,
        "cand_audio": cand_audio,
        "dataset_infos": dataset_infos
    }


DATASET_PARSER_NAME = "musiccaps"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_musiccaps_dataset(model_args, data_args, **kwargs):
    repo, _, _ = EVAL_DATASET_HF_PATH[kwargs['dataset_name']]
    data_root = kwargs.get("data_root", None)
    if data_root is None or data_root == "None":
        data_root = os.path.join(data_args.data_basedir, "audio-tasks")

    dataset = load_hf_dataset_local(repo, data_root, split_globs="*.parquet", split="train")
    # dataset = sample_dataset(dataset, **kwargs)

    kwargs['model_backbone'] = model_args.model_backbone

    # kwargs['image_resolution'] = data_args.image_resolution

    # get eval samples
    def _keep_mask(batch):
        is_audioset_evals = batch["is_audioset_eval"]
        keep = [is_audioset_eval == "True" for is_audioset_eval in is_audioset_evals]
        return keep

    dataset = dataset.filter(_keep_mask, batched=True, num_proc=4)
    dataset = dataset.add_column("idx", list(range(len(dataset))))

    # handle overlapping captions for query
    cap_to_idx = defaultdict(list)
    for i, caption in tqdm(enumerate(dataset["caption"][:])):
        cap_to_idx[caption].append(i)

    print_master(
        f"dataset size: {len(dataset)}, overlapping captions: {len(dataset) - len(cap_to_idx)}"
    )
    dataset = dataset.map(
        lambda x: data_prepare(x, cap_to_idx=cap_to_idx, **kwargs),
        batched=True,
        batch_size=8,
        num_proc=1,  # 4,
        drop_last_batch=False,
        load_from_cache_file=False
    )
    dataset = dataset.select_columns(
        [
            "query_text", "query_image", "query_audio", "cand_text", "cand_image", "cand_audio",
            "dataset_infos"
        ]
    )
    corpus = None  # No additional corpus

    return dataset, corpus
