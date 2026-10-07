import os

from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from src.model.processor import process_input_text
from src.utils.basic_utils import print_master
from src.utils.hf_utils import load_gtzan_dataset_local

TASK_INST_QRY = "Identify the musical genre category of the audio."


@add_metainfo_hook
def data_prepare(batch_dict, **kwargs):
    model_backbone = kwargs['model_backbone']
    local_dir = kwargs["local_dir"]
    query_texts, query_images, query_audio = [], [], []
    cand_texts, cand_images, cand_audio = [], [], []
    dataset_infos = []

    for i in range(len(batch_dict["audio_path"])):
        audio_path = batch_dict["audio_path"][i].replace("data/GTZAN/genres/", "")
        caption = batch_dict["label"][i]
        audio = os.path.join(local_dir, "genres_original", audio_path)

        query_images.append([None])
        query_texts.append(
            [process_input_text(TASK_INST_QRY, model_backbone, add_audio_token=True)]
        )
        query_audio.append([audio])

        candidate_captions = [caption]
        cand_images.append([None] * len(candidate_captions))
        cand_texts.append(candidate_captions)
        cand_audio.append([None] * len(candidate_captions))
        dataset_infos.append({"cand_names": candidate_captions, "label_name": candidate_captions})

    return {
        "query_text": query_texts,
        "query_image": query_images,
        "query_audio": query_audio,
        "cand_text": cand_texts,
        "cand_image": cand_images,
        "cand_audio": cand_audio,
        "dataset_infos": dataset_infos
    }


@AutoEvalPairDataset.register("gtzan")
def load_gtzan_dataset(model_args, data_args, **kwargs):
    repo, _, _ = EVAL_DATASET_HF_PATH[kwargs['dataset_name']]
    data_root = kwargs.get("data_root", None)
    if data_root is None or data_root == "None":
        data_root = os.path.join(data_args.data_basedir, "audio-tasks")

    dataset, local_dir = load_gtzan_dataset_local(repo, data_root)
    unique_captions = set(list(dataset["label"]))
    print_master(f"dataset size: {len(dataset)} with {len(unique_captions)} classes")

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs["local_dir"] = local_dir

    dataset = dataset.map(
        lambda x: data_prepare(x, **kwargs),
        batched=True,
        batch_size=8,
        num_proc=1,
        remove_columns=dataset.column_names,
        load_from_cache_file=False
    )

    dataset = dataset.select_columns(
        [
            "query_text", "query_image", "query_audio", "cand_text", "cand_image", "cand_audio",
            "dataset_infos"
        ]
    )

    return dataset, None
