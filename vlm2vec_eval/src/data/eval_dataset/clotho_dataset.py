import os

from collections import defaultdict

from datasets import Dataset

from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from src.model.processor import process_input_text
from src.utils.basic_utils import print_master
from src.utils.hf_utils import load_hf_csv_dataset_local

TASK_INST_QRY = "Find an audio that contains the following content:"
TASK_INST_TGT = "Understand the content of the provided audio."
A2T_TASK_INST_QRY = "Find a caption that corresponds to the provided audio."


@add_metainfo_hook
def data_prepare(batch_dict, audio2text=None, text2audio=None, **kwargs):
    model_backbone = kwargs['model_backbone']
    eval_mode = kwargs.get("eval_mode", "T2A")
    audio_prefix = kwargs.get("audio_prefix", None)
    assert audio_prefix is not None

    query_texts, query_images, query_audio = [], [], []
    cand_texts, cand_images, cand_audio = [], [], []
    dataset_infos = []

    for i in range(len(batch_dict["file_name"])):
        file_name = batch_dict["file_name"][i]
        audio_file = os.path.join(audio_prefix, file_name)
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
            cand_audio.append([audio_file])
            dataset_infos.append({"cand_names": [file_name], "label_name": text2audio[caption]})
        else:
            query_images.append([None])
            query_texts.append(
                [process_input_text(A2T_TASK_INST_QRY, model_backbone, add_audio_token=True)]
            )
            query_audio.append([audio_file])

            candidate_captions = audio2text[file_name]  # [Cap1, Cap2, Cap3, Cap4, Cap5]
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


@AutoEvalPairDataset.register("clotho")
def load_clotho_dataset(model_args, data_args, **kwargs):
    temp = kwargs.get("temp", False)
    num_sample_per_subset = kwargs.get("num_sample_per_subset", None)

    repo, _, _ = EVAL_DATASET_HF_PATH[kwargs['dataset_name']]
    data_root = kwargs.get("data_root", None)
    if data_root is None or data_root == "None":
        data_root = os.path.join(data_args.data_basedir, "audio-tasks")
    eval_mode = kwargs.get("eval_mode", "T2A")

    dataset, local_dir = load_hf_csv_dataset_local(
        repo, data_root, data_files="clotho_captions_evaluation.csv"
    )

    # unroll datasets to have columns from (file_name, caption_1~5) -> (fil_name, caption)
    df = dataset.to_pandas()
    df_long = df.melt(
        id_vars="file_name",
        value_vars=[f"caption_{i + 1}" for i in range(5)],
        value_name="caption"
    )
    df_long = df_long[["file_name", "caption"]]
    if temp:
        print(f"original size: {len(df_long)}")
        df_long = df_long.drop_duplicates(subset=["file_name"])
        print(f"one to one size: {len(df_long)}")
        print()
    dataset = Dataset.from_pandas(df_long, preserve_index=False)
    # for analysis
    if num_sample_per_subset is not None and num_sample_per_subset < dataset.num_rows:
        num_rows = int(num_sample_per_subset)
        dataset = dataset.select(range(num_rows))

    audio2text = defaultdict(list)
    text2audio = defaultdict(list)
    unique_indices = []
    seen_yids = set()

    for idx, row in enumerate(dataset):
        audio_id = row["file_name"]
        cap = row["caption"]
        audio2text[audio_id].append(cap)
        text2audio[cap].append(audio_id)

        if audio_id not in seen_yids:
            unique_indices.append(idx)
            seen_yids.add(audio_id)

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
    kwargs["audio_prefix"] = os.path.join(local_dir, "evaluation")

    dataset = dataset.map(
        lambda x: data_prepare(x, audio2text=audio2text, text2audio=text2audio, **kwargs),
        batched=True,
        batch_size=8,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=dataset.column_names
    )

    dataset = dataset.select_columns(
        [
            "query_text", "query_image", "query_audio", "cand_text", "cand_image", "cand_audio",
            "dataset_infos"
        ]
    )

    return dataset, None
