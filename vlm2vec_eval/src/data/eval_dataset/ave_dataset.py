import os

import pandas as pd

from datasets import Dataset

from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import (
    AutoEvalPairDataset,
    ImageVideoInstance,
    add_metainfo_hook,
)
from src.model.processor import process_input_text
from src.utils.hf_utils import download_ave
from src.utils.vision_utils.vision_utils import process_video_frames


@add_metainfo_hook
def data_prepare(batch_dict, **kwargs):
    num_frames = kwargs['num_frames']
    eval_path = kwargs["eval_path"]
    frame_root = os.path.join(eval_path, "frames")
    audio_root = os.path.join(eval_path, "audios")
    model_backbone = kwargs['model_backbone']
    eval_mode = kwargs["eval_mode"]
    assert eval_mode is not None

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    query_audio, cand_audio = [], []

    for video_id, start, end in zip(batch_dict['video_id'], batch_dict['start'], batch_dict['end']):

        # prepare video
        frame_dir = os.path.join(frame_root, f"{video_id}")
        assert os.path.exists(frame_dir), frame_dir
        frames = ImageVideoInstance(
            bytes=[None] * num_frames,
            paths=process_video_frames(frame_dir, num_frames=num_frames),
            resolutions=[None] * num_frames,
        ).to_dict()

        # prepare audio
        audio_name = f"{video_id}_{video_id}_{start}_{end}.wav"
        audio_dir = os.path.join(audio_root, audio_name)
        assert os.path.exists(audio_dir), audio_dir

        if eval_mode == "A2V":
            query_prompt = "Find a video that corresponds to the given audio."
            query_text = [process_input_text(query_prompt, model_backbone, add_audio_token=True)]
            query_image = [None]
            query_aud = [audio_dir]

            cand_prompt = "Understand the content of the provided video."
            cand_text = [process_input_text(cand_prompt, model_backbone, add_video_token=True)]
            cand_image = [frames]
            cand_aud = [None]

            dataset_info = {"cand_names": [video_id], "label_name": video_id}

        elif eval_mode == "V2A":
            query_prompt = "Find a audio that corresponds to the given video."
            query_text = [process_input_text(query_prompt, model_backbone, add_video_token=True)]
            query_image = [frames]
            query_aud = [None]

            cand_prompt = "Understand the content of the provided audio."
            cand_text = [process_input_text(cand_prompt, model_backbone, add_audio_token=True)]
            cand_image = [None]
            cand_aud = [audio_dir]

            dataset_info = {"cand_names": [video_id], "label_name": video_id}
        else:
            raise ValueError(f"invalid eval_mode: {eval_mode}")

        query_images.append(query_image)
        query_audio.append(query_aud)
        query_texts.append(query_text)

        cand_images.append(cand_image)
        cand_audio.append(cand_aud)
        cand_texts.append(cand_text)

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


DATASET_PARSER_NAME = "ave"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_ave_dataset(model_args, data_args, **kwargs):
    repo, _, _ = EVAL_DATASET_HF_PATH[kwargs['dataset_name']]
    data_root = kwargs.get("data_root", None)
    if data_root is None or data_root == "None":
        data_root = os.path.join(data_args.data_basedir, "audiovisual-tasks")
    eval_path = download_ave(repo, data_root)

    ann_file = os.path.join(eval_path, "testSet.txt")
    df = pd.read_csv(ann_file, sep="&", names=["category", "video_id", "quality", "start", "end"])
    dataset = Dataset.from_pandas(df, preserve_index=False)

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs["data_root"] = data_root
    kwargs["eval_path"] = eval_path

    dataset = dataset.map(
        lambda x: data_prepare(x, **kwargs),
        batched=True,
        batch_size=8,
        num_proc=1,  # 4
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
