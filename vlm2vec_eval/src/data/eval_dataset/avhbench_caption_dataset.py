import json
import os

from datasets import Dataset

from src.data.eval_dataset.base_eval_dataset import (
    AutoEvalPairDataset,
    ImageVideoInstance,
    add_metainfo_hook,
)
from src.model.processor import process_input_text
from src.utils.hf_utils import download_avhbench, get_avhbench_valid_video_ids
from src.utils.vision_utils.vision_utils import process_video_frames


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    model_backbone = kwargs['model_backbone']
    eval_mode = kwargs["eval_mode"]
    assert eval_mode is not None and eval_mode in ["VA2T", "T2VA"]

    local_dir = kwargs["local_dir"]
    num_frames = kwargs['num_frames']
    assert num_frames == 8

    frame_root = os.path.join(local_dir, "frames")
    audio_root = os.path.join(local_dir, "audio")

    query_texts, query_images, query_audio = [], [], []
    cand_texts, cand_images, cand_audio = [], [], []
    dataset_infos = []

    for video_id, query, answer in zip(
        batch_dict["video_id"], batch_dict['text'], batch_dict['label']
    ):
        # pre-extracted frames (ffmpeg) + 16kHz mono audio (PyAV), not raw mp4 decode
        frame_dir = os.path.join(frame_root, f"{video_id}")
        assert os.path.exists(frame_dir), frame_dir
        frames = ImageVideoInstance(
            bytes=[None] * num_frames,
            paths=process_video_frames(frame_dir, num_frames=num_frames),
            resolutions=[None] * num_frames,
        ).to_dict()
        audio_dir = os.path.join(audio_root, f"{video_id}.wav")
        assert os.path.exists(audio_dir), audio_dir

        if eval_mode == "VA2T":
            query_text = [
                process_input_text(
                    query, model_backbone, add_video_token=True, add_audio_token=True
                )
            ]
            query_image = [frames]
            query_aud = [audio_dir]

            cand_image, cand_aud, cand_text = [None], [None], [answer]
            dataset_info = {"cand_names": [answer], "label_name": answer}
        else:
            assert eval_mode == "T2VA"
            query_prompt = "Find a video that corresponds to the following caption:"
            query_image, query_aud = [None], [None]
            query_text = [process_input_text(query_prompt, model_backbone, text=answer)]

            prompt = "Understand the content of the provided video."
            cand_image = [frames]
            cand_aud = [audio_dir]
            cand_text = [
                process_input_text(
                    prompt, model_backbone, add_video_token=True, add_audio_token=True
                )
            ]
            dataset_info = {"cand_names": [video_id], "label_name": video_id}

        query_texts.append(query_text)
        query_images.append(query_image)
        query_audio.append(query_aud)

        cand_texts.append(cand_text)
        cand_images.append(cand_image)
        cand_audio.append(cand_aud)

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


DATASET_PARSER_NAME = "avhbench_caption"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_avhbench_caption_dataset(model_args, data_args, *args, **kwargs):
    # AVHBench is only distributed via Google Drive (no HF mirror yet). download_avhbench()
    # also extracts frames/audio for every referenced video_id (see av_extraction_utils.py).
    data_root = os.path.join(data_args.data_basedir, "audiovisual-tasks")
    local_dir = download_avhbench(data_root)
    valid_ids = get_avhbench_valid_video_ids(local_dir)

    anns = json.load(open(os.path.join(local_dir, "QA.json"), "r"))
    anns = [ann for ann in anns if ann["label"] not in ["Yes", "No"]]

    n_before = len(anns)
    anns = [ann for ann in anns if ann["video_id"] in valid_ids]
    n_after = len(anns)
    if n_after < n_before:
        print(
            f"[avhbench_caption] kept {n_after}/{n_before} rows "
            f"(dropped {n_before - n_after} with failed/missing frame+audio extraction)"
        )

    dataset = Dataset.from_list(anns)

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
    dataset = dataset.select_columns(
        [
            "query_text", "query_image", "query_audio", "cand_text", "cand_image", "cand_audio",
            "dataset_infos"
        ]
    )
    return dataset, None
