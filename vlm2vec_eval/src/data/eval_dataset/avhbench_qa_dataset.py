import json
import os

from datasets import Dataset

from src.data.eval_dataset.base_eval_dataset import (
    AutoEvalPairDataset,
    ImageVideoInstance,
    add_metainfo_hook,
)
from src.utils.hf_utils import download_avhbench, get_avhbench_valid_video_ids
from src.utils.vision_utils.vision_utils import process_video_frames


def process_query(query, prompt, video_token="", audio_token=''):
    if prompt:
        query = f'{video_token}{audio_token}{prompt} {query}'
    else:
        query = f'{query} {video_token}{audio_token}'
    return query


TASK_PROMPT = "Given a video with audio and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question:"  # noqa
OPTIONS = ['yes', 'no']


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
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

        answer = answer.lower()  # Yes -> yes

        query = process_query(query + ' (A) yes; (B) no.', prompt=TASK_PROMPT)
        query_texts.append([query])
        query_images.append([frames])
        query_audio.append([audio_dir])

        cand_texts.append(OPTIONS)
        cand_images.append([None] * len(OPTIONS))
        cand_audio.append([None] * len(OPTIONS))

        dataset_info = {
            "video_id": video_id,
            "query": query,
            "cand_names": OPTIONS,
            "answer": answer,
            "label_name": answer,
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


DATASET_PARSER_NAME = "avhbench_qa"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_avhbench_qa_dataset(model_args, data_args, *args, **kwargs):
    # AVHBench is only distributed via Google Drive (no HF mirror yet). download_avhbench()
    # also extracts frames/audio for every referenced video_id (see av_extraction_utils.py).
    data_root = os.path.join(data_args.data_basedir, "audiovisual-tasks")
    local_dir = download_avhbench(data_root)
    valid_ids = get_avhbench_valid_video_ids(local_dir)

    anns = json.load(open(os.path.join(local_dir, "QA.json"), "r"))
    anns = [ann for ann in anns if ann["label"] in ["Yes", "No"]]

    n_before = len(anns)
    anns = [ann for ann in anns if ann["video_id"] in valid_ids]
    n_after = len(anns)
    if n_after < n_before:
        print(
            f"[avhbench_qa] kept {n_after}/{n_before} rows "
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
