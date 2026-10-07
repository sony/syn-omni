import json
import os
import shutil

from datasets import Audio, Dataset, Video

from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import (
    RESOLUTION_MAPPING,
    AutoEvalPairDataset,
    ImageVideoInstance,
    add_metainfo_hook,
)
from src.model.processor import process_input_text
from src.utils.av_extraction_utils import extract_frames_and_audio
from src.utils.dist_utils import barrier, is_main
from src.utils.hf_utils import load_hf_dataset_local
from src.utils.vision_utils.vision_utils import process_video_frames


def _derive_video_id(video):
    return os.path.splitext(os.path.basename(video["path"]))[0]


def _ensure_frames_and_audio_extracted(data_root, dataset):
    """
    Materialize each row's video bytes to a scratch mp4, extract frames (ffmpeg) +
    16kHz mono audio (PyAV) via av_extraction_utils (same logic used for AVHBench),
    then drop the scratch mp4s. Idempotent: skipped entirely once the manifest exists.
    """
    frame_root = os.path.join(data_root, "frames")
    audio_root = os.path.join(data_root, "audio")
    manifest_path = os.path.join(data_root, "converted_video_ids.json")

    if is_main() and not os.path.exists(manifest_path):
        tmp_video_dir = os.path.join(data_root, "_tmp_videos")
        os.makedirs(tmp_video_dir, exist_ok=True)

        video_ids = []
        for ex in dataset:
            vid = _derive_video_id(ex["video"])
            tmp_path = os.path.join(tmp_video_dir, f"{vid}.mp4")
            if not os.path.exists(tmp_path):
                with open(tmp_path, "wb") as f:
                    f.write(ex["video"]["bytes"])
            video_ids.append(vid)

        extract_frames_and_audio(
            video_dir=tmp_video_dir,
            frame_root=frame_root,
            audio_root=audio_root,
            video_ids=video_ids,
            max_frames_saved=32,
            manifest_path=manifest_path,
        )
        shutil.rmtree(tmp_video_dir, ignore_errors=True)
    barrier()

    assert os.path.exists(manifest_path), manifest_path
    return frame_root, audio_root, set(json.load(open(manifest_path)))


def _load_or_build_valor32k(repo, data_root):
    """
    Returns (annotations: list[{"video_id","description"}], frame_root, audio_root,
    valid_ids: set[str]).
    """
    frame_root = os.path.join(data_root, "frames")
    audio_root = os.path.join(data_root, "audio")
    manifest_path = os.path.join(data_root, "converted_video_ids.json")
    annotations_path = os.path.join(data_root, "annotations.json")

    if os.path.exists(manifest_path) and os.path.exists(annotations_path):
        valid_ids = set(json.load(open(manifest_path)))
        annotations = json.load(open(annotations_path))
        return annotations, frame_root, audio_root, valid_ids

    dataset = load_hf_dataset_local(
        repo, os.path.dirname(data_root), split_globs="data/test*.parquet"
    )
    dataset = dataset.cast_column("video", Video(decode=False))
    dataset = dataset.cast_column("audio", Audio(decode=False))

    _, _, valid_ids = _ensure_frames_and_audio_extracted(data_root, dataset)

    if is_main() and not os.path.exists(annotations_path):
        annotations = [
            {
                "video_id": _derive_video_id(ex["video"]),
                "description": ex["description"]
            } for ex in dataset
        ]
        with open(annotations_path, "w") as f:
            json.dump(annotations, f)
    barrier()

    annotations = json.load(open(annotations_path))
    return annotations, frame_root, audio_root, valid_ids


def _construct_modalities(video_id, caption, **kwargs):
    image_resolution = kwargs['image_resolution']
    num_frames = kwargs['num_frames']
    frame_root = kwargs["frame_root"]
    audio_root = kwargs["audio_root"]
    model_backbone = kwargs['model_backbone']
    eval_mode = kwargs["eval_mode"]
    assert eval_mode is not None

    frame_dir = os.path.join(frame_root, f"{video_id}")
    assert os.path.exists(frame_dir), frame_dir
    frames = ImageVideoInstance(
        bytes=[None] * num_frames,
        paths=process_video_frames(frame_dir, num_frames=num_frames),
        resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * num_frames,
    ).to_dict()

    audio_dir = os.path.join(audio_root, f"{video_id}.wav")
    assert os.path.exists(audio_dir), audio_dir

    # per-task different query and corpus combinations
    query_modalities = eval_mode.split("2")[0]
    corpus_modalities = eval_mode.split("2")[1]

    if "T" in query_modalities:  # text-only query
        assert query_modalities == "T"
        query_prompt = "Find a video that corresponds to the following caption:"
        query_image, query_aud = [None], [None]
        query_text = [process_input_text(query_prompt, model_backbone, text=caption)]

        cand_image = [frames if "V" in corpus_modalities else None]
        cand_aud = [audio_dir if "A" in corpus_modalities else None]

        # consider composed TVA embedding for pilot study
        if corpus_modalities == "T":
            # corpus: T only
            prompt = caption  # text-only embedding (identity)
        elif "T" in corpus_modalities:
            # corpus: TA, TV, TVA
            prompt = f"Understand the content of the provided video with the following caption: {caption}"  # noqa
        else:
            # corpus: V, A, or VA
            prompt = "Understand the content of the provided video."

        cand_text = [
            process_input_text(
                prompt,
                model_backbone,
                add_video_token=True if "V" in corpus_modalities else False,
                add_audio_token=True if "A" in corpus_modalities else False
            )
        ]
        dataset_info = {"cand_names": [video_id], "label_name": video_id}

    elif "T" in corpus_modalities:  # text-only corpus
        query_image = [frames if "V" in query_modalities else None]
        query_aud = [audio_dir if "A" in query_modalities else None]
        query_text = [
            process_input_text(
                "Find a caption that corresponds to the provided video.",
                model_backbone,
                add_video_token=True if "V" in corpus_modalities else False,
                add_audio_token=True if "A" in corpus_modalities else False
            )
        ]

        cand_image, cand_aud, cand_text = [None], [None], [caption]
        dataset_info = {"cand_names": [caption], "label_name": caption}

    elif eval_mode == "A2V":
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
        raise ValueError(f"Unknown eval_mode value: {eval_mode}")

    return query_image, query_aud, query_text, cand_image, cand_aud, cand_text, dataset_info


@add_metainfo_hook
def data_prepare(batch_dict, **kwargs):
    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    query_audio, cand_audio = [], []

    for video_id, caption in zip(batch_dict['video_id'], batch_dict['description']):
        (query_image, query_aud, query_text, cand_image, cand_aud, cand_text,
         dataset_info) = _construct_modalities(video_id, caption, **kwargs)

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


DATASET_PARSER_NAME = "valor32k"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_valor32k_dataset(model_args, data_args, **kwargs):
    num_sample_per_subset = kwargs.get("num_sample_per_subset", None)

    repo, _, _ = EVAL_DATASET_HF_PATH[kwargs['dataset_name']]
    data_root = os.path.join(data_args.data_basedir, "valor32k-retrieval", "mteb__VALOR-32K")

    annotations, frame_root, audio_root, valid_ids = _load_or_build_valor32k(repo, data_root)

    n_before = len(annotations)
    annotations = [a for a in annotations if a["video_id"] in valid_ids]
    n_after = len(annotations)
    if n_after < n_before:
        print(
            f"[valor32k] kept {n_after}/{n_before} rows "
            f"(dropped {n_before - n_after} with failed/missing frame+audio extraction)"
        )

    dataset = Dataset.from_list(annotations)

    if num_sample_per_subset is not None and num_sample_per_subset < dataset.num_rows:
        num_rows = int(num_sample_per_subset)
        dataset = dataset.select(range(num_rows))

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs["frame_root"] = frame_root
    kwargs["audio_root"] = audio_root

    dataset = dataset.map(
        lambda x: data_prepare(x, **kwargs),
        batched=True,
        batch_size=64,
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
    corpus = None  # No additional corpus
    return dataset, corpus
