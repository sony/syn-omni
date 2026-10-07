import os

from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import (
    RESOLUTION_MAPPING,
    AutoEvalPairDataset,
    ImageVideoInstance,
    add_metainfo_hook,
)
from src.model.processor import process_input_text
from src.utils.basic_utils import print_master
from src.utils.dataset_utils import load_hf_dataset, sample_dataset
from src.utils.vision_utils.vision_utils import process_video_frames

T2V_TASK_INST_QRY = "Find a video that contains the following visual content:"
T2V_TASK_INST_TGT = "Understand the content of the provided video."
V2T_TASK_INST_QRY = "Find a caption that corresponds to the provided video."


@add_metainfo_hook
def data_prepare(batch_dict, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    num_frames = kwargs['num_frames']
    frame_root = kwargs['frame_root']
    model_backbone = kwargs['model_backbone']
    eval_mode = kwargs.get("eval_mode", "T2V")  # by defalt, text to video retrieval

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for video_name, video_path, caption in (
        zip(batch_dict['video_id'], batch_dict['video'], batch_dict['caption'])
    ):
        # load frame paths
        frame_dir = os.path.join(frame_root, video_name)
        assert os.path.exists(frame_dir)
        video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
        frames = ImageVideoInstance(
            bytes=[None] * num_frames,
            paths=video_frame_paths,
            resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * num_frames,
        ).to_dict()

        if eval_mode == "T2V":
            query_texts.append(
                [process_input_text(T2V_TASK_INST_QRY, model_backbone, text=caption)]
            )
            query_images.append([None])

            cand_texts.append(
                [process_input_text(T2V_TASK_INST_TGT, model_backbone, add_video_token=True)]
            )
            cand_images.append([frames])
            dataset_infos.append({"cand_names": [video_name], "label_name": video_name})
        else:
            assert eval_mode == "V2T"
            query_texts.append(
                [process_input_text(V2T_TASK_INST_QRY, model_backbone, add_video_token=True)]
            )
            query_images.append([frames])

            cand_texts.append([caption])
            cand_images.append([None])
            dataset_infos.append({"cand_names": [caption], "label_name": caption})

    return {
        "query_text": query_texts,
        "query_image": query_images,
        "cand_text": cand_texts,
        "cand_image": cand_images,
        "dataset_infos": dataset_infos
    }


DATASET_PARSER_NAME = "msrvtt"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_msrvtt_dataset(model_args, data_args, **kwargs):
    dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[kwargs['dataset_name']])
    dataset = sample_dataset(dataset, **kwargs)

    # caption overlap check
    captions = dataset["caption"][:]
    print_master(f"data len: {len(captions)}, unique caption len: {len(set(captions))}...\n")

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution

    dataset = dataset.map(
        lambda x: data_prepare(x, **kwargs),
        batched=True,
        batch_size=256,
        num_proc=1,  # 4
        drop_last_batch=False,
        load_from_cache_file=False
    )
    dataset = dataset.select_columns(
        ["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"]
    )
    corpus = None  # No additional corpus

    return dataset, corpus
