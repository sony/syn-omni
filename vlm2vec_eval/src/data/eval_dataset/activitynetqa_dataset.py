import os

from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from src.model.processor import VLM_VIDEO_TOKENS
from src.utils.dataset_utils import load_hf_dataset, sample_dataset
from src.utils.vision_utils.vision_utils import load_frames, process_video_frames


# spacing
def process_query(query, prompt, video_token=''):
    if prompt:
        query = f'{video_token}{prompt} {query}'
    else:
        query = f'{query} {video_token}'
    return query


TASK_PROMPT = "Given a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question:"  # noqa
OPTIONS = ['yes', 'no']


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    model_backbone = kwargs['model_backbone']
    frame_root = kwargs['frame_root']
    num_frames = kwargs['num_frames']
    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for video_name, query, answer, question_id in zip(
        batch_dict['video_name'], batch_dict['question'], batch_dict['answer'],
        batch_dict['question_id']
    ):
        query = process_query(
            query + '? (A) yes; (B) no.',
            prompt=TASK_PROMPT,
            video_token=VLM_VIDEO_TOKENS[model_backbone]
        )
        query_texts.append([query])
        frame_dir = f'{frame_root}/v_{video_name}'
        assert os.path.exists(frame_dir)
        frames = load_frames(frame_dir)
        if not frames:
            raise FileNotFoundError

        qry_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
        assert len(qry_frame_paths)
        qry_frames = {
            "bytes": [None] * len(qry_frame_paths),
            "paths": qry_frame_paths,
            "resolutions": [None] * len(qry_frame_paths)
        }
        query_images.append([qry_frames])
        cand_texts.append(OPTIONS)
        cand_images.append([None] * len(OPTIONS))
        dataset_info = {
            "question_id": question_id,
            "video_id": video_name,
            "query": query,
            "cand_names": OPTIONS,
            "answer": answer,
            "label_name": answer,
            "answer_idx": OPTIONS.index(answer),
            "qry_frame_paths": qry_frame_paths,
        }
        dataset_infos.append(dataset_info)
    if len(query_texts) == 0:
        print('something went wrong')
    # print_rank(f"dataset.map(): global_dataset_name={kwargs.get('global_dataset_name', DATASET_PARSER_NAME)}, batch_size={batch_size}, processed_batch_size={len(query_texts)}")
    return {
        "query_text": query_texts,
        "query_image": query_images,
        "cand_text": cand_texts,
        "cand_image": cand_images,
        "dataset_infos": dataset_infos
    }


DATASET_PARSER_NAME = "activitynetqa"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_activitynetqa_dataset(model_args, data_args, *args, **kwargs):
    dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[kwargs['dataset_name']])
    dataset = sample_dataset(dataset, **kwargs)

    kwargs['dataset_name'] = DATASET_PARSER_NAME
    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['video_export_dir'] = kwargs.get("video_export_dir", None)
    kwargs['global_dataset_name'] = DATASET_PARSER_NAME
    dataset = dataset.map(
        lambda x: data_prepare(x, **kwargs),
        batched=True,
        batch_size=256,
        num_proc=4,
        drop_last_batch=False,
        load_from_cache_file=False
    )
    return dataset, None
