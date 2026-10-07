import json
import os

import pandas as pd

# all datasets and tasks
TASKS = {
    "image": {
        "I-CLS": [
            "ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365", "ImageNet-A",
            "ImageNet-R", "ObjectNet", "Country211"
        ],
        "I-VQA": [
            "OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W", "ScienceQA",
            "VizWiz", "GQA", "TextVQA"
        ],
        "I-RET": [
            "VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i", "MSCOCO_i2t",
            "NIGHTS", "WebQA", "FashionIQ", "Wiki-SS-NQ", "OVEN", "EDIS"
        ],
        "VG": ["MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"]
    },
    "video": {
        "V-CLS": ["K700", "SmthSmthV2", "HMDB51", "UCF101", "Breakfast"],
        "T2V RET": ["DiDeMo", "MSR-VTT", "MSVD", "VATEX", "YouCook2"],
        "V2T RET": ["DiDeMo_V2T", "MSR-VTT_V2T", "MSVD_V2T", "VATEX_V2T", "YouCook2_V2T"],
        "M-RET": ["QVHighlight", "Charades-STA", "MomentSeeker"],
        "V-VQA": ["MVBench", "Video-MME", "NExTQA", "EgoSchema", "ActivityNetQA"]
    },
    "audio": {
        "A-CLS": ["ESC50", "GTZAN", "NSynth"],
        "T2A RET": ["AudioCaps", "Clotho", "MusicCaps", "SoundDescs"],
        "A2T RET": ["AudioCaps_A2T", "Clotho_A2T", "MusicCaps_A2T"],
        "A-QA": ["MMAU", "MMAR"],  # NO ClothoAQA
    },
    "audiovisual": {
        "A2V RET": ["AVE_A2V", "VALOR32k_A2V"],
        "V2A RET": ["AVE_V2A", "VALOR32k_V2A"],
        "T2VA RET": ["AVHBench_T2VA", "VALOR32k_T2VA"],
        "VA2T RET": ["AVHBench_VA2T", "VALOR32k_VA2T"],
        "AV-QA": ["AVHBenchQA", "DailyOmni"]
    },
}

dataset_to_task = {
    dataset: task
    for modality, task_dict in TASKS.items()
    for task, dataset_list in task_dict.items()
    for dataset in dataset_list
}

# "exp_name": OUTPUT_DIR  # output directory for a specific model generated from evaluation
EXP_DICT = {

}  # yapf: disable
MODELS = [k for k in EXP_DICT.keys()]

SAVE_NAME = "save_name.csv"


def main():
    metric_dict = {}
    avg_metric_dict = {}
    modality_metric_dict = {}
    for exp in MODELS:
        metric_dict[exp] = {}
        avg_metric_dict[exp] = {}
        modality_metric_dict[exp] = {}

        exp_path = EXP_DICT[exp]
        for modality, task_dict in TASKS.items():
            modality_scores = []
            modality_scores_v2 = []

            for sub_task, subtasks in task_dict.items():
                scores = []
                for subtask in subtasks:
                    score_file = os.path.join(exp_path, modality, f"{subtask}_score.json")
                    hit1 = json.load(open(score_file, "r"))["hit@1"] * 100
                    metric_key = f"{modality}/{sub_task}/{subtask}"
                    metric_dict[exp][metric_key] = hit1
                    scores.append(hit1)
                avg_metric_key = f"{modality}/{sub_task} AVG ({len(subtasks)})"
                avg_metric_dict[exp][avg_metric_key] = sum(scores) / len(scores)
                modality_scores.extend(scores)
                modality_scores_v2.append(sum(scores) / len(scores))
            modality_metric_key = f"{modality} AVG ({len(modality_scores)})"
            modality_metric_dict[exp][modality_metric_key] = sum(modality_scores_v2
                                                                 ) / len(modality_scores_v2)

    avg1 = pd.DataFrame(modality_metric_dict)
    avg2 = pd.DataFrame(avg_metric_dict)
    main_df = pd.DataFrame(metric_dict)
    avg1.loc[f"Avg All ({len(main_df)})"] = avg1.mean(axis=0)

    df_all = pd.concat((avg1, avg2, main_df), axis=0)
    df_all.to_csv(SAVE_NAME)


if __name__ == "__main__":
    main()
