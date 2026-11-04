import logging
import os

from typing import Any, Dict, List, Optional

from datasets import load_dataset

from tevatron.retriever.datasets.asset_registry import ASSET_REGISTRY

from .base_dataset_class import BaseDataset

logger = logging.getLogger(__name__)


def _split_prompt_and_text(view, query, corpus):
    prompt_dict = {  # (query prompt, with_query_text, corpus prompt, with_corpus_text)
        # clsf
        "ImageNet_1K": ("Represent the given image for classification", False, None, True),
        "N24News": ("Represent the given news image with the following caption for domain classification:", True, None, True),  # noqa
        "HatefulMemes": ("Represent the given image for binary classification to determine whether it constitutes hateful speech or not", False, None, True),  # noqa
        "VOC2007": ("Identify the object shown in the image", False, None, True),
        "SUN397": ("Identify the scene shown in the image", False, None, True),

        # qa
        "OK-VQA": ("Represent the given image with the following question:", True, None, True),
        "A-OKVQA": ("Represent the given image with the following question:", True, None, True),
        "DocVQA": ("Represent the given image with the following question:", True, None, True),
        "InfographicsVQA": ("Represent the given image with the following question:", True, None, True),  # noqa
        "ChartQA": ("Represent the given image with the following question:", True, None, True),
        "Visual7W": ("Represent the given image with the following question:", True, None, True),

        # retrieval
        "VisDial": ("Represent the given dialogue about an image, which is used for image retrieval:", True, "Represent the given image", False),  # noqa
        "CIRR": ("Given an image, find a similar everyday image with the described changes:", True,"Represent the given image." , False),  # noqa
        "VisualNews_t2i": ("Retrieve an image of this news caption.", True, "Represent the given image.", False),  # noqa
        "VisualNews_i2t": ("Find a caption for the news in the given photo.", False, None, True),
        "MSCOCO_t2i": ("Find me an everyday image that matches the given caption:", True, "Represent the given image" , False),  # noqa
        "MSCOCO_i2t": ("Find an image caption describing the given everyday image.", False, None , True),  # noqa
        "NIGHTS": ("Find a day-to-day image that looks similar to the provided image.", False, "Represent the given image.", False),  # noqa
        "WebQA": ("Find a Wikipedia image that answers this question:", True, "Represent the given Wikipedia image with related text information:", True),  # noqa
        "MSCOCO": ("Select the portion of the image that isolates the object labeled as", True, "Represent the given cropped image of the object", False),  # noqa
    }
    assert view in prompt_dict
    (query_prompt, with_query_text, corpus_prompt, with_corpus_text) = prompt_dict[view]
    if with_query_text:
        assert query_prompt in query
        query_text = query.split(query_prompt)[1].strip()
    else:
        query_text = None

    if with_corpus_text:
        if corpus_prompt is not None:
            assert corpus_prompt in corpus
            corpus_text = corpus.split(corpus_prompt)[1].strip()
        else:
            corpus_text = corpus
    else:
        corpus_text = None
    return (query_prompt, query_text), (corpus_prompt, corpus_text)


def build_text_image_content(
    prompt: Optional[str], text: Optional[str], image_rel_path: str, asset_prefix: str
) -> List[Dict[str, Any]]:
    image_rel_path = (image_rel_path or "").strip()
    image = os.path.join(asset_prefix, image_rel_path) if image_rel_path else None

    item_dict = {"prompt": prompt, "text": text, "image": image}
    return item_dict


def is_empty_row(row: Dict[str, Any], image_token: str = "<|image_1|>") -> bool:
    """
    Empty row definition:
      - query side is empty: (not qry) and (not qry_image_path)
        OR
      - passage side is empty: (not pos_text) and (not pos_image_path)
    """
    # normalize to safe strings
    qry = (row["qry"] or "").strip()
    qry_img = (row["qry_image_path"] or "").strip()
    pos = (row["pos_text"] or "").strip()
    pos_img = (row["pos_image_path"] or "").strip()

    query_empty = (qry == "") and (qry_img == "")
    passage_empty = (pos == "") and (pos_img == "")

    # also filter cases where image token exists, but image path not specified
    query_image_empty = (image_token in qry) and (qry_img == "")
    passage_image_empty = (image_token in pos) and (pos_img == "")
    return query_empty or passage_empty or query_image_empty or passage_image_empty


def filter_empty_rows(
    ds,
    subset_name,
    image_token: str = "<|image_1|>",
    num_proc: int | None = None,
    batched: bool = True
):

    # batched=True case: return a list[bool] mask
    def _keep_mask(batch: Dict[str, List[Any]]):
        qrys = batch["qry"]
        qry_imgs = batch["qry_image_path"]
        poss = batch["pos_text"]
        pos_imgs = batch["pos_image_path"]

        keep = []
        for qry, qry_img, pos, pos_img in zip(qrys, qry_imgs, poss, pos_imgs):
            if subset_name == "WebQA":
                qry = qry.replace(image_token, "")  # patch
            row = {
                "qry": qry,
                "qry_image_path": qry_img,
                "pos_text": pos,
                "pos_image_path": pos_img,
            }
            keep.append(not is_empty_row(row, image_token=image_token))
        return keep

    return ds.filter(_keep_mask, batched=True, num_proc=num_proc)


class MMEBTrainDataset(BaseDataset):

    def __init__(self, data_args, cfg):
        super().__init__(data_args, cfg)

    def _build_dataset(self):
        num_proc = self.num_proc
        ann_path = os.path.join(self.asset_prefix, self.view, "original*.parquet")
        dataset = load_dataset("parquet", data_files=ann_path, split="train")
        self.dataset = filter_empty_rows(
            dataset, subset_name=self.view, num_proc=num_proc, batched=True
        )

    def _check_and_download_assets(self):
        downloader = ASSET_REGISTRY.get(self.data, None)
        if downloader is not None:
            download = self.data_args.dataset_download
            self.asset_prefix = downloader(
                self.data_root, self.name, dataset_name=self.view, download=download
            )

    def _build_modality_dict(self, row, **kwargs):
        query = row.get("qry", "").replace("<|image_1|>", "").strip()
        passage = row.get("pos_text", "").replace("<|image_1|>", "").strip()

        query_img_path = row.get("qry_image_path", "")
        passage_img_path = row.get("pos_image_path", "")

        ((query_prompt, query_text),
         (corpus_prompt, corpus_text)) = _split_prompt_and_text(self.view, query, passage)

        formatted_query = build_text_image_content(
            prompt=query_prompt,
            text=query_text,
            image_rel_path=query_img_path,
            asset_prefix=self.asset_prefix,
        )
        formatted_passage = build_text_image_content(
            prompt=corpus_prompt,
            text=corpus_text,
            image_rel_path=passage_img_path,
            asset_prefix=self.asset_prefix,
        )

        return formatted_query, formatted_passage
