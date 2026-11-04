from .audio_text_dataset import AudioTextDataset
from .avinstruct import AVInstructDataset
from .avqa import AVQADataset
from .base_dataset import BaseRetrievalDataset
from .llavahound import LLaVAHoundCaptionDataset, LLaVAHoundQADataset
from .mmeb_train import MMEBTrainDataset
from .musiccaps import MusicCapsDataset
from .musicinstruct import MusicInstructDataset
from .pvd import PVDataset
from .wavcaps import WavCapsDataset

DATASET_REGISTRY = {
    "base_retrieval": BaseRetrievalDataset,
    "llavahound_caption": LLaVAHoundCaptionDataset,
    "llavahound_qa": LLaVAHoundQADataset,
    "mmeb_train": MMEBTrainDataset,
    "musicinstruct": MusicInstructDataset,
    "avqa": AVQADataset,
    "avinstruct": AVInstructDataset,
    "musiccaps": MusicCapsDataset,
    "audio_text_data": AudioTextDataset,
    "wavcaps": WavCapsDataset,
    "pvd": PVDataset,
}
