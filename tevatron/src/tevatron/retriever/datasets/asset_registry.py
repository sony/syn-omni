from .asset_downloader import (
    activitynet_video_downloader,
    avinstruct_downloader,
    didemo_video_downloader,
    llavahound_downloader,
    mmeb_train_downloader,
    msrvtt_video_downloader,
    msvd_video_downloader,
    musicbench_audio_downloader,
    musicinstruct_downloader,
    pvd_downloader,
    wavcaps_downloader,
)

ASSET_REGISTRY = {
    "msvd": msvd_video_downloader,
    "msrvtt": msrvtt_video_downloader,
    "activitynet": activitynet_video_downloader,
    "didemo": didemo_video_downloader,
    "musicbench": musicbench_audio_downloader,
    "llavahound": llavahound_downloader,
    "mmeb_train": mmeb_train_downloader,
    "musicinstruct": musicinstruct_downloader,
    "avinstruct": avinstruct_downloader,
    "wavcaps": wavcaps_downloader,
    "pvd": pvd_downloader,
}
