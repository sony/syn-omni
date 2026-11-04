import json
import logging
import os

from datasets import Dataset as HF_Dataset

from tevatron.retriever.datasets.asset_registry import ASSET_REGISTRY
from tevatron.retriever.datasets.utils import load_video_frames

from .base_dataset_class import BaseDataset

logger = logging.getLogger(__name__)


class AVInstructDataset(BaseDataset):

    def _build_dataset(self):
        anns = []
        for _data_files in self.data_files:
            _data_files = os.path.join(self.asset_prefix, _data_files)
            anns.extend(json.load(open(_data_files, "r")))

        valid_ids_path = os.path.join(self.asset_prefix, "valid_ids.json")
        assert os.path.exists(valid_ids_path), \
            f"no avqa videos prepared: {valid_ids_path}; run with --dataset_download True"
        valid_ids = set(json.load(open(valid_ids_path, "r")))

        # filter annotations by the successfully extracted videos
        filtered_anns = []
        for ann in anns:
            if "_" not in ann["video_id"]:
                # skip music-avqa samples (long, 1min videos)
                continue

            avqa_excludes = ["WvJBXBn8EXA_000015", "sOA-ZjDqSp0_000022", "nHGoKS23x9U_000000", "Xv4AVT2QYhA_000100", "C_pnsyNXphA_000030", "fAMipExdVms_000011", '-Lv13WPa4xk_000170', '-MGxPJLnfW0_000504', '-XUb2wq5U7E_000135', '-XUb2wq5U7E_000228', '-aNuyRVE25c_000088', '-h8I3cqpgnA_000085', '-hzQtnz2GuE_000213', '-hzQtnz2GuE_000240', '-jSirHVQAOk_000011', '-kPDH9n0PG4_000310', '-wHYt6692rw_000414', '03hFQ4wxzRE_000087', '03hFQ4wxzRE_000099', '0DioVNaipVI_000202', '0F0YvdKcUu0_000040', '0F0YvdKcUu0_000060', '0JE4WXw4VFs_000022', '0JE4WXw4VFs_000032', '0N0C0Wbe6AI_000030', '0QdW02J2hRQ_000326', '0QdW02J2hRQ_000342', '0Rqa18lmDtw_000153', '0Wzh7Hou30s_000175', '0XGMNVKYVbs_000357', '0XSNrKgA94g_000030', '0aPwSLMzze4_000005', '0dxxgy3H8sQ_000224', '0oy8C07tqpk_000995', '18212B4yfLg_000080', '190Psk1FZEc_000032', '190Psk1FZEc_000093', '1EN0zAegNKI_000012', '1ETrMAVZdY8_000030', '1NqprCWM3Dc_000229', '1OMUQwqU5L8_000180', '1PZC1Sz9XFM_000010', '1e0_4dwF9A4_000030', '1esWjj1Z8GE_000353', '1hwkxJOl1PY_000020', '1jVv21yg4qw_001135', '1lJZ3GZGxm0_000131', '1oJAVJPX0YY_000020', '1xhlUqrlje8_000039', '21Dc8a8X8qg_000007', '2ALAQg1eMWY_000531', '2KxW-XXQWdI_000022', '2M3x2wvIBBU_000004', '2PTujSVEfwo_000024', '2VG7r4RXuEo_000310', '2W4dw8XuetE_000050', '2Yt6pKLU_10_000000', '2_26akyADLs_000035', '2utaodyuqRY_000050', '3AaHmZvZclA_000048', '3RKsH2Jkisc_000010', '3qpNZkxd-OA_000030', '4MuPg9zkTzI_000011', '52GBb6ArCqA_000090', '5b6D5c-ULNY_000235', '6MxUq79Dr4w_000253', '8ePlL__G4Aw_000005', '97y13cd2E5U_000017', 'BBUElSZwU6U_000000', 'BJ7_yh9_19U_000006', 'BL0frJ7hEng_000208', 'CBvtXmlORYU_000030', 'CEOC7a6DjqQ_000111', 'COvPKvoAz_c_000000', 'CThe_IxJs-I_000030', 'CX2bsgtQoJU_000030', 'D3OWF8T8-H4_000168', 'DxI5FzXKRZk_000030', 'E_Tqm4Twvf0_000010', 'HP5IYVg9xxA_000183', 'HqP3nvOa8Qs_000196', 'JsTj-8bNObw_000694', 'KOJFnwjNvdE_000048', 'M6IfpC1iDPk_000060', 'ON_1RUMO5AY_000021', 'OUnOCwt9FjQ_000040', 'QEr9K9o5a4A_000030', 'Qm7qy9cJcO8_000003', 'S8xzP4Bs0lc_000027', 'UnUW__o8SLo_000084', 'UqI3333ulUg_000059', 'VyJam_Oprq4_000020', 'WKaj5kZh13Q_000028', 'WrO6kv9Az_Y_000057', 'XfYylx3WaRM_000083', 'Y9fxIvG54SQ_000030', 'Yhj_R9_f4qg_000015', 'YwLy4Szt6eQ_000216', 'YxuSDGXhwg0_000018', 'Z9jhyJ-0FB8_000030', 'Z9rU1GfYU-4_000076', 'ZIgZ_OKQyF0_000215', 'ZexJgASGkyU_000073', 'ZzWXZ5cVF20_000030', '_wApMwZD2G0_000030', 'aOZLQhoACtQ_000060', 'bxts5znycds_000067', 'e5mHdmvF8uc_000140', 'ewjs2aqkmXw_000037', 'gWHf2dvKavw_000015', 'htzGgSLsKtw_000001', 'i7eLyHyp_qM_000030', 'jGzV4RfUc7w_000000', 'jSZSgsSgiL4_000019', 'jYvet1IHRAM_000030', 'jq_GWSrHRTU_000320', 'ju9VQ3xXxFw_000020', 'l8RebKdoFXo_000030', 'm0g3mvwKsOg_000010', 'mBcgA8593XU_000330', 'mEdosepdsq4_000040', 'nOV3wofF1jw_000030', 'nOV3wofF1jw_000041', 'nOr0M1EXZSg_000200', 'nP9_duu5FL0_000006', 'n_k5FKt3X64_000030', 'olhcxl5azWE_000015', 'ou-5wdKzJ6c_000147', 'ou-5wdKzJ6c_000165', 'rDNVf4rtp-U_000163', 'rIFsSxEY44I_000437', 'rJimmm70DLo_000000', 'rbyM2qDkpz4_000010', 'rjYARQtW5jg_000079', 'ryBaZTA2FNw_000030', 'sOwc2jAgqOQ_000016', 'sW41Cbfa1xY_000090', 'sa6z0mxsBo8_000030', 'sjO5epSBSdY_000047', 't4_tU-zSF-c_000000', 'tDCdATTKsAc_000070', 'tGyD4D8t1IU_000010', 'tXHvy5OL4A0_000027', 'tkUxSPT6_D8_000020', 'uFfjq0JGXcI_000154', 'un2kcLYinRo_000030', 'vOLKqHTXmfE_000130', 'wkj56nX5PS8_000173', 'xTPmoYwgKf4_000030', 'z9Sy1VD_WoE_000040', 'zLozVjMxPns_000005', 'zRTrhDiyq2o_000009', '-6gxzS2ydY4_000100', '-MtYMG8dmq4_000098', '-ovFIwvxiwI_000002', '-rE8CpfEUxQ_000060', '03FBDSuWg1k_000429', '03gauhxTvn0_000040', '0QBjH8gIf1I_000090', '180ipjKA4WE_000070', '1EUE2fs-hZA_000060', '1gsVyTb-bpI_000029', '21sANXoTHRM_000090', '2F6quyMLi3I_000017', '2ZYgLySs6j0_000040', '2tMwmcoGhus_000080', '386PZW3EbH4_000012', '3Wt_ldHN6QE_000113', '3MVJwvIwfro_000064', '3xuqDfm_bY8_000030', '4BiAGSdG4I4_000028', '54WNmykydBo_000083', '5eaDdvKRubI_000030', '68LbBh_uP4k_000199', '6BxMf10fgq0_000030', '6EHt6k0MCqU_000030', '67BsqRkh-dU_000010', '6d_RTzFfQUE_000289', '6on8OsUCIRc_000009', '6eR4-X9_bog_000030', '72-UFjSlpvM_000314', '7TMdhrlYovQ_000010', '7waWJ4PD23U_000090', '8AXNGUMaxI8_000016', '8_OFbRpXIOk_000030', 'AcNG-vRGoPE_000020', 'BEXcmFXzn44_000000', 'BdUgCoYxuWo_000179', 'Br4vjAGbVl0_000005', 'BsjoMDuHniE_000010', 'BndvWLg8hkM_000080', 'BzjPU48ihwQ_000015', 'CEOC7a6DjqQ_000037', 'Cb2k9JwDkDQ_000030', 'CTxq1zDlg2s_000030', 'CcopmE2srOU_000070', 'CqI98TCsUZw_000040', 'D5vVmwsLtCo_000022', 'DFYOtfJsXgM_000058', 'EF7pVapQQtE_000125', 'EGPN1YVr8C0_000040', 'Eb-1_P_CiRA_000003', 'EoquowIMQWA_000000', 'HBW2mqqOhT0_000080', 'HzIIDttuRNY_000015', 'IS9n31HdN6E_000110', 'Ie2xHRNnrDU_000080', 'J8xnnXILOOw_000024', 'JOlS4h6oTKw_000026', 'KSmL4L5SI6E_000003', 'KksMNKXuiNw_000012', 'Kia_gYhJ7k0_000030', 'LTQOc63shJc_000095', 'LYP0nFAka3Y_000060', 'Lt4vAGdQIvU_000086', 'MRazdvVjmbQ_000003', 'MDvCAjl4FlU_000080', 'MSmg48orZqI_000065', 'NEASruN00vY_000030', 'NeJetBbkQ-o_000001', 'NsO8j9P99Xo_000010', 'O8aMDUSXhBw_000010', 'O8fN3f8zpyY_000030', 'NzZHB9OpoFU_000090', 'OExC5APNfkg_000030', 'OI1yNIYSDpc_000030', 'OSHPQIGmOMQ_000057', 'OIGovbDeybw_000030', 'ObV0Fz5rJHQ_000035', 'Op6o3GWHzPQ_000123', 'OgswFsHIkeg_000058', 'OjwXIYOEuBc_000080', 'Ozt43eCJ6AM_000260', 'P09qD5REYso_000061', 'P0Nvk6Dk6Bw_000030', 'PvnP164MdVM_000016', 'PzDRgxLgCfY_000100', 'PmZ2kdlUQD8_000060', 'PvnP164MdVM_000070', 'Qvnm9Zi_Y_M_000111', 'R_iiyKHZfz4_000012', 'RgiM_kC4tdM_000017', 'SLrb_8ksUq4_000000', 'SUKGkMu6o_A_000060', 'SftMBDbpkoA_000004', 'Sm_670Aqo5g_000003', 'TRRHq8Sr57Q_000030', 'TWtvtp4iU4Q_000030', 'T7XD4NxxZm8_000070', 'Ta6wdv4lVKM_000048', 'UBX1QvUbBXg_000100', 'Tszdt6LigP8_000070', 'Ttckxy6OIQo_000080', 'UgrbuCAp5qQ_000030', 'UjASO29kNgM_000190', 'V-Sm4BmSX-g_000121', 'VznVCznVL9c_000030', 'XYTAXqEWA34_000510', 'Y3OgnFoSdw4_000009', 'XspazHCZqIk_000030', 'YD9g1A4_NpI_000030', 'YQW1Kp3fBaA_000090', 'Ye72yJyWxs8_000041', 'Yg8YFu_iPXE_000037', 'YVwgyAPjxFQ_000060', 'Yfh8NyEKM3M_000050', 'YvuHBp0F-NA_000030', 'ZE-TygRDXsA_000010', '_-xTBYzC-mA_000009', '_KErY6-d4I4_000086', '_caFr8RrH-Q_000030', 'a0KnoH0oea8_000030', 'aZ8dakM5L9U_000030', 'afhXA-Gsc0M_000156', 'bLPv7gqhzPw_000000', 'bBfEkxp_5z4_000060', 'bRjxB4yuUhQ_000025', 'c5EJlvjK0iQ_000118', 'dMGJmdzhja0_000196', 'dG7RFy7mAyY_000030', 'eBFPD8YrqiA_000140', 'emBOW1xaZ_0_000030', 'edo0FqKspI4_000020', 'f5wADp98HsQ_000002', 'fNttvad732o_000006', 'fpOBHTPDp-c_000011', 'fwCi21ZYUIk_000020', 'g7bhQwjUSq0_000080', 'gCOqV3nUhuw_000060', 'gpSrXa5kMls_000030', 'gtWWoHE9Iws_000242', 'h1GbgVQLXOc_000088', 'hB7i7_WunIU_000060', 'hYXzvL-i-ow_000060', 'hiU8NlFzt0c_000060', 'hqtHiZ_hK8c_000070', 'id-RLDdMNd0_000060', 'jFCl-v0DO4c_000350', 'jHm0mBsqJ6Y_000069', 'jkqI4h9pBpQ_000070', 'kDYQlgtSs0A_000040', 'kZGSz4uYQ7Q_000090', 'l9Qc0Ey44eQ_000120', 'lXVtBj1RO3Q_000010', 'll_w_X6tdQU_000060', 'mbnMnm-S3bI_000008', 'mTG_kwsPVP0_000080', 'mWYUJwz3Jxo_000070', 'mxS_LTY_NzI_000030', 'nu5VmYIBcA0_000160', 'oEOkYR5gq3M_000050', 'pczNVFzNNLI_000030', 'q-5CFkRPs4A_000001', 'q74Zzmi8BRk_000070', 'rFEXl4w_dDY_000070', 'rFUKCgXAqq8_000000', 'rF9veq9_oCc_000030', 'rQC750bL4iE_000060', 's44oO5rBB28_000190', 'rtvzbdWOYhg_000040', 'sD9gTAFDq40_000044', 's4m6qg64A4s_000050', 'sWdJR9dInhw_000019', 'spplzADWm90_000002', 'suyg9tlfzbM_000220', 'simd0Xwhd2Q_000040', 'skgVkVbw5SM_000030', 'tD9HzR6J4vY_000030', 'tcsOCNsi9BE_000065', 'tupSLysCAz4_000050', 'tgI7EufO1L4_000030', 'tlyLCgkcW8s_000065', 'uCppmoZiXUY_000323', 'uAB0hA7zxKI_000030', 'wGG7dEgqUOo_000060', 'wV0iMbAVe8M_000050', 'xJPrRpevxsA_000030', 'xRO2FFv_cfY_000093', 'xIBh9MOGGKY_000030', 'xk_8TdtGuiI_000176', 'xscfyo6SVrk_000030', 'y4mkfzG1muA_000050', 'yD68zBpXJVo_000122', 'xto6OjVtkIk_000085', 'xvinwK9b570_000070', 'ybl-cb4hV0Y_000120', 'ye9s1GrJJjQ_000194', 'yPyldlYz_tA_000062', 'yoNGOt1pyq0_000030', 'zWnKdFtKSEI_000010', 'zstMUBh6698_000003']  # noqa  # yapf:disable
            if ann["video_id"] in avqa_excludes:  # skip manually-found broken videos
                continue
            if ann["video_id"] in valid_ids:
                inst, resp = ann["conversations"][0]["value"], ann["conversations"][1]["value"]
                inst = inst.replace("<Q>", "").replace("<video>", "")
                new_ann = {
                    "question": inst,
                    "answer": resp,
                    "video_id": ann['video_id'],
                    "prefix": self.asset_prefix
                }
                filtered_anns.append(new_ann)

        self.dataset = HF_Dataset.from_list(filtered_anns)

    def _check_and_download_assets(self):
        downloader = ASSET_REGISTRY.get(self.data, None)
        if downloader is not None:
            download = self.data_args.dataset_download
            self.asset_prefix = downloader(self.data_root, "avinstruct_data", download=download)

    def _build_modality_dict(self, row, **kwargs):
        question, answer, video_id, prefix = row["question"], row["answer"], row["video_id"], row[
            "prefix"]  # noqa

        frame_path = os.path.join(prefix, "frames", video_id)
        frames = load_video_frames(frame_path, self.data_args.video_nframes)
        audio_path = os.path.join(prefix, "audio", f"{video_id}.wav")
        assert len(frames) and os.path.exists(audio_path)
        audio = audio_path

        formatted_query = {"audio": audio, "prompt": question, "video": frames}
        formatted_pos_doc = {"text": answer, "audio": None, "image": None, "video": None}

        return formatted_query, formatted_pos_doc
