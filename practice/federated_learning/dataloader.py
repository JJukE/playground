import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def music2genre(label_dir):
    label_dir = Path(label_dir)
    music_genre = {}
    for file in label_dir.iterdir():
        assert file.is_file(), f"There is any unknown directory: {file}"
        name = file.stem
        json_file = label_dir / (name + ".json")
        with open(json_file, "r") as f:
            genre_dict = json.load(f)
        genre = genre_dict["style2"]
        music_genre[name] = genre
    return music_genre


class DanceDataset(Dataset):
    def __init__(
        self,
        split: str,
        dataset_path: str,
        data_name: str,
        seq_len: int, # full sequence length
        stride: int = None, # number of stride
        num_windows: int = None, # num of windows (window: num of samples in the length)
        wav_path: str = None,
        # dim_smpl: int = 139,
        aist_split_path: str = None,
        finedance_partial: str = None,
    ):
        """
        features are extracted by librosa
        AISTPP: 30FPS or 60FPS
            sampling rate: hopsize * fps (original sr of wav files: 48000)
            hop_size: 512
            if (FPS: 30, length: 5s, stride_length: 0.5s)
                # window_size: 512 * 30 * 5 = 76800 (76800 audio samples in 5s)
                seq_len: 30 * 5 = 150
                stride: 30 * 0.5 = 15
                num_windows: 150 // (30 * 0.5) = 10
        FINEDANCE: 30FPS
            sampling rate: hopsize * fps (original sr of wav files: 76800, but not given)
            hop size: 512
            if (FPS: 30, seq_len: 1024, num_windows: 5)
                # window_size: 512 * 30 * 5 = 76800 (384000 audio samples in 5s)
                seq_len: 1024
                num_windows: 5
                stride: 1024 // (30*5) = 6
        """
        assert data_name in ["AISTPP_30FPS", "AISTPP_60FPS", "FINEDANCE"]
        self.motion_dir = Path(dataset_path) / data_name / "motion"
        self.music_dir = Path(dataset_path) / data_name / "music"
        self.train = split == "train"
        self.seq_len = seq_len
        if "AISTPP" in data_name:
            assert stride is not None and aist_split_path is not None
            self.stride = stride
            self.num_windows = seq_len // stride
        if "FINEDANCE" in data_name:
            assert num_windows is not None and finedance_partial is not None
            self.partial = finedance_partial
            self.num_windows = num_windows
            self.stride = seq_len // num_windows
        
        self.aist_split_path = Path(aist_split_path) if aist_split_path is not None else None
        
        if "FINEDANCE" in data_name:
            self.music2genre = music2genre(Path(dataset_path) / data_name / "label_json")
        
        self.motion_indices = []
        self.music_indices = []
        motion_all, music_all, genre_list, wav_list = [], [], [], []
        ignore_list, train_list, test_list = self.get_train_test_list(dataset=data_name)
        if self.train:
            self.data_list = train_list
        else:
            self.data_list = test_list
        
        total_length = 0
        for name in tqdm(self.data_list):
            fname = name + ".npy"
            if fname[:-4] in ignore_list:
                continue
            
            motion = np.load(self.motion_dir / fname)
            music = np.load(self.music_dir / fname)
            min_len = min(motion.shape[0], music.shape[0])
            motion, music = motion[:min_len], music[:min_len]
            if motion.shape[-1] == 319:
                assert "FINEDANCE" in data_name
                motion = motion[:, :139]
            elif motion.shape[-1] == 151:
                assert "AIST" in data_name
            else:
                raise ValueError("Input motion shape error: ", motion.shape)
            num_sequences = (min_len - self.seq_len) // self.stride + 1
            
            wav = str(Path(wav_path) / (name + ".wav"))
            if "FINEDANCE" in data_name:
                genre = self.music2genre[name]
                genre = torch.tensor([GENRES_FINEDANCE[genre]])
            elif "AISTPP" in data_name:
                genre = fname.split("_")[0]
                genre = torch.tensor([GENRES_AIST[genre]])
            
            if self.train:
                valid_indices = [
                    i for i in range(num_sequences)
                    if motion[i*self.stride : i*self.stride+self.seq_len].std(axis=0).mean() > 0.07
                ]
                genre_list.extend(len(valid_indices) * [genre])
                wav_list.extend(len(valid_indices) * [wav])
                indices = np.array(valid_indices) * self.stride + total_length
            else:
                genre_list.extend(num_sequences * [genre])
                wav_list.extend(num_sequences * [wav])
                indices = np.arange(num_sequences) * self.stride + total_length
            
            # augmentation (deprecated)
            # if mix:
            #     motion_indices = []
            #     music_indices = []
            #     num = (len(indices) - 1) // 8 + 1
            #     for i in range(num):
            #         motion_indices_tmp, music_indices_tmp = np.meshgrid(indices[i*8 : (i+1)*8], indices[i*8 : (i+1)*8])
            #         motion_indices.extend(motion_indices_tmp.reshape((-1)).tolist())
            #         music_indices.extend(music_indices_tmp.reshape((-1)).tolist())
            # else:
            motion_indices = indices.tolist()
            music_indices = indices.tolist()
            
            self.motion_indices.extend(motion_indices)
            self.music_indices.extend(music_indices)
            total_length += min_len
            
            motion_all.append(motion)
            music_all.append(music)
        
        self.motion = np.concatenate(motion_all, axis=0).astype(np.float32)
        self.music = np.concatenate(music_all, axis=0).astype(np.float32)
        self.genre_list = genre_list
        self.wav_list = wav_list
        print(f"Loaded {data_name} which has {len(self.motion_indices)} samples...")


    def __len__(self):
        return len(self.motion_indices) # number of sliced sequences
    
    
    def __getitem__(self, idx):
        motion = self.motion[self.motion_indices[idx]:self.motion_indices[idx] + self.seq_len]
        music = self.music[self.music_indices[idx]:self.music_indices[idx] + self.seq_len]
        genre = self.genre_list[idx]
        wav = self.wav_list[idx]
        return {"motion": motion, "music": music, "genre": genre, "wav": wav}
    
    
    def get_train_test_list(self, dataset="FineDance"):
        if dataset in ["AISTPP_30FPS", "AISTPP_60FPS"]:
            assert self.aist_split_path is not None
            with open(self.aist_split_path / "crossmodal_train.txt", "r") as f:
                train = [line.strip() for line in f]
            with open(self.aist_split_path / "crossmodal_test.txt", "r") as f:
                test = [line.strip() for line in f]
            # with open(self.aist_split_path / "crossmodal_val.txt", "r") as f:
            #     test = [line.strip() for line in f]
            with open(self.aist_split_path / "ignore_list.txt", "r") as f:
                ignore = [line.strip() for line in f]
            return ignore, train, test
        
        elif dataset == "AISTPP_LONG263":
            train, test, ignore = [], [], []
            for file in Path(self.motion_dir).iterdir():
                if file.suffix != ".npy":
                    continue
                file = file.stem
                if file.split("_")[-1] in ["mLH5", "mJS4", "mBR3", "mMH2", "mPO1", "mWA0"]:
                    test.append(file)
                else:
                    train.append(file)
            return ignore, train, test

        else:
            all_items = [str(i).zfill(3) for i in range(1, 212)]
            
            test = {"063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"}
            ignore = {"116", "117", "118", "119", "120", "121", "122", "123", "202"}
            tradition = {"005", "007", "008", "015", "017", "018", "021", "022", "023", "024", "025", "026", "027", "028", "029", "030", "032", "032", "033", "034", "035", "036", "037", "038", "039", "040", "041", "042", "043", "044", "045", "046", "047", "048", "049", "050", "051", "072", "073", "074", "075", "076", "077", "078", "079", "080", "081", "082", "083", "104", "105", "106", "107", "108", "109", "110", "111", "112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "122", "123", "126", "127", "132", "133", "134",  "135", "136", "137", "138", "139", "140", "141", "142", "143", "144", "145", "146", "147", "148", "151", "152", "153", "154", "155", "170"}
            
            train = [item for item in all_items if item not in test]
            
            if self.partial == "full":
                return list(ignore), train, list(test)
            elif self.partial == "modern":
                train = [item for item in train if item not in tradition]
                test = [item for item in test if item not in tradition]
                return list(ignore), train, test
            elif self.partial == "tradition":
                train = [item for item in train if item in tradition]
                test = [item for item in test if item in tradition]
                return list(ignore), train, test


def load_dataloaders(
    batch_size, num_workers, dataset_path, data_name, seq_len,
    stride=None, num_windows=None, wav_path=None, aist_split_path=None, finedance_partial=None, ddp=None, **kwargs
):
    # [local diffusion, global diffusion, ld finetuning]
    ds_train = DanceDataset(
        split="train",
        dataset_path=dataset_path,
        data_name=data_name,
        seq_len=seq_len, # [256, 1024, 1024]
        stride=stride, # [8, 10, 8]
        num_windows=num_windows,
        # dim_smpl=dim_smpl, # 139
        wav_path=wav_path,
        aist_split_path=aist_split_path,
        finedance_partial=finedance_partial,
    )
    ds_valid = DanceDataset(
        split="valid",
        dataset_path=dataset_path,
        data_name=data_name,
        seq_len=seq_len, # [256, 1024, 1024]
        stride=stride, # [8, 10, 8]
        num_windows=num_windows,
        # dim_smpl=dim_smpl, # 139
        wav_path=wav_path,
        aist_split_path=aist_split_path,
        finedance_partial=finedance_partial,
    )

    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    dl_valid = DataLoader(ds_valid, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return dl_train, dl_valid


SMPL_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]

SMPLX_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13,
                        15, 17, 16, 19, 18, 21, 20, 22, 24, 23,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
SMPLX_POSE_FLIP_PERM = []
for i in SMPLX_JOINTS_FLIP_PERM:
    SMPLX_POSE_FLIP_PERM.append(3*i)
    SMPLX_POSE_FLIP_PERM.append(3*i+1)
    SMPLX_POSE_FLIP_PERM.append(3*i+2)

def flip_pose(pose):
    """ Flip pose. The flipping is based on SMPL-X parameters. """
    pose = pose[:,SMPLX_POSE_FLIP_PERM]
    # we also negate the second and the third dimension of the axis-angle
    pose[:,1::3] = -pose[:,1::3]
    pose[:,2::3] = -pose[:,2::3]
    return pose

GENRES_AIST = {
    'gBR': 0,
    'gPO': 1,
    'gLO': 2,
    'gMH': 3,
    'gLH': 4,
    'gHO': 5,
    'gWA': 6,
    'gKR': 7,
    'gJS': 8,
    'gJB': 9,
}

GENRES_FINEDANCE = {
    'Breaking': 0,
    'Popping': 1,
    'Locking': 2,
    'Hiphop':3,
    'Urban':4,
    'Jazz':5,
    'jazz':5,

    'Tai':6,
    'Uighur':7,
    'Hmong':8,
    'Dai':6,
    'Wei':7,
    'Miao':8,

    'HanTang':9,
    'ShenYun':10,
    'Kun':11,
    'DunHuang':12,

    'Korean':13,
    'Choreography':14,
    'Chinese':15,
}
