import re
from pathlib import Path
import torch
from torch.utils.data import Dataset
from sst     import create_sst
from config  import DEFAULT_STL_DIR, POINTS_PER_CS
import glob
import os

class ShipSSTDataset(Dataset):
    def __init__(self, stl_dir: str, N: int = POINTS_PER_CS):
        # load all .stl files regardless of prefix
        self.files = sorted(glob.glob(os.path.join(stl_dir, "*.stl")), key=self._num_key)
        self.N = N

    def _num_key(self, path):
        # optional: extract first number for ordering
        m = re.search(r"\d+", os.path.basename(path))
        return int(m.group()) if m else float('inf')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        stl_path = self.files[idx]
        # ─── ADD THESE LINES ───
        print(f"Start: Dataset __getitem__ idx={idx}, file={stl_path}")
        # ────────────────────────
        try:
            sst = create_sst(stl_path, N=self.N)
            print("Done:",stl_path)
            return torch.from_numpy(sst).float()
        except MemoryError:
            print(f"[MEMORY ERROR] at idx={idx}, file={stl_path}")
            raise
        except Exception as e:
            print(f"[ERROR] idx={idx}, file={stl_path}, reason={e}")
            # you can either return a dummy tensor or re-raise:
            raise
