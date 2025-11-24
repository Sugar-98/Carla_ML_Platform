from common.config import GlobalConfig
import numpy as np

class config_LSS(GlobalConfig):
  def __init__(self):
    super().__init__()
    self.use_PilotNet = False
    self.use_CIL = False

    #model parameters------------------------------------------
    xbound=[-32.0, 32.0, 0.25]
    ybound=[-32.0, 32.0, 0.25]
    zbound=[-10.0, 10.0, 20.0]
    dbound=[4.0, 32.0, 0.5]
    
    self.grid_conf = {
      'xbound': xbound,
      'ybound': ybound,
      'zbound': zbound,
      'dbound': dbound,
    }

    self.outC = 11
    
    #Training parameters------------------------------------------
    self.use_post_augment = True
    self.use_bev_post_augment = True

    eps = 1e-6
    self.ignore_class = [2, 5, 6, 7, 8, 10]
    hist = np.array([1.0510e+08, 1.5803e+07, 9.9302e+06, 8.3205e+06, 2.4915e+06,
                    6.2880e+03, 2.1756e+04, 6.2400e+02, 3.5760e+04, 2.8545e+06, 5.4320e+03])
    # --- 無視クラス以外のみで正規化 ---
    mask = np.ones_like(hist, dtype=bool)
    mask[self.ignore_class] = False
    p = hist[mask] / (hist[mask].sum() + eps)
    # --- 無視クラス以外のみに対して重み計算 ---
    median_p = np.median(p[p > 0])
    w_valid = median_p / (p + eps)
    w_valid = np.clip(w_valid, a_min=None, a_max=3.0)
    w_valid = w_valid / w_valid.mean()
    # --- 元の配列に戻す ---
    w = np.zeros_like(hist, dtype=np.float32)
    w[mask] = w_valid
    w[self.ignore_class] = 0.0  # 念のため明示的に0代入
    # --- 背景の下限 ---
    w[0] = max(w[0].item(), 0.5)
    self.bev_semantic_weights = w.tolist()