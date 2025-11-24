import torch

from train_lib.train_engine import Engine, Model_wrapper
from train_lib.data import CARLA_Data
from torch import nn
import matplotlib.pyplot as plt
from common.utils import get_bev_semantic_rgb
import numpy as np
import torch.nn.functional as F
import cv2

class LSS_wrapper(Model_wrapper):
  def __init__(self, model, config, device):
    super().__init__(model, config, device)

    if self.config.use_label_smoothing:
      label_smoothing = self.config.label_smoothing_alpha
    else:
      label_smoothing = 0.0
    self.bev_semantic_weights = torch.tensor(self.config.bev_semantic_weights).to(self.device)
    self.compute_loss = nn.CrossEntropyLoss(weight=self.bev_semantic_weights,
                                            label_smoothing=label_smoothing,
                                            ignore_index=-1)
    
    self.last_imgs = None
    self.last_intrins = None
    self.last_rots = None
    self.last_trans = None
    self.last_post_rots = None
    self.last_post_trans = None
    self.last_pred_bev_semantics = None
    self.last_label_bev_semantics = None


  def load_data_compute_loss(self, data):
    imgs        = data["rgb_multi_cam"].to(self.device, dtype=torch.float32)
    intrins     = data["intrins"].to(self.device, dtype=torch.float32)
    rots        = data["rots"].to(self.device, dtype=torch.float32)
    trans       = data["trans"].to(self.device, dtype=torch.float32)
    post_rots   = data["post_rots"].to(self.device, dtype=torch.float32)
    post_trans  = data["post_trans"].to(self.device, dtype=torch.float32)

    pred_bev_semantics = self.model(imgs, rots, trans, intrins, post_rots, post_trans)

    label_bev_semantics = data["bev_semantic"].to(self.device, dtype=torch.long)

    loss = self.compute_loss(pred_bev_semantics, label_bev_semantics)

    loss_individual = {} #Return empty when model output is single

    self.last_imgs = imgs[-1]
    self.last_intrins = intrins[-1]
    self.last_rots = rots[-1]
    self.last_trans = trans[-1]
    self.last_post_rots = post_rots[-1]
    self.last_post_trans = post_trans[-1]
    self.last_pred_bev_semantics = pred_bev_semantics[-1]
    self.last_label_bev_semantics = label_bev_semantics[-1]

    return pred_bev_semantics, label_bev_semantics, loss, loss_individual
  
  def plot_model_out(self):
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    # ---- 予測/ラベルBEVをRGB化 ----
    #pred_bev_semantics = torch.argmax(self.last_pred_bev_semantics, dim=0).detach().cpu().numpy()
    pred_bev_semantics = F.softmax(self.last_pred_bev_semantics, dim=0).detach().cpu().numpy()
    pred_bev_semantics_rgb = get_bev_semantic_rgb(pred_bev_semantics, self.config.bev_classes_list)

    #label_bev_semantics = self.last_label_bev_semantics.detach().cpu().numpy()
    label_bev_semantics = F.one_hot(self.last_label_bev_semantics, num_classes=self.config.outC)
    label_bev_semantics = label_bev_semantics.permute(2, 0, 1).float().detach().cpu().numpy()
    label_bev_semantics_rgb = get_bev_semantic_rgb(label_bev_semantics, self.config.bev_classes_list)

    # ---- 自車矩形を描画 ----
    h, w, _ = pred_bev_semantics_rgb.shape
    center_x, center_y = w // 2, h // 2
    # lincoln.mkz_2017 size: 4.9m × 2.1m
    px_per_meter = 1 / self.config.grid_conf['xbound'][2]  # 4 px/m
    veh_length_px = int(4.9 * px_per_meter)   # ≈20 px
    veh_width_px  = int(2.1 * px_per_meter)   # ≈8 px

    half_L = veh_length_px // 2
    half_W = veh_width_px // 2

    top_left     = (center_x - half_W, center_y - half_L)
    bottom_right = (center_x + half_W, center_y + half_L)

    color_bgr = (0, 255, 255)

    pred_bev_semantics_rgb = cv2.rectangle(pred_bev_semantics_rgb.copy(), top_left, bottom_right, color_bgr, thickness=-1)
    label_bev_semantics_rgb = cv2.rectangle(label_bev_semantics_rgb.copy(), top_left, bottom_right, color_bgr, thickness=-1)

    # ---- 6カメラを横長に合成 ----
    img = self.last_imgs.detach().cpu().numpy()    # [N,3,H,W] (BGR/標準化想定)
    mean = np.array(self.config.rgb_mean)
    std  = np.array(self.config.rgb_std)
    img = img.transpose(0, 2, 3, 1)                # [N,H,W,C]
    img_rgb = (img * std + mean) * 255
    img_rgb = img_rgb[..., ::-1].clip(0, 255).astype(np.uint8)  # BGR→RGB

    n, H, W, C = img_rgb.shape
    canvas = np.full((2*H, 3*W, C), 0, dtype=np.uint8)

    # 固定配置（あなたの元コードと同じレイアウト）
    placements = {
        0: (0*H, 1*W, 1*H, 2*W),  # FRONT
        1: (1*H, 1*W, 2*H, 2*W),  # BACK
        2: (1*H, 2*W, 2*H, 3*W),  # BACK_LEFT
        3: (0*H, 0*W, 1*H, 1*W),  # FRONT_LEFT
        4: (0*H, 2*W, 1*H, 3*W),  # FRONT_RIGHT
        5: (1*H, 0*W, 2*H, 1*W),  # BACK_RIGHT
    }
    for idx in range(min(n, 6)):
        y0, x0, y1, x1 = placements[idx]
        canvas[y0:y1, x0:x1] = img_rgb[idx]

    # ---- 図の作成（1ウィンドウに集約） ----
    if not hasattr(self, "_fig") or self._fig is None:
        self._fig = plt.figure(num="Model Output", constrained_layout=True)
        gs = self._fig.add_gridspec(
            nrows=2, ncols=2,
            width_ratios=[1, 1],
            height_ratios=[1, 0.75],  # 下段はやや低めに
        )
        self._ax_pred  = self._fig.add_subplot(gs[0, 0])
        self._ax_label = self._fig.add_subplot(gs[0, 1])
        self._ax_rgb   = self._fig.add_subplot(gs[1, :])

        # 上段は“正方形”を維持
        for ax in (self._ax_pred, self._ax_label):
            ax.set_aspect('equal', adjustable='box')

    # ---- 描画更新 ----
    self._ax_pred.clear()
    self._ax_pred.imshow(pred_bev_semantics_rgb)
    self._ax_pred.set_title("Pred")
    self._ax_pred.axis("off")

    self._ax_label.clear()
    self._ax_label.imshow(label_bev_semantics_rgb)
    self._ax_label.set_title("Label")
    self._ax_label.axis("off")

    self._ax_rgb.clear()
    self._ax_rgb.imshow(canvas)
    self._ax_rgb.set_title("Surround Cameras")
    self._ax_rgb.axis("off")

    self._fig.canvas.draw_idle()
    plt.pause(0.02)