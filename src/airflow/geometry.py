# process_layout.py
# - rasterize_walls_mask: 手绘/标注为“墙”的对象 -> 栅格墙掩膜(True=墙)
# - wall_mask_from_floorplan_image_bytes: 从底图 PNG/JPG 自动识别墙 -> 栅格墙掩膜
from __future__ import annotations
from typing import List, Dict, Sequence, Optional, Tuple
import io
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from PIL import Image, ImageOps


# ---------- 基础工具 ----------
def _to_df(obj) -> pd.DataFrame:
    if obj is None: return pd.DataFrame()
    if isinstance(obj, pd.DataFrame): return obj.copy()
    try: return pd.DataFrame(obj)
    except Exception: return pd.DataFrame()

def _safe_float(x, default=0.0):
    try: return float(x)
    except Exception: return float(default)

def _center_xy(row: pd.Series):
    t = str(row.get("type", "")).lower()
    if t == "circle":
        return _safe_float(row.get("left", 0.0)), _safe_float(row.get("top", 0.0))
    # rect/path：用中心
    return _safe_float(row.get("left", 0.0)) + _safe_float(row.get("width", 0.0))/2.0, \
           _safe_float(row.get("top", 0.0))  + _safe_float(row.get("height",0.0))/2.0

def _pix_to_grid(x, y, canvas_w, canvas_h, grid_w, grid_h):
    gx = int(np.clip(round(x / max(canvas_w,1e-6) * (grid_w-1)), 0, grid_w-1))
    gy = int(np.clip(round(y / max(canvas_h,1e-6) * (grid_h-1)), 0, grid_h-1))
    return gx, gy

def _pix_to_grid_box(left, top, width, height, canvas_w, canvas_h, grid_w, grid_h):
    gx0, gy0 = _pix_to_grid(left,           top,            canvas_w, canvas_h, grid_w, grid_h)
    gx1, gy1 = _pix_to_grid(left+width,     top+height,     canvas_w, canvas_h, grid_w, grid_h)
    x0, x1 = sorted([gx0, gx1]); y0, y1 = sorted([gy0, gy1])
    return x0, y0, x1, y1


# ---------- 画布对象 -> 墙体掩膜（保持你现有的手画墙能力） ----------
def rasterize_walls_mask(
    objects: Sequence[Dict],
    canvas_w: int, canvas_h: int,
    grid_w: int, grid_h: int,
    wall_color: str = "#444444",
    thickness_px: int = 4,
    label_keywords: Sequence[str] = ("墙", "wall"),
) -> np.ndarray:
    df = _to_df(objects)
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    if df.empty:
        return mask

    wc = (wall_color or "").lower()
    kw = tuple([str(k).lower() for k in (label_keywords or [])])

    for _, row in df.iterrows():
        typ   = str(row.get("type","")).lower()
        stroke= str(row.get("stroke","")).lower()
        label = str(row.get("label","")).lower()

        is_wall = False
        if wc and stroke.startswith(wc):
            is_wall = True
        if not is_wall and kw:
            if any(k in label for k in kw):
                is_wall = True
        if not is_wall:
            continue

        if typ == "rect":
            x0,y0,x1,y1 = _pix_to_grid_box(
                _safe_float(row.get("left",0.0)),
                _safe_float(row.get("top",0.0)),
                _safe_float(row.get("width",0.0)),
                _safe_float(row.get("height",0.0)),
                canvas_w, canvas_h, grid_w, grid_h
            )
            mask[y0:y1+1, x0:x1+1] = True

        elif typ in ("path", "freedraw", "line"):
            # 手绘线：用其 bbox + thickness 近似
            w = _safe_float(row.get("width",0.0))  + thickness_px
            h = _safe_float(row.get("height",0.0)) + thickness_px
            l = _safe_float(row.get("left",0.0))   - thickness_px/2
            t = _safe_float(row.get("top",0.0))    - thickness_px/2
            x0,y0,x1,y1 = _pix_to_grid_box(l, t, w, h, canvas_w, canvas_h, grid_w, grid_h)
            mask[y0:y1+1, x0:x1+1] = True

        elif typ == "circle":
            # 圆形墙体（很少见），以包围盒处理
            r = _safe_float(row.get("radius",6.0))
            l = _safe_float(row.get("left",0.0)) - r
            t = _safe_float(row.get("top",0.0))  - r
            w = 2*r; h = 2*r
            x0,y0,x1,y1 = _pix_to_grid_box(l, t, w, h, canvas_w, canvas_h, grid_w, grid_h)
            mask[y0:y1+1, x0:x1+1] = True

    # 让墙“厚实”一点（网格层面 1~2 格）
    if thickness_px > 0:
        s = np.ones((3,3), dtype=bool)
        for _ in range(max(1, int(round(thickness_px/4)))):
            mask = ndi.binary_dilation(mask, structure=s)

    return mask


# ---------- 位图底图 -> 自动识别墙体 ----------
def wall_mask_from_floorplan_image_bytes(
    img_bytes: bytes,
    canvas_w: int, canvas_h: int,
    grid_w: int, grid_h: int,
    mode: str = "dark",            # "dark"：RGB都较暗视为墙；"gray": 灰度阈值
    dark_thresh: int = 80,         # 0黑—255白，越小越“黑”
    gray_thresh: int = 180,        # 0黑—255白，阈值以下视为墙
    dilate_px: int = 6,            # 画布像素的加厚
    close_px: int  = 2,            # 闭运算消除小缺口
    invert: bool = False           # 如果图纸是白墙黑底，可反转
) -> np.ndarray:
    """
    把 PNG/JPG 平面图转成栅格墙体掩膜(True=墙)：
    - 假设“墙线较深色，背景较浅色”
    - 形态学闭运算 + 膨胀，让细墙线在降采样后仍被捕获
    """
    # 读取 & 适配画布尺寸
    im = Image.open(io.BytesIO(img_bytes))
    im = ImageOps.exif_transpose(im).convert("RGB")
    im = im.resize((canvas_w, canvas_h), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.uint8)

    if invert:
        arr = 255 - arr

    R = arr[...,0].astype(np.int16)
    G = arr[...,1].astype(np.int16)
    B = arr[...,2].astype(np.int16)
    gray = (0.299*R + 0.587*G + 0.114*B).astype(np.uint8)

    if mode == "dark":
        bw = (R < dark_thresh) & (G < dark_thresh) & (B < dark_thresh)
    else:  # "gray"
        bw = (gray < gray_thresh)

    # 形态学清理：闭运算填小孔、然后膨胀加厚
    if close_px > 0:
        se = np.ones((close_px, close_px), dtype=bool)
        bw = ndi.binary_closing(bw, structure=se)
    if dilate_px > 0:
        se = np.ones((dilate_px, dilate_px), dtype=bool)
        bw = ndi.binary_dilation(bw, structure=se)

    # 为避免降采样漏检，在采样前再做一次按网格尺度的稳健加厚
    sx = max(1, int(np.ceil(canvas_w / max(grid_w,1))))
    sy = max(1, int(np.ceil(canvas_h / max(grid_h,1))))
    se2 = np.ones((max(1, sy//2), max(1, sx//2)), dtype=bool)
    bw = ndi.binary_dilation(bw, structure=se2)

    # 由画布到仿真网格：最近邻采样（足够粗稳健）
    ys = np.linspace(0, canvas_h-1, grid_h).astype(int)
    xs = np.linspace(0, canvas_w-1, grid_w).astype(int)
    wall_grid = bw[ys[:,None], xs[None,:]]
    return wall_grid.astype(bool)
