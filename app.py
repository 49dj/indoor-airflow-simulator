# app.py — 用户友好型界面版本
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps

from airflow.simulation import (
    simulate_field,
    plot_heatmap_with_suggestions,
    suggest_supply_exhaust,
)
from airflow.geometry import (
    rasterize_walls_mask,
    wall_mask_from_floorplan_image_bytes,
)

# ========== 小工具 ==========
def objects_to_df(json_data: Dict) -> pd.DataFrame:
    """Fabric.js 对象 → DataFrame，便于解析"""
    if not json_data or "objects" not in json_data or not json_data["objects"]:
        return pd.DataFrame()
    rows = []
    for obj in json_data["objects"]:
        rows.append({
            "type": obj.get("type",""),
            "left": float(obj.get("left",0.0)),
            "top": float(obj.get("top",0.0)),
            "width": float(obj.get("width",0.0)),
            "height": float(obj.get("height",0.0)),
            "radius": float(obj.get("radius",0.0)),
            "fill": str(obj.get("fill","")).lower(),
            "stroke": str(obj.get("stroke","")).lower(),
            "strokeWidth": float(obj.get("strokeWidth",1.0)),
            "label": obj.get("name","") or obj.get("label",""),
        })
    return pd.DataFrame(rows)

def center_xy(row: pd.Series) -> Tuple[float,float]:
    """取对象中心（circle 用其 left/top 作为中心——与 st_canvas 的输出兼容）"""
    if str(row.get("type","")) == "circle":
        return float(row["left"]), float(row["top"])
    return float(row["left"]) + float(row["width"])/2.0, float(row["top"]) + float(row["height"])/2.0

def _mask_to_rle_rows(mask: np.ndarray):
    """布尔墙体掩膜 → 行程式 RLE（每行 y: [[x_start,x_end], ...]）"""
    H, W = mask.shape
    rows = []; total = 0
    for y in range(H):
        runs = []
        x = 0
        row = mask[y]
        while x < W:
            if row[x]:
                xs = x
                while x + 1 < W and row[x + 1]:
                    x += 1
                xe = x
                runs.append([int(xs), int(xe)])
                total += (xe - xs + 1)
            x += 1
        rows.append({"y": int(y), "runs": runs})
    return rows, int(total)

def _build_wall_json(mask_to_dump: np.ndarray, *,
                     canvas_w: int, canvas_h: int,
                     grid_w: int, grid_h: int,
                     source_mode: str,
                     floorplan_name: str | None,
                     auto_params: dict | None,
                     manual_params: dict | None):
    rows, total = _mask_to_rle_rows(mask_to_dump)
    payload = {
        "version": "1.0",
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "canvas": {"width_px": int(canvas_w), "height_px": int(canvas_h)},
        "grid": {
            "width": int(grid_w), "height": int(grid_h),
            "mapping_note": "grid x in [0..W-1], y in [0..H-1]; pixel ≈ (x/(W-1)*canvas_w, y/(H-1)*canvas_h)"
        },
        "source": {
            "mode": source_mode,  # "final" / "auto_only" / "manual_only"
            "floorplan_filename": floorplan_name,
            "auto_params": auto_params,
            "manual_params": manual_params
        },
        "encoding": {
            "type": "rle_rows",
            "run_interval": "inclusive [x_start, x_end]",
            "row_field": "y",
            "runs_field": "runs"
        },
        "data": rows,
        "totals": {
            "wall_cells": total,
            "ratio": round(total / max(1, grid_w * grid_h), 6)
        }
    }
    return payload


# ========== 页面基础信息 ==========
st.set_page_config(
    page_title="Indoor thermal / airflow prototype",
    layout="wide"
)

st.title("室内热场与气流探索原型")
st.info("数值探索原型：未经实验校准或网格收敛验证；温度和速度仅供定性探索。示例应使用虚构布局。")
st.caption("把平面图当作底图，自动识别**墙体**（挡风/部分阻热），在其上放置热/冷源与送回风点，快速预览室内的热力图与风场。")

with st.expander("📘 快速上手（Strongly Recommended）", expanded=True):
    st.markdown("""
**Step 1｜上传底图（PNG/JPG）**
- 推荐白底黑墙线的虚构示例平面图；如是 CAD，请先“打印/导出为 PNG”。
- 勾选“反转”处理黑底白墙的图。

**Step 2｜墙体识别（自动 ∪ 手绘）**
- 侧栏调节阈值/闭运算/加厚，观察“墙体识别预览”。
- 如需补门洞/新隔断，用“手绘线/墙”，系统会与自动结果**取并集**。

**Step 3｜放置对象**
- **矩形**=热源（正功率）；**圆形**=冷源（负功率）。
- **绿色点**=送风；**紫色点**=回风。
- 送/回风点靠近某个边界时，会在最近的边界开“入口/出口”段。

**Step 4｜仿真与导出**
- 点击画布后方区域触发更新，右侧将显示热力图与风矢量。
- 在“导出墙体识别结果（JSON）”处，可导出**最终/仅自动/仅手绘**的墙体数据（RLE 编码、小而清晰）。
    """)

st.divider()

# ========== 侧栏：设置 ==========
st.sidebar.header("① 画布大小")
CANVAS_W = st.sidebar.number_input("画布宽度 (px)", 600, 2400, 1000, step=50, help="显示与识别均在此尺寸下进行；不会影响仿真网格分辨率。")
CANVAS_H = st.sidebar.number_input("画布高度 (px)", 400, 1600, 700, step=50)

st.sidebar.markdown("---")
st.sidebar.header("② 底图 / 自动识别墙体")
bg_file = st.sidebar.file_uploader("上传平面图（PNG/JPG）", type=["png","jpg","jpeg"], help="建议白底黑墙线；CAD 请先导出为 PNG/JPG。")
auto_mode = st.sidebar.selectbox("识别方式", ["深色=墙（推荐）", "灰度阈值"], index=0)
invert_bg = st.sidebar.checkbox("反转（黑底白墙时勾选）", value=False)
if auto_mode == "深色=墙（推荐）":
    DARK_T = st.sidebar.slider("深色阈值（0黑→255白）", 0, 255, 80, help="RGB 同时低于阈值则视作“墙”。")
    CLOSE_PX = st.sidebar.slider("闭运算：修补细缝 (px)", 0, 12, 2)
    DILATE_PX= st.sidebar.slider("加厚：避免降采样漏墙 (px)", 0, 24, 6)
else:
    GRAY_T = st.sidebar.slider("灰度阈值（下方=墙）", 0, 255, 180, help="灰度值低于阈值视作“墙”。")
    CLOSE_PX = st.sidebar.slider("闭运算：修补细缝 (px)", 0, 12, 2)
    DILATE_PX= st.sidebar.slider("加厚：避免降采样漏墙 (px)", 0, 24, 6)

COMBINE_WALLS = st.sidebar.checkbox("与手动画墙合并（取并集，推荐）", value=True, help="勾选后：最终墙体 = 自动识别 ∪ 手绘。")

st.sidebar.markdown("---")
st.sidebar.header("③ 绘图工具")
tool = st.sidebar.selectbox(
    "当前工具",
    ["矩形（热源）", "圆形（冷源）", "点（送/回风）", "手绘线/墙", "选择/移动"],
    index=0, help="**矩形**=热源；**圆形**=冷源；**点**=送/回风（颜色识别）。"
)
mode_map = {
    "矩形（热源）": "rect",
    "圆形（冷源）": "circle",
    "点（送/回风）": "point",
    "手绘线/墙": "freedraw",
    "选择/移动": "transform",
}
canvas_mode = mode_map[tool]
stroke_width = st.sidebar.slider("描边宽度", 1, 12, 4)
stroke_color = st.sidebar.color_picker("画笔颜色（线/点）", "#222222")
fill_color   = st.sidebar.color_picker("填充颜色（矩形/圆）", "#ffcccc")
point_radius = st.sidebar.slider("点大小（仅点工具）", 2, 20, 8)

st.sidebar.markdown("---")
st.sidebar.header("④ 热/冷源功率")
use_area_mapping = st.sidebar.checkbox("按面积映射功率（更直观）", value=True)
rect_W_per_px2   = st.sidebar.number_input("矩形功率密度 W/px²（正）", value=0.8, step=0.1)
circ_W_per_px2   = st.sidebar.number_input("圆形功率密度 W/px²（负）", value=-0.8, step=0.1)
default_rect_W   = st.sidebar.number_input("矩形固定功率 W（备用）", value=500.0, step=50.0)
default_circ_W   = st.sidebar.number_input("圆形固定功率 W（备用）", value=-500.0, step=50.0)

st.sidebar.markdown("---")
st.sidebar.header("⑤ 送/回风设置")
SUPPLY_COLOR  = st.sidebar.color_picker("送风颜色（识别）", "#00cc66")
EXHAUST_COLOR = st.sidebar.color_picker("回风颜色（识别）", "#9933ff")
DEFAULT_STRENGTH = st.sidebar.number_input("送/回风默认强度", 0.1, 10.0, 1.0, step=0.1, help="越大代表入口/出口段越宽，送排量越强。")
col_sc, col_ec = st.sidebar.columns(2)
if col_sc.button("画笔→送风色"): stroke_color = SUPPLY_COLOR
if col_ec.button("画笔→回风色"): stroke_color = EXHAUST_COLOR

st.sidebar.markdown("---")
st.sidebar.header("⑥ 手绘墙参数")
WALL_COLOR = st.sidebar.color_picker("墙体描边颜色（识别）", "#444444", help="与手绘线的颜色匹配即识别为墙。")
WALL_DILATE = st.sidebar.slider("墙体膨胀（网格格数）", 0, 5, 1, help="让手绘墙在网格里更“厚实”。")

st.sidebar.markdown("---")
st.sidebar.header("⑦ 网格与可视化")
GRID_W = st.sidebar.slider("网格列数（W）", 20, 240, 120)
GRID_H = st.sidebar.slider("网格行数（H）", 15, 200, 80)
SIGMA_PX = st.sidebar.slider("热源扩展 σ（像素）", 10, 300, 120, step=10, help="控制热/冷源的影响半径。")
ARROW_SCALE = st.sidebar.slider("风矢量箭头长度（越大越短）", 40, 200, 120)

st.sidebar.markdown("---")
st.sidebar.header("⑧ 建议点（可选）")
K_SUPPLY  = st.sidebar.slider("建议送风点个数", 0, 10, 3)
K_EXHAUST = st.sidebar.slider("建议回风点个数", 0, 10, 2)

# ========== 背景图与自动识别 ==========
bg_image: Optional[Image.Image] = None
auto_wall_mask: Optional[np.ndarray] = None

if bg_file is not None:
    # 铺到底图（显示用）
    bg_image = Image.open(bg_file)
    bg_image = ImageOps.exif_transpose(bg_image).convert("RGB")
    bg_image = bg_image.resize((CANVAS_W, CANVAS_H), Image.BILINEAR)

    # 自动识别墙体（用于仿真）
    raw_bytes = bg_file.getvalue()
    if auto_mode == "深色=墙（推荐）":
        auto_wall_mask = wall_mask_from_floorplan_image_bytes(
            raw_bytes, CANVAS_W, CANVAS_H, GRID_W, GRID_H,
            mode="dark", dark_thresh=int(DARK_T),
            dilate_px=int(DILATE_PX), close_px=int(CLOSE_PX),
            invert=bool(invert_bg)
        )
    else:
        auto_wall_mask = wall_mask_from_floorplan_image_bytes(
            raw_bytes, CANVAS_W, CANVAS_H, GRID_W, GRID_H,
            mode="gray", gray_thresh=int(GRAY_T),
            dilate_px=int(DILATE_PX), close_px=int(CLOSE_PX),
            invert=bool(invert_bg)
        )

# ========== 画布（含不兼容回退） ==========
st.subheader("① 画布与对象")
st.caption("在底图上直接拖拽绘制：**矩形=热源、圆形=冷源、绿色点=送风、紫色点=回风、灰色线/块=墙（手绘）**。在画布空白处单击可刷新。")

def _draw_canvas_with_bg(bg_img):
    common_kwargs = dict(
        fill_color=fill_color,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        width=CANVAS_W,
        height=CANVAS_H,
        drawing_mode=canvas_mode,
        point_display_radius=point_radius,
        display_toolbar=True,
        update_streamlit=True,
        key="canvas_v6",
    )
    try:
        return st_canvas(background_image=bg_img, **common_kwargs)
    except TypeError as e:
        if "_image_to_url" in str(e):
            st.warning("当前环境与画布组件的“底图铺设”不完全兼容，已自动回退为纯白底。**不影响墙体自动识别与仿真**。")
            return st_canvas(background_image=None, background_color="#FFFFFF", **common_kwargs)
        raise

canvas_res = _draw_canvas_with_bg(bg_image)
json_data = canvas_res.json_data if canvas_res is not None else None
objects_df = objects_to_df(json_data)

if not objects_df.empty:
    with st.expander("调试视图：已识别的对象属性", expanded=False):
        st.dataframe(objects_df, use_container_width=True, hide_index=True)

# ========== 解析对象 ==========
def build_sources_from_objects(df: pd.DataFrame) -> List[Dict]:
    if df is None or df.empty: return []
    out = []
    for _, r in df.iterrows():
        t = str(r["type"])
        if t not in ("rect", "circle"): continue
        x, y = center_xy(r)
        if use_area_mapping:
            if t == "rect":
                area = max(1.0, float(r["width"]) * float(r["height"]))
                load = rect_W_per_px2 * area
            else:
                area = max(1.0, np.pi * (float(r["radius"])**2))
                load = circ_W_per_px2 * area
        else:
            load = default_rect_W if t == "rect" else default_circ_W
        out.append({"x": float(x), "y": float(y), "load": float(load)})
    return out

def extract_supply_exhaust(df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    if df is None or df.empty: return [], []
    supply, exhaust = [], []
    for _, r in df.iterrows():
        stroke = str(r.get("stroke","")).lower()
        if not stroke: continue
        if stroke.startswith(SUPPLY_COLOR.lower()):
            x, y = center_xy(r); supply.append({"x": float(x), "y": float(y), "strength": DEFAULT_STRENGTH})
        elif stroke.startswith(EXHAUST_COLOR.lower()):
            x, y = center_xy(r); exhaust.append({"x": float(x), "y": float(y), "strength": DEFAULT_STRENGTH})
    return supply, exhaust

sources = build_sources_from_objects(objects_df)
supply, exhaust = extract_supply_exhaust(objects_df)

# 手绘墙 → 栅格
manual_wall_mask = rasterize_walls_mask(
    objects=objects_df.to_dict("records"),
    canvas_w=CANVAS_W, canvas_h=CANVAS_H,
    grid_w=GRID_W, grid_h=GRID_H,
    wall_color=WALL_COLOR.lower(),
    thickness_px=int(stroke_width + 2*WALL_DILATE),
    label_keywords=["墙","wall"]
)

# 合并：最终墙体
if auto_wall_mask is not None and COMBINE_WALLS:
    wall_mask = np.logical_or(auto_wall_mask, manual_wall_mask)
elif auto_wall_mask is not None:
    wall_mask = auto_wall_mask
else:
    wall_mask = manual_wall_mask

with st.expander("② 墙体识别预览（栅格）", expanded=False):
    st.caption("白=空气，黑=墙。此图用于检查识别效果（与仿真网格对齐）。")
    if auto_wall_mask is not None:
        st.image((~auto_wall_mask).astype(np.uint8)*255, clamp=True, caption="自动识别（缩放到网格）")
    st.image((~wall_mask).astype(np.uint8)*255, clamp=True, caption="最终用于仿真的墙体（自动 ∪ 手绘）")

# ========== 导出：墙体 JSON ==========
if (auto_wall_mask is not None) or np.any(manual_wall_mask) or np.any(wall_mask):
    st.subheader("③ 导出墙体识别结果（JSON）")
    st.caption("导出的 JSON 记录了画布/网格信息、识别来源与参数、以及“按行的墙段 RLE 编码”。后续可在你的工程里**精准重建墙体**。")

    export_options = []
    option_labels = []
    if auto_wall_mask is not None:
        export_options.append(("auto_only", auto_wall_mask))
        option_labels.append("仅自动识别")
    if np.any(manual_wall_mask):
        export_options.append(("manual_only", manual_wall_mask))
        option_labels.append("仅手绘")
    export_options.append(("final", wall_mask))
    option_labels.append("最终（自动 ∪ 手绘｜推荐）")

    idx_default = len(option_labels) - 1
    choice_label = st.radio("选择导出对象：", option_labels, index=idx_default, horizontal=True)
    choice_key, mask_chosen = export_options[option_labels.index(choice_label)]

    # 元数据
    if auto_wall_mask is not None:
        if auto_mode == "深色=墙（推荐）":
            auto_params = {"mode":"dark","dark_thresh":int(DARK_T),"close_px":int(CLOSE_PX),"dilate_px":int(DILATE_PX),"invert":bool(invert_bg)}
        else:
            auto_params = {"mode":"gray","gray_thresh":int(GRAY_T),"close_px":int(CLOSE_PX),"dilate_px":int(DILATE_PX),"invert":bool(invert_bg)}
    else:
        auto_params = None

    manual_params = {"wall_color":WALL_COLOR.lower(),"thickness_px":int(stroke_width + 2*WALL_DILATE),"label_keywords":["墙","wall"]} \
                    if np.any(manual_wall_mask) else None

    # 文件名
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = {"final":"final","auto_only":"auto","manual_only":"manual"}[choice_key]
    default_name = f"walls_{base}_{GRID_W}x{GRID_H}_{ts}.json"
    filename = st.text_input("导出文件名", value=default_name)

    floorplan_name = (bg_file.name if bg_file is not None else None)
    json_obj = _build_wall_json(
        mask_chosen,
        canvas_w=CANVAS_W, canvas_h=CANVAS_H,
        grid_w=GRID_W, grid_h=GRID_H,
        source_mode=choice_key,
        floorplan_name=floorplan_name,
        auto_params=auto_params,
        manual_params=manual_params
    )
    json_text = json.dumps(json_obj, ensure_ascii=False, indent=2)

    st.download_button(
        "⬇️ 下载墙体 JSON",
        data=json_text,
        file_name=filename,
        mime="application/json",
        help="下载包含网格、来源参数与墙体 RLE 的 JSON 文件"
    )

# ========== 仿真与可视化 ==========
if not objects_df.empty or (auto_wall_mask is not None):
    st.subheader("④ 仿真与可视化")
    st.caption("点击画布空白处刷新。墙体**完全挡风**，墙内仅导热（较慢）。热源只在空气网格注入。")

    heat, u, v = simulate_field(
        sources=sources,
        supply=supply,
        exhaust=exhaust,
        canvas_w=CANVAS_W, canvas_h=CANVAS_H,
        grid_w=GRID_W, grid_h=GRID_H,
        sigma_px=SIGMA_PX,
        backend="ffd",
        wall_mask=wall_mask,
        calib=None,
        flow_iters=320,
        flow_scale=1.2,
        rho_air=1.2, mu_air=1.8e-5, k_air=0.026, cp_air=1005.0,
        beta=3e-3, g_mag=9.81,
        k_solid=0.005, rho_solid=2200.0, cp_solid=880.0,
        fan_sigma=8, tin_mode="auto", tin_alpha=0.5, tin_fixed=-5.0,
    )

    sup_sug, exh_sug = suggest_supply_exhaust(
        heatmap=heat, k_supply=K_SUPPLY, k_exhaust=K_EXHAUST, min_separation=8
    )

    fig = plot_heatmap_with_suggestions(
        heatmap=heat,
        sources=sources,
        u=u, v=v,
        supply=supply, exhaust=exhaust,
        supply_suggest=sup_sug, exhaust_suggest=exh_sug,
        wall_mask=wall_mask,
        arrow_scale=ARROW_SCALE,
    )
    st.pyplot(fig, use_container_width=True)

    # 建议点导出（可选）
    H, W = heat.shape
    def norm_list(lst, typ):
        return [{"type": typ, "x_norm": round(x/(W-1), 4), "y_norm": round(y/(H-1), 4)} for (x,y) in lst]
    rows = norm_list(sup_sug, "supply") + norm_list(exh_sug, "exhaust")
    if rows:
        df_pts = pd.DataFrame(rows)
        with st.expander("建议送/回风点（标准化坐标）", expanded=False):
            st.dataframe(df_pts, use_container_width=True, hide_index=True)
            csv = df_pts.to_csv(index=False).encode("utf-8")
            st.download_button("下载建议点 CSV", data=csv, file_name="suggest_points_norm.csv", mime="text/csv")

else:
    st.info("请先上传平面图或在画布上绘制对象。完成墙体识别后，放置热/冷源与送回风点，再下滑查看仿真结果与导出选项。")
