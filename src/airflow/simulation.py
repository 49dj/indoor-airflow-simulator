# simulation.py
# 墙体=固体掩膜：风不可穿透、墙内速度=0；热在墙内仅传导，跨界热通量用谐均值限制（体现“墙传导较慢”）

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
rcParams['axes.unicode_minus'] = False

# ================= 基础工具 =================
def _pix_to_grid(x, y, canvas_w, canvas_h, grid_w, grid_h):
    gx = int(np.clip(round(x / max(canvas_w,1e-6) * (grid_w-1)), 0, grid_w-1))
    gy = int(np.clip(round(y / max(canvas_h,1e-6) * (grid_h-1)), 0, grid_h-1))
    return gx, gy

# --------- （保留：供旧路径调用）画布对象 -> 墙体掩膜 ----------
def build_wall_mask_from_objects(objects_df, canvas_w, canvas_h, grid_w, grid_h, wall_color="#444444", dilate=1):
    solid = np.zeros((grid_h, grid_w), dtype=bool)
    if not objects_df: return solid
    wc = str(wall_color).lower()

    def mark_rect(x0,y0,w,h):
        gx0, gy0 = _pix_to_grid(x0, y0, canvas_w, canvas_h, grid_w, grid_h)
        gx1, gy1 = _pix_to_grid(x0 + w, y0 + h, canvas_w, canvas_h, grid_w, grid_h)
        x_min, x_max = sorted([gx0, gx1]); y_min, y_max = sorted([gy0, gy1])
        solid[y_min:y_max+1, x_min:x_max+1] = True

    def mark_circle(cx, cy, r):
        gx, gy = _pix_to_grid(cx, cy, canvas_w, canvas_h, grid_w, grid_h)
        rx = max(1, int(round(r / max(canvas_w,1e-6) * (grid_w-1))))
        ry = max(1, int(round(r / max(canvas_h,1e-6) * (grid_h-1))))
        Y, X = np.ogrid[:grid_h, :grid_w]
        mask = ((X - gx)**2 / max(rx*rx,1)) + ((Y - gy)**2 / max(ry*ry,1)) <= 1.0
        solid[mask] = True

    for r in objects_df:
        stroke = str(r.get("stroke","")).lower()
        label  = str(r.get("label","")).lower()
        t      = str(r.get("type",""))
        is_wall = (wc and stroke.startswith(wc)) or ("wall" in label) or ("墙" in label)
        if not is_wall:
            continue
        if t == "rect":
            mark_rect(r["left"], r["top"], r["width"], r["height"])
        elif t == "circle":
            mark_circle(r["left"], r["top"], r.get("radius",6))
        else:
            mark_rect(r["left"], r["top"], r["width"], r["height"])

    # 膨胀让墙更“厚”
    for _ in range(max(0,int(dilate))):
        s = solid.copy()
        s[1:, :] |= solid[:-1, :]
        s[:-1,:] |= solid[1: , :]
        s[:, 1:] |= solid[:, :-1]
        s[:, :-1]|= solid[:, 1: ]
        solid = s
    return solid

# ================= 建议点位 =================
def _pick_peaks(arr, k, min_sep):
    H, W = arr.shape
    chosen = []
    cand = np.dstack(np.unravel_index(np.argsort(arr.ravel())[::-1], (H, W)))[0]
    for (y, x) in cand:
        ok = True
        for (yy, xx) in chosen:
            if (y-yy)**2 + (x-xx)**2 < min_sep**2:
                ok = False; break
        if ok:
            chosen.append((y, x))
            if len(chosen) >= k: break
    return [(x, y) for (y, x) in chosen]

def suggest_supply_exhaust(heatmap, k_supply=3, k_exhaust=2, min_separation=8):
    sup = _pick_peaks(heatmap, k_supply, min_separation)
    exh = _pick_peaks(-heatmap, k_exhaust, min_separation)
    return sup, exh

# ================= 数值算子（带墙体限制） =================
def _central_diff_x(a): return (np.roll(a, -1, 1) - np.roll(a, 1, 1))*0.5
def _central_diff_y(a): return (np.roll(a, -1, 0) - np.roll(a, 1, 0))*0.5
def _laplace(a):        return (np.roll(a,-1,1)+np.roll(a,1,1)+np.roll(a,-1,0)+np.roll(a,1,0)-4*a)

def _impenetrable_faces(u, v, solid):
    """不可穿透：固体内速度=0；跨墙法向速度=0（左右面影响u，上下面影响v）"""
    u = u.copy(); v = v.copy()
    # 左右相邻若任一为墙，则这条竖直面法向速度=0
    left_wall  = solid | np.roll(solid,  1, axis=1)
    right_wall = solid | np.roll(solid, -1, axis=1)
    u[left_wall]  = 0.0
    u[right_wall] = 0.0
    # 上下相邻若任一为墙，则这条水平面法向速度=0
    up_wall   = solid | np.roll(solid,  1, axis=0)
    down_wall = solid | np.roll(solid, -1, axis=0)
    v[up_wall]   = 0.0
    v[down_wall] = 0.0
    # 固体内速度=0
    u[solid] = 0.0; v[solid] = 0.0
    return u, v

def _advect_upwind_masked(phi, u, v, fluid):
    """一阶迎风，对流只在流体-流体对之间发生；跨墙/固体对流项=0"""
    # x 向
    phi_r = np.roll(phi, -1, 1); phi_l = np.roll(phi, 1, 1)
    fluid_r = np.roll(fluid, -1, 1); fluid_l = np.roll(fluid, 1, 1)
    dphidx_pos = np.where(fluid & fluid_l, phi - phi_l, 0.0)
    dphidx_neg = np.where(fluid & fluid_r, phi_r - phi, 0.0)
    dphidx = np.where(u >= 0, dphidx_pos, dphidx_neg)

    # y 向
    phi_d = np.roll(phi, -1, 0); phi_u = np.roll(phi, 1, 0)
    fluid_d = np.roll(fluid, -1, 0); fluid_u = np.roll(fluid, 1, 0)
    dphidy_pos = np.where(fluid & fluid_u, phi - phi_u, 0.0)
    dphidy_neg = np.where(fluid & fluid_d, phi_d - phi, 0.0)
    dphidy = np.where(v >= 0, dphidy_pos, dphidy_neg)

    return u*dphidx + v*dphidy

def _variable_k_diffusion_harmonic(T, k, rho_cp):
    """∇·(k∇T)：面导热率取谐均值（低导热侧主导），更贴近跨界传热"""
    T_r = np.roll(T, -1, 1); T_l = np.roll(T, 1, 1)
    T_d = np.roll(T, -1, 0); T_u = np.roll(T, 1, 0)

    k_r = 2.0 * k * np.roll(k, -1, 1) / np.maximum(k + np.roll(k, -1, 1), 1e-9)
    k_l = 2.0 * k * np.roll(k,  1, 1) / np.maximum(k + np.roll(k,  1, 1), 1e-9)
    k_d = 2.0 * k * np.roll(k, -1, 0) / np.maximum(k + np.roll(k, -1, 0), 1e-9)
    k_u = 2.0 * k * np.roll(k,  1, 0) / np.maximum(k + np.roll(k,  1, 0), 1e-9)

    diff = (k_r*(T_r - T) - k_l*(T - T_l) + k_d*(T_d - T) - k_u*(T - T_u))
    return diff / np.maximum(rho_cp, 1e-9)

def _project_pressure(u_star, v_star, mask_fluid, outlet_mask, niters=160):
    """压力投影：∇²p = div(u*)；仅在流体区（且非 Dirichlet 出口）迭代"""
    H, W = u_star.shape
    rhs = _central_diff_x(u_star) + _central_diff_y(v_star)
    p = np.zeros_like(u_star)
    can_update = (mask_fluid) & (~outlet_mask)

    for _ in range(int(niters)):
        pn = p.copy()
        lap_nb = (np.roll(p,-1,1)+np.roll(p,1,1)+np.roll(p,-1,0)+np.roll(p,1,0) - rhs)
        pn[can_update] = 0.25 * lap_nb[can_update]
        # Neumann 边界（复制）
        pn[0,:]  = pn[1,:];   pn[-1,:] = pn[-2,:]
        pn[:,0]  = pn[:,1];   pn[:,-1] = pn[:,-1]
        pn[outlet_mask] = 0.0
        p = pn

    u = u_star - _central_diff_x(p)
    v = v_star - _central_diff_y(p)
    return u, v, p

# ================= 入口/出口（兜底，确保总体送→回风趋势） =================
def _build_inout_from_points(supply, exhaust, canvas_w, canvas_h, grid_w, grid_h, strength_scale=1.0):
    H, W = grid_h, grid_w
    inlet_u = np.zeros((H, W)); inlet_v = np.zeros((H, W))
    inlet_mask = np.zeros((H, W), dtype=bool)
    outlet_mask = np.zeros((H, W), dtype=bool)

    def nearest_side(gx, gy):
        d = [gx, (W-1-gx), gy, (H-1-gy)]  # left, right, top, bottom
        return int(np.argmin(d))

    def mark_segment(side, gy, gx, halfw, speed, is_inlet=True):
        if side == 0:  # left
            y0 = max(0, gy-halfw); y1 = min(H-1, gy+halfw)
            for y in range(y0, y1+1):
                if is_inlet:
                    inlet_mask[y,0] = True; inlet_u[y,0] = +speed; inlet_v[y,0] = 0.0
                else:
                    outlet_mask[y,0] = True
        elif side == 1:  # right
            y0 = max(0, gy-halfw); y1 = min(H-1, gy+halfw)
            for y in range(y0, y1+1):
                if is_inlet:
                    inlet_mask[y,W-1] = True; inlet_u[y,W-1] = -speed; inlet_v[y,W-1] = 0.0
                else:
                    outlet_mask[y,W-1] = True
        elif side == 2:  # top
            x0 = max(0, gx-halfw); x1 = min(W-1, gx+halfw)
            for x in range(x0, x1+1):
                if is_inlet:
                    inlet_mask[0,x] = True; inlet_u[0,x] = 0.0; inlet_v[0,x] = +speed
                else:
                    outlet_mask[0,x] = True
        else:           # bottom
            x0 = max(0, gx-halfw); x1 = min(W-1, gx+halfw)
            for x in range(x0, x1+1):
                if is_inlet:
                    inlet_mask[H-1,x] = True; inlet_u[H-1,x] = 0.0; inlet_v[H-1,x] = -speed
                else:
                    outlet_mask[H-1,x] = True

    for s in (supply or []):
        gx, gy = _pix_to_grid(s["x"], s["y"], canvas_w, canvas_h, grid_w, grid_h)
        side = nearest_side(gx, gy)
        w = int(2 + 2*float(s.get("strength",1.0)))
        U = max(0.2, float(s.get("strength",1.0))*strength_scale)
        mark_segment(side, gy, gx, w, U, is_inlet=True)

    for e in (exhaust or []):
        gx, gy = _pix_to_grid(e["x"], e["y"], canvas_w, canvas_h, grid_w, grid_h)
        side = nearest_side(gx, gy)
        w = int(2 + 2*float(e.get("strength",1.0)))
        mark_segment(side, gy, gx, w, speed=0.0, is_inlet=False)

    # 若没有出口，自动在对侧开口
    if not np.any(outlet_mask) and np.any(inlet_mask):
        counts = [np.sum(inlet_mask[:,0]), np.sum(inlet_mask[:,-1]), np.sum(inlet_mask[0,:]), np.sum(inlet_mask[-1,:])]
        side = int(np.argmax(counts))
        side = {0:1,1:0,2:3,3:2}[side]
        if side in (0,1):
            for y in range(0, H, max(1, H//8)):
                outlet_mask[y, 0 if side==0 else -1] = True
        else:
            for x in range(0, W, max(1, W//8)):
                outlet_mask[0 if side==2 else -1, x] = True

    return inlet_mask, inlet_u, inlet_v, outlet_mask

# ================= 热源场（体热源） =================
def _build_qdot_field(sources, canvas_w, canvas_h, grid_w, grid_h, sigma_px=80):
    H, W = grid_h, grid_w
    q = np.zeros((H, W), dtype=float)
    if not sources: return q
    sx = max(1.0, sigma_px / max(canvas_w,1e-6) * W)
    sy = max(1.0, sigma_px / max(canvas_h,1e-6) * H)
    yy, xx = np.mgrid[0:H, 0:W]
    for s in sources:
        gx, gy = _pix_to_grid(s["x"], s["y"], canvas_w, canvas_h, W, H)
        amp = float(s.get("load", 0.0))
        q += amp * np.exp(-((xx - gx)**2)/(2*sx*sx) - ((yy - gy)**2)/(2*sy*sy))
    return q

# ================= N–S + 能量（带墙约束） =================
def _ns_brinkman_with_walls(
    sources, supply, exhaust,
    canvas_w, canvas_h, grid_w, grid_h,
    solid_mask,
    rho_air=1.2, mu_air=1.8e-5, k_air=0.026, cp_air=1005.0,
    beta=3e-3, g_mag=9.81,
    k_solid=0.005, rho_solid=2200.0, cp_solid=880.0,
    sigma_px=100, flow_iters=300, flow_scale=1.2,
    # 兼容 app.py 传参（此版本暂未使用到，但要先接住以防报错）
    fan_sigma=8, tin_mode="auto", tin_alpha=0.5, tin_fixed=-5.0,
):
    H, W = grid_h, grid_w
    fluid = ~solid_mask

    # 物性场：墙体更“保温”时应设 k_solid < k_air
    k = np.where(fluid, k_air, k_solid)
    rho = np.where(fluid, rho_air, rho_solid)
    cp  = np.where(fluid, cp_air,  cp_solid)
    rho_cp = rho * cp
    nu = mu_air / max(rho_air, 1e-9)

    # 入口/出口（兜底保证送→回）
    inlet_mask, inlet_u, inlet_v, outlet_mask = _build_inout_from_points(
        supply, exhaust, canvas_w, canvas_h, grid_w, grid_h, strength_scale=flow_scale
    )

    # 体热源（只在流体区注入，避免“墙内发热”）
    qdot = _build_qdot_field(sources, canvas_w, canvas_h, grid_w, grid_h, sigma_px=sigma_px)
    qdot *= fluid.astype(float)

    # 初值
    u = np.zeros((H, W), dtype=float)
    v = np.zeros((H, W), dtype=float)
    T = np.zeros((H, W), dtype=float)  # 相对温度（背景=0）

    penalty_alpha = 80.0  # Brinkman 惩罚（固体区使速度快速→0）
    cfl = 0.4; diff_limit = 0.4

    for _ in range(int(flow_iters)):
        umax = max(1e-6, float(np.max(np.abs(u))), float(np.max(np.abs(v))))
        alpha_max = float(np.max(k / np.maximum(rho_cp,1e-9)))
        dt_cfl  = cfl / max(umax, 1e-3)
        dt_diff = diff_limit / max(alpha_max + nu, 1e-6)
        dt = max(1e-3, min(dt_cfl, dt_diff, 0.5))

        # ---- 动量（显式） ----
        adv_u = _advect_upwind_masked(u, u, v, fluid)
        adv_v = _advect_upwind_masked(v, u, v, fluid)
        lap_u = _laplace(u); lap_v = _laplace(v)

        # 浮力（y向，屏幕坐标向下为正，因此热空气上浮=负）
        fy_buoy = - beta * g_mag * (T - 0.0)
        alpha = np.where(fluid, 0.0, penalty_alpha)

        u_star = u + dt * (-adv_u + nu*lap_u - alpha*u)
        v_star = v + dt * (-adv_v + nu*lap_v - alpha*v + fy_buoy)

        # 入口速度强制（法向吹/吸）
        u_star[inlet_mask[:,0]]  = inlet_u[inlet_mask[:,0]]
        u_star[inlet_mask[:,-1]] = inlet_u[inlet_mask[:,-1]]
        v_star[0,:][inlet_mask[0,:]]   = inlet_v[0,:][inlet_mask[0,:]]
        v_star[-1,:][inlet_mask[-1,:]] = inlet_v[-1,:][inlet_mask[-1,:]]

        # 关键：不可穿透 + 固体内速度=0
        u_star, v_star = _impenetrable_faces(u_star, v_star, solid_mask)

        # ---- 压力投影 ----
        u, v, p = _project_pressure(u_star, v_star, mask_fluid=fluid, outlet_mask=outlet_mask, niters=160)
        u, v = _impenetrable_faces(u, v, solid_mask)  # 再次强制（稳）

        # ---- 温度（对流+传导+体热源）----
        adv_T  = _advect_upwind_masked(T, u, v, fluid)              # 仅流体-流体对流
        diff_T = _variable_k_diffusion_harmonic(T, k, rho_cp)       # 多材料传导（跨界谐均值）
        src_T  = qdot / np.maximum(rho_cp, 1e-9)

        # 显式更新（稳态伪时间）
        T = T + dt * (-adv_T + diff_T + src_T)

        # 简单稳定化
        T = np.clip(T, -1e9, 1e9)

        # 外边界（非 in/out）近似无滑移
        left  = (~inlet_mask[:,0]) & (~outlet_mask[:,0])
        right = (~inlet_mask[:,-1]) & (~outlet_mask[:,-1])
        top   = (~inlet_mask[0,:]) & (~outlet_mask[0,:])
        bot   = (~inlet_mask[-1,:])& (~outlet_mask[-1,:])
        u[:,0][left]=0.0; v[:,0][left]=0.0
        u[:,-1][right]=0.0; v[:,-1][right]=0.0
        u[0,:][top]=0.0; v[0,:][top]=0.0
        u[-1,:][bot]=0.0; v[-1,:][bot]=0.0

    return T, u, v

# ================= 统一仿真入口 =================
def _apply_calib(T, calib):
    if not calib: return T
    a = float(calib.get("scale",1.0)); b = float(calib.get("bias",0.0))
    return a*T + b

def simulate_field(
    sources, supply, exhaust,
    canvas_w, canvas_h, grid_w, grid_h,
    sigma_px=120, backend="ffd",
    wall_mask=None,
    diff_alpha=0.1, diff_steps=20,
    calib=None,
    flow_iters=300, flow_scale=1.2,
    # 物性
    rho_air=1.2, mu_air=1.8e-5, k_air=0.026, cp_air=1005.0,
    beta=3e-3, g_mag=9.81,
    k_solid=0.005, rho_solid=2200.0, cp_solid=880.0,
    # 兼容 app.py 里传入的参数（此版本先不使用）
    fan_sigma=8, tin_mode="auto", tin_alpha=0.5, tin_fixed=-5.0,
    **kwargs,
):
    H, W = grid_h, grid_w
    solid = wall_mask if wall_mask is not None else np.zeros((H,W), dtype=bool)

    if backend not in {'ffd', 'builtin'}:
        raise ValueError('Only the built-in prototype solver is provided.')

    # 3) FFD→此处映射为：内置 N–S + 墙体约束解算器
    T, u, v = _ns_brinkman_with_walls(
        sources, supply, exhaust,
        canvas_w, canvas_h, grid_w, grid_h,
        solid_mask=solid,
        rho_air=rho_air, mu_air=mu_air, k_air=k_air, cp_air=cp_air,
        beta=beta, g_mag=g_mag,
        k_solid=k_solid, rho_solid=rho_solid, cp_solid=cp_solid,
        sigma_px=sigma_px, flow_iters=flow_iters, flow_scale=flow_scale,
        fan_sigma=fan_sigma, tin_mode=tin_mode, tin_alpha=tin_alpha, tin_fixed=tin_fixed,
    )

    T[solid] = 0.0  # 可视化时让墙区显示为 0（背景），以突出空气温度
    return _apply_calib(T, calib), u, v

# ================= 绘图（叠加墙体） =================
def plot_heatmap_with_suggestions(
    heatmap, sources, u=None, v=None, supply=None, exhaust=None,
    supply_suggest=None, exhaust_suggest=None, wall_mask=None,
    flip_vertical=True, arrow_scale=120
):
    H, W = heatmap.shape
    if flip_vertical:
        disp = np.flipud(heatmap)
        u_disp = np.flipud(u) if u is not None else None
        v_disp = -np.flipud(v) if v is not None else None
        wm = np.flipud(wall_mask) if wall_mask is not None else None
        fy = lambda y: (H-1-y)
    else:
        disp = heatmap; u_disp=u; v_disp=v; wm=wall_mask; fy=lambda y:y

    fig, ax = plt.subplots(figsize=(8,6), dpi=120)
    im = ax.imshow(disp, origin="lower", cmap="coolwarm")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("相对热量")

    # —— 墙体半透明遮罩 + 等值线，方便观察对流/温度影响 ——
    if wm is not None:
        ax.imshow(np.where(wm, 1.0, np.nan), origin="lower", alpha=0.30, cmap="gray")
        try:
            ax.contour(wm.astype(float), levels=[0.5], colors='k', linewidths=1.0, origin="lower")
        except Exception:
            pass

    # —— 送/回风点（置顶） ——
    if supply:
        xs=[p["x"] for p in supply]; ys=[fy(p["y"]) for p in supply]
        ax.plot(xs, ys, "^", ms=9, color="#00cc66", markeredgecolor="k", label="送风（现有）", zorder=6)
    if exhaust:
        xs=[p["x"] for p in exhaust]; ys=[fy(p["y"]) for p in exhaust]
        ax.plot(xs, ys, "D", ms=8, color="#9933ff", markeredgecolor="k", label="回风（现有）", zorder=6)

    # —— 热/冷源标记 ——
    for s in (sources or []):
        ax.plot(s["x"], fy(s["y"]), "o", ms=6,
                color=("#ff4444" if s.get("load",0)>0 else "#1f6bff"),
                markeredgecolor="k", alpha=0.9, zorder=6)

    # —— 建议点 ——
    if supply_suggest:
        xs=[x for x,_ in supply_suggest]; ys=[fy(y) for _,y in supply_suggest]
        ax.plot(xs, ys, "x", ms=10, mew=2, color="#00cc66", label="送风（建议）", zorder=7)
    if exhaust_suggest:
        xs=[x for x,_ in exhaust_suggest]; ys=[fy(y) for _,y in exhaust_suggest]
        ax.plot(xs, ys, "x", ms=10, mew=2, color="#9933ff", label="回风（建议）", zorder=7)

    # —— 风矢量（scale 越大箭头越短） ——
    if u_disp is not None and v_disp is not None:
        stride = max(1, W//24)
        Y, X = np.mgrid[0:H:stride, 0:W:stride]
        ax.quiver(X, Y, u_disp[::stride,::stride], v_disp[::stride,::stride],
                  color="k", alpha=0.35, scale=arrow_scale)

    ax.set_xlim(0, W-1); ax.set_ylim(0, H-1)
    ax.set_xlabel("x（网格）"); ax.set_ylabel("y（网格）")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("墙体生效：风绕墙/不可穿透，墙内仅传导（慢） / 热场与建议点")
    return fig
