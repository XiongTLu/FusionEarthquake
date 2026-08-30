import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from labellines import labelLines
from matplotlib.lines import Line2D
from utils import truncate_catalog_by_threshold, datetime_to_hours

# ===================== 基本设置 =====================
plot_earthquake = "Campotosto"   # "Visso", "Norcia", "Campotosto"
target_M = 3.0

HighestMcut = 3.0
Mcut_etas = 3.0   # 固定

LINEWIDTH = 1.0
LINE_ALPHA = 0.9

zero_color = "#6e6e6e"

colors = ["#184e77", "#1e6091", "#1a759f", "#168aad", "#34a0a4", "#52b69a", "#76c893", "#99d98c", "#b5e48c", "#d9ed92"]

if plot_earthquake == "Campotosto":
    lowestMcut = 1.4
    Mcut_npp = 1.3
    custom_colors = colors[1:]
else:
    lowestMcut = 1.2
    Mcut_npp = 1.2
    custom_colors = colors

# ===================== 工具函数：生成某个截断阈值下的 target event 时间轴 =====================
def build_target_event_days(raw_catalog, trunc_mcut, earthquake, target_M):
    """
    基于给定 trunc_mcut：
    1) 截断 catalog
    2) 取 test 区间
    3) 取 M >= target_M 的 target events
    4) 返回相对第0个 target event 的天数数组 x_days_full
    """
    truncated_catalog = truncate_catalog_by_threshold(raw_catalog, trunc_mcut)

    times = np.array(truncated_catalog['time'])
    mags = np.array(truncated_catalog['mw'])
    datetime_truncated = pd.to_datetime(
        truncated_catalog[['year', 'month', 'day', 'hour', 'minute', 'second']]
    )

    timeupto = {"Visso": 1200, "Norcia": 1800, "Campotosto": 3600}[earthquake]
    is_test = (times >= timeupto)

    M_test = mags[is_test]
    datetime_test = datetime_truncated[is_test]

    idx_target_all = np.where(M_test >= target_M)[0]
    if len(idx_target_all) == 0:
        raise ValueError(
            f"No target events found in test set with M >= {target_M} "
            f"under truncation Mcut={trunc_mcut:.1f}"
        )

    date_target_all = datetime_test.iloc[idx_target_all].reset_index(drop=True)
    x_days_full = (
        (date_target_all - date_target_all.iloc[0]).dt.total_seconds() / 86400.0
    ).to_numpy(dtype=float)

    return x_days_full


# ===================== 读取真实 catalog =====================
raw_data_file = './data/Catalogs/Amatrice_CAT5.v20210504_reduced_cols.csv'
AVN_catalog = pd.read_csv(raw_data_file)
AVN_catalog = datetime_to_hours(AVN_catalog).dropna()

# ===================== 分别构造两张图自己的参考 x 轴 =====================
# 图1：参考 ETAS(3.0)
x_days_etas_full = build_target_event_days(
    raw_catalog=AVN_catalog,
    trunc_mcut=Mcut_etas,
    earthquake=plot_earthquake,
    target_M=target_M
)

# 图2：参考 NPP(1.2 / 1.3)
x_days_npp_full = build_target_event_days(
    raw_catalog=AVN_catalog,
    trunc_mcut=Mcut_npp,
    earthquake=plot_earthquake,
    target_M=target_M
)

print(f'ETAS reference target-event length (Mcut={Mcut_etas:.1f}): {len(x_days_etas_full)}')
print(f'NPP  reference target-event length (Mcut={Mcut_npp:.1f}): {len(x_days_npp_full)}')

# ===================== 路径 =====================
etas_file_path   = './results_ETAS_trunc_VP'
NPP_file_path    = './results_NPP_trunc_seed'
fusion_file_path = './results_Fusion_trunc_VP_seed'

# ===================== 固定基准文件先读取一次 =====================
# ETAS 固定 3.0
etas_file_name = f'results_ETAS_trunc_VP_Mcut{Mcut_etas:.1f}_{plot_earthquake}.csv'
etas_fp = os.path.join(etas_file_path, etas_file_name)
etas_file = pd.read_csv(etas_fp)
etas_temp_pointwise_full = etas_file['ETAS_trunc_time_pointwise_ll'].to_numpy(float)

# NPP 固定 1.2 / 1.3
npp_file_name = f'results_NPP_trunc_Mcut{Mcut_npp:.1f}_{plot_earthquake}_seed1.csv'
npp_fp = os.path.join(NPP_file_path, npp_file_name)
npp_file = pd.read_csv(npp_fp)
npp_temp_pointwise_full = npp_file['NPP_time_like'].to_numpy(float)

print(f'Fixed ETAS file: {etas_fp}')
print(f'Fixed NPP  file: {npp_fp}')
print(f'ETAS fixed likelihood length: {len(etas_temp_pointwise_full)}')
print(f'NPP  fixed likelihood length: {len(npp_temp_pointwise_full)}')

# ===================== 存放每个 Mcut 的曲线 =====================
mcut_list = []

cig_fusion_vs_etas_list = []
cig_fusion_vs_npp_list = []

x_days_etas_list = []
x_days_npp_list = []

# 记录对齐区间，便于排查
etas_align_spans = []
npp_align_spans = []

# ===================== 循环读取多个 Fusion Mcut =====================
mcut_values = [round(x, 1) for x in np.arange(lowestMcut, HighestMcut + 0.1, 0.2)]

for Mcut in mcut_values:

    x_days_full = build_target_event_days(
        raw_catalog=AVN_catalog,
        trunc_mcut=Mcut,
        earthquake=plot_earthquake,
        target_M=target_M
    )

    fusion_file_name = f'results_Fusion_trunc_VP_Mcut{Mcut:.1f}_{plot_earthquake}_seed1.csv'
    fusion_fp = os.path.join(fusion_file_path, fusion_file_name)

    print(f"Fusion Mcut = {Mcut:.1f}")
    print("Reading files:")
    print(etas_fp)
    print(npp_fp)
    print(fusion_fp)

    fusion_file = pd.read_csv(fusion_fp)
    fusion_temp_pointwise = fusion_file['Fusion_time_like'].to_numpy(float)

    etas_temp_pointwise = etas_temp_pointwise_full.copy()
    npp_temp_pointwise = npp_temp_pointwise_full.copy()

    L = min(len(etas_temp_pointwise), len(fusion_temp_pointwise), len(npp_temp_pointwise))
    
    # ---------- 图1：Fusion(Mcut) vs ETAS(3.0) ----------
    etas_use = etas_temp_pointwise[:L]
    fusion_use_for_etas = fusion_temp_pointwise[:L]

    pw_gain_fusion_vs_etas = fusion_use_for_etas - etas_use
    cig_fusion_vs_etas = np.cumsum(pw_gain_fusion_vs_etas)

    # ---------- 图2：Fusion(Mcut) vs NPP(fixed) ----------
    npp_use = npp_temp_pointwise[:L]
    fusion_use_for_npp = fusion_temp_pointwise[:L]

    pw_gain_fusion_vs_npp = fusion_use_for_npp - npp_use
    cig_fusion_vs_npp = np.cumsum(pw_gain_fusion_vs_npp)

    x_days_etas = x_days_etas_full[-L:]
    x_days_npp = x_days_npp_full[-L:]
    x_days_fusion = x_days_full[-L:]

    assert len(x_days_etas) == len(x_days_npp) == len(x_days_fusion) == L, print(
        f"Length mismatch after alignment: x_days_etas={len(x_days_etas)}, "
        f"x_days_npp={len(x_days_npp)}, "
        f"x_days_fusion={len(x_days_fusion)}, "
        f"ETAS={len(etas_use)}, NPP={len(npp_use)}, Fusion={len(fusion_use_for_etas)}"
    )

    # ---------- 保存 ----------
    mcut_list.append(Mcut)

    cig_fusion_vs_etas_list.append(cig_fusion_vs_etas)
    cig_fusion_vs_npp_list.append(cig_fusion_vs_npp)

    x_days_etas_list.append(x_days_etas)
    x_days_npp_list.append(x_days_npp)


# ===================== 统一 y 轴范围 =====================
all_vals_etas = np.concatenate(cig_fusion_vs_etas_list)
all_vals_npp  = np.concatenate(cig_fusion_vs_npp_list)

ymin_etas = float(np.min(all_vals_etas))
ymax_etas = float(np.max(all_vals_etas))
ymin_npp  = float(np.min(all_vals_npp))
ymax_npp  = float(np.max(all_vals_npp))

ypad_etas = 0.05 * (ymax_etas - ymin_etas) if ymax_etas > ymin_etas else 1.0
ypad_npp  = 0.05 * (ymax_npp - ymin_npp) if ymax_npp > ymin_npp else 1.0

ylim_low_etas  = ymin_etas - ypad_etas
ylim_high_etas = ymax_etas + ypad_etas
ylim_low_npp   = ymin_npp - ypad_npp
ylim_high_npp  = ymax_npp + ypad_npp

# ===================== 图1：Fusion(Mcut) relative to ETAS(3.0) =====================
fig1, ax1 = plt.subplots(figsize=(8.0, 4.8))

for idx, Mcut in enumerate(mcut_list):
    color = custom_colors[idx % len(custom_colors)]
    ax1.plot(
        x_days_etas_list[idx],
        cig_fusion_vs_etas_list[idx],
        linestyle='-',
        linewidth=LINEWIDTH,
        alpha=LINE_ALPHA,
        color=color,
        label=f'{Mcut:.1f}'
    )

ax1.axhline(0, linestyle='--', color=zero_color, linewidth=LINEWIDTH)
ax1.set_xlabel('Time(days)')
ax1.set_ylabel('CIG time')
ax1.set_title(f'{plot_earthquake} | Fusion relative to ETAS({Mcut_etas:.1f})')
ax1.set_ylim(ylim_low_etas, ylim_high_etas)
ax1.ticklabel_format(axis='x', style='plain', useOffset=False)

labelLines(ax1.get_lines(), align=False, fontsize=12)
fig1.tight_layout()

# ===================== 图2：Fusion(Mcut) relative to NPP(fixed) =====================
fig2, ax2 = plt.subplots(figsize=(8.0, 4.8))

for idx, Mcut in enumerate(mcut_list):
    color = custom_colors[idx % len(custom_colors)]
    ax2.plot(
        x_days_npp_list[idx],
        cig_fusion_vs_npp_list[idx],
        linestyle='-',
        linewidth=LINEWIDTH,
        alpha=LINE_ALPHA,
        color=color,
        label=f'{Mcut:.1f}'
    )

ax2.axhline(0, linestyle='--', color=zero_color, linewidth=LINEWIDTH)
ax2.set_xlabel('Time(days)')
ax2.set_ylabel('CIG time')
ax2.set_title(f'{plot_earthquake} | Fusion relative to NPP({Mcut_npp:.1f})')
ax2.set_ylim(ylim_low_npp, ylim_high_npp)
ax2.ticklabel_format(axis='x', style='plain', useOffset=False)

labelLines(ax2.get_lines(), align=False, fontsize=12)
fig2.tight_layout()

# ===================== 保存 =====================
out_dir = './figures'
os.makedirs(out_dir, exist_ok=True)

fname1 = f'temporal_CIG_Fusion_multiMcut_vs_ETAS{Mcut_etas:.1f}_{plot_earthquake}_target_events'
fname2 = f'temporal_CIG_Fusion_multiMcut_vs_NPP{Mcut_npp:.1f}_{plot_earthquake}_target_events'

for ext in ['svg', 'pdf', 'png']:
    if ext == 'png':
        fig1.savefig(os.path.join(out_dir, f'{fname1}.{ext}'), facecolor='white')
        fig2.savefig(os.path.join(out_dir, f'{fname2}.{ext}'), facecolor='white')
    else:
        fig1.savefig(os.path.join(out_dir, f'{fname1}.{ext}'), transparent=True)
        fig2.savefig(os.path.join(out_dir, f'{fname2}.{ext}'), transparent=True)



# ===================== 单独保存图例 SVG =====================
legend_fig, legend_ax = plt.subplots(figsize=(6.0, 0.8))
legend_ax.axis('off')

legend_handles = [
    Line2D([0], [0],
           color=custom_colors[idx % len(custom_colors)],
           lw=LINEWIDTH,
           alpha=LINE_ALPHA,
           label=f'{Mcut:.1f}')
    for idx, Mcut in enumerate(mcut_list)
]

legend = legend_ax.legend(
    handles=legend_handles,
    loc='center',
    ncol=5,   # 一行排开；如果太长可改成 4 或 5
    frameon=False,
    handlelength=2.2,
    columnspacing=1.0,
    handletextpad=0.5,
    fontsize=12,
    title='Mcut'
)

legend_svg_name = f'temporal_CIG_legend_{plot_earthquake}.svg'
# legend_fig.savefig(
#     os.path.join(out_dir, legend_svg_name),
#     format='svg',
#     transparent=True,
#     bbox_inches='tight',
#     pad_inches=0.02
# )

print(os.path.join(out_dir, legend_svg_name))

plt.show()

# ===================== 输出对齐摘要 =====================
print("\nSaved to:")
print(os.path.join(out_dir, f'{fname1}.svg'))
print(os.path.join(out_dir, f'{fname1}.pdf'))
print(os.path.join(out_dir, f'{fname1}.png'))
print(os.path.join(out_dir, f'{fname2}.svg'))
print(os.path.join(out_dir, f'{fname2}.pdf'))
print(os.path.join(out_dir, f'{fname2}.png'))
