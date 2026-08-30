import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import sys
from tqdm import tqdm
from utils import (
    truncate_catalog_by_threshold,  
    datetime_to_hours,
    append_burn_in_to_test_set      
)

# ==== A4 Single Column (≈ 90 mm) ====
import matplotlib as mpl

# ---- 关键：导出体积优化（新增） ----
mpl.rcParams["agg.path.chunksize"] = 10000     # PDF 路径分块，避免超长路径
mpl.rcParams["svg.fonttype"] = "none"          # SVG 文本不转曲线，AI 可编辑
mpl.rcParams["pdf.compression"] = 9

# 版式尺寸
FIG_WIDTH_IN   = 3.54    # ≈ 90 mm
ROW_HEIGHT_IN  = 2.20    # 建图时常用: height ≈ n_rows * ROW_HEIGHT_IN
WSPACE, HSPACE = 0.28, 0.25

# 样式参数
LW          = 1.2
MS          = 5          # marker 不小于 5，印刷更清晰
MEW         = 1.2
LINE_ALPHA  = 0.60
FILL_ALPHA  = 0.30
BAR_WIDTH   = 0.05
TRAIN_COLOR = "#757575"

# 调色板 / 线型 / 标记
COLORS = {
    "ETAS": "#abc8e5",
    "NPP": "#eab67a",
    "NPP-ETAS (params)": "#d8a0c1",
    "NPP-ETAS": "#c16e71",
}
LINESTYLES = {k: "-" for k in COLORS}
MARKERS    = {"ETAS": "s", "NPP": "o", "NPP-ETAS (params)": "p", "NPP-ETAS": "p"}

# 全局 rc
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,       # 刻度
    "axes.labelsize": 13,  # 轴标签
    "legend.fontsize": 11,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "savefig.dpi": 300,
    "figure.dpi": 300,
    "legend.frameon": False,
})
# ==== End Single ====

LABEL_FONTSIZE  = plt.rcParams["axes.labelsize"]          # 13
TICK_LABELSIZE  = plt.rcParams["font.size"]               # 11
LEGEND_FONTSIZE = plt.rcParams["legend.fontsize"]         # 11

def select_training_testing_partition(earthquake):
    if earthquake == 'Visso':
        timeupto = 1200
    elif earthquake == 'Norcia':
        timeupto = 1800
    elif earthquake == 'Campotosto':
        timeupto = 3600
    else:
        print('Invalid argument')
        sys.exit()
    return timeupto

def main():
    # ===== 读取完整目录 =====
    AVN_catalog = pd.read_csv('./data/Catalogs/Amatrice_CAT5.v20210504_reduced_cols.csv')
    AVN_catalog = datetime_to_hours(AVN_catalog)
    AVN_catalog = AVN_catalog.dropna()

    # ===== 统计信息（可选） =====
    print('Total events:', len(AVN_catalog))
    print('Events Mw ≥ 3.0:', int((AVN_catalog['mw'] >= 3.0).sum()))

    # ===== 绘制全部背景地震（不使用 Mcut、不使用 earthquake）=====
    mags_full = AVN_catalog['mw'].to_numpy()
    datetime_full = pd.to_datetime(
        AVN_catalog[['year', 'month', 'day', 'hour', 'minute', 'second']]
    )
    mask_large = mags_full >= 3.0
    mask_small = ~mask_large

    # === 使用单栏参数的画布 ===
    fig, ax = plt.subplots(figsize=(7.09, 3.0))


    # 散点尺寸（沿用 Mw^2 放大规则）
    SIZE_K_SMALL = 1.0   # 小震放大系数（原来是 4）
    SIZE_K_LARGE = 1.3   # 大震放大系数（原来是 4，略大以突出 ≥3.0）

    size_small = (mags_full[mask_small] ** 2) * SIZE_K_SMALL
    size_large = (mags_full[mask_large] ** 2) * SIZE_K_LARGE

    # Mw < 3.0（灰）—— 点云栅格化，降低 SVG/PDF 体积
    ax.scatter(
        datetime_full[mask_small], mags_full[mask_small],
        s=size_small,
        facecolors='#bdbdbd', edgecolors='none',
        alpha=0.4, label='Mw < 3.0',
        zorder=1, rasterized=True
    )
    # Mw ≥ 3.0（#82add0）—— 同样栅格化（点很多）
    ax.scatter(
        datetime_full[mask_large], mags_full[mask_large],
        s=size_large,
        facecolors='#82add0', edgecolors='none',
        alpha=0.9, label='Mw ≥ 3.0',
        zorder=2, rasterized=True
    )

    # 轴范围与刻度
    xmin = datetime_full.min() - pd.Timedelta(days=3)
    xmax = datetime_full.max() + pd.Timedelta(days=3)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim(0, 7.5)

    ax.set_ylabel("Magnitude (Mw)", fontsize=LABEL_FONTSIZE)
    ax.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())

    # 主震标注（保持矢量）
    mainshocks = [
        ('Amatrice', pd.Timestamp('2016-08-24 01:36:32')),
        ('Visso', pd.Timestamp('2016-10-26 19:18:08')),
        ('Norcia', pd.Timestamp('2016-10-30 06:40:17')),
        ('Campotosto', pd.Timestamp('2017-01-18 09:25:40')),
    ]
    label_offsets = {
        'Visso':      {'days': -2, 'ha': 'right', 'y': 7.3},
        'Norcia':     {'days':  +2, 'ha': 'left',  'y': 7.3},
        'Amatrice':   {'days':   0, 'ha': 'center','y': 7.3},
        'Campotosto': {'days':   0, 'ha': 'center','y': 7.3},
    }

    for label, t in mainshocks:
        ax.axvline(t, color='#eb6368', linestyle='--', linewidth=LW, alpha=0.7, zorder=3)
        off = label_offsets[label]
        ax.annotate(
            label,
            xy=(t, 6.5),
            xytext=(t + pd.Timedelta(days=off['days']), off['y']),
            ha=off['ha'], fontsize=TICK_LABELSIZE, color='#eb6368',
            arrowprops=dict(arrowstyle='->', lw=LW, color='#eb6368'),
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='gray'),
            zorder=3
        )

    # 刻度标签字号微调
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE)

    # 图例（与点色一致）
    legend_handles = [
        mlines.Line2D([], [], color='none', marker='o', markersize=MS+2,
                      markerfacecolor='#d6d6d6', markeredgewidth=0, label='Mw < 3.0'),
        mlines.Line2D([], [], color='none', marker='o', markersize=MS+2,
                      markerfacecolor='#82add0', markeredgewidth=0, label='Mw ≥ 3.0')
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=LEGEND_FONTSIZE, frameon=False)

    # 保存/显示
    paper_fig_dir = './figures_paper'
    os.makedirs(paper_fig_dir, exist_ok=True)

    save_path_svg = os.path.join(paper_fig_dir, 'figure_1_a_2.svg')
    save_path_pdf = os.path.join(paper_fig_dir, 'figure_1_a_2.pdf')
    save_path_png = os.path.join(paper_fig_dir, 'figure_1_a_2.png')

    plt.tight_layout()

    # 透明底导出（SVG/PDF 重新开启）
    # plt.savefig(save_path_svg, dpi=300, bbox_inches='tight', transparent=False)
    # plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight', transparent=False)
    # plt.savefig(save_path_png, dpi=300, transparent=True)

    plt.show()

    # print(f"✅ 图像已保存到: {save_path_svg}")
    # print(f"✅ 图像已保存到: {save_path_pdf}")
    # print(f"✅ 图像已保存到: {save_path_png}")

if __name__ == '__main__':
    main()
