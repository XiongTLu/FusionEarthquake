import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from utils import datetime_to_hours


# ===================== 基本设置 =====================
plot_earthquakes = ["Visso", "Norcia", "Campotosto"]  # 从左到右的三列顺序

Mcut_etas = 3.0

# A4 宽度：210 mm = 8.27 inch
A4_WIDTH_INCH = 8.27
FIG_HEIGHT_INCH = 8.80


# ===================== 可选：读取 catalog，用于检查 =====================
raw_data_file = './data/Catalogs/Amatrice_CAT5.v20210504_reduced_cols.csv'
AVN_catalog = pd.read_csv(raw_data_file)
AVN_catalog = datetime_to_hours(AVN_catalog).dropna()


# ===================== 路径设置 =====================
etas_file_path   = './results_ETAS_trunc_VP'
NPP_file_path    = './results_NPP_trunc_seed'
fusion_file_path = './results_Fusion_trunc_fixed_zhang_seed'

out_dir = './figures'
os.makedirs(out_dir, exist_ok=True)


# ===================== 绘图风格设置 =====================
plt.rcParams.update({
    "font.family": "Arial",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "axes.linewidth": 1.0,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 11,
})


# ===================== 手动写入 Table S1 的统计结果 =====================
# 注意：这里不重新计算，直接使用你表格中的 Positive、95% CI 和 p value
TABLE_S1_STATS = {
    ("Fusion - ETAS", "Visso"): {
        "positive": "54.9%",
        "ci": "51.7–57.9%",
        "p": "0.0025"
    },
    ("Fusion - ETAS", "Norcia"): {
        "positive": "58.4%",
        "ci": "55.1–61.7%",
        "p": r"$8.5 \times 10^{-7}$"
    },
    ("Fusion - ETAS", "Campotosto"): {
        "positive": "57.3%",
        "ci": "49.3–65.0%",
        "p": "0.086"
    },

    ("Fusion - NPP", "Visso"): {
        "positive": "64.7%",
        "ci": "61.7–67.6%",
        "p": r"$1.8 \times 10^{-20}$"
    },
    ("Fusion - NPP", "Norcia"): {
        "positive": "65.7%",
        "ci": "62.4–68.8%",
        "p": r"$3.0 \times 10^{-20}$"
    },
    ("Fusion - NPP", "Campotosto"): {
        "positive": "47.3%",
        "ci": "39.5–55.3%",
        "p": "0.568"
    },

    ("NPP - ETAS", "Visso"): {
        "positive": "43.6%",
        "ci": "40.5–46.7%",
        "p": r"$6.7 \times 10^{-5}$"
    },
    ("NPP - ETAS", "Norcia"): {
        "positive": "50.8%",
        "ci": "47.4–54.1%",
        "p": "0.682"
    },
    ("NPP - ETAS", "Campotosto"): {
        "positive": "59.3%",
        "ci": "51.3–66.9%",
        "p": "0.027"
    },
}


def get_mcut_fusion_npp(earthquake):
    if earthquake == "Campotosto":
        return 1.3, 1.3
    return 1.2, 1.2


def beautify_main_axis(ax):
    ax.minorticks_off()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(
        axis='both',
        which='major',
        direction='in',
        length=5,
        width=1.0,
        labelsize=8,
        top=False,
        right=False
    )
    for side in ['left', 'bottom']:
        ax.spines[side].set_linewidth(1.0)

    # 右侧边框由 twinx 的 ax_count 控制，避免双右轴重叠
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)


def beautify_count_axis(ax, count_color):
    """
    美化右侧 Event count 坐标轴。
    右侧 y 轴刻度同样朝内，并与柱状图颜色保持一致。
    """
    ax.minorticks_off()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.tick_params(
        axis='y',
        which='major',
        direction='in',
        length=5,
        width=1.0,
        colors=count_color,
        labelsize=8
    )

    # 避免 twin axis 额外显示 x 轴刻度
    ax.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False
    )

    ax.spines['right'].set_linewidth(1.0)
    ax.spines['right'].set_color(count_color)
    ax.spines['top'].set_linewidth(1.0)
    ax.yaxis.label.set_color(count_color)


# ===================== 工具函数 =====================
def read_required_column(csv_path, column_name):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if column_name not in df.columns:
        raise KeyError(
            f"Column '{column_name}' not found in {csv_path}. "
            f"Available columns: {df.columns.tolist()}"
        )

    return df[column_name].to_numpy(float)


def get_x_range(gain, pad_ratio=0.15):
    gain = np.asarray(gain, dtype=float)

    if len(gain) == 0:
        raise ValueError("Empty gain array.")

    xmin = np.min(gain)
    xmax = np.max(gain)

    if np.isclose(xmin, xmax):
        return xmin - 1.0, xmax + 1.0

    pad = pad_ratio * (xmax - xmin)
    return xmin - pad, xmax + pad


def make_bin_edges_with_zero(xmin, xmax, bins=25):
    """
    Generate uniform-width bin edges and force 0 to be one of the bin edges.
    """
    if xmin > xmax:
        xmin, xmax = xmax, xmin

    if np.isclose(xmin, xmax):
        xmin -= 1.0
        xmax += 1.0

    bin_width = (xmax - xmin) / bins
    bin_width = max(bin_width, 1e-12)

    left_edge = -np.ceil(abs(min(xmin, 0.0)) / bin_width) * bin_width
    right_edge = np.ceil(max(xmax, 0.0) / bin_width) * bin_width

    if np.isclose(left_edge, right_edge):
        right_edge = left_edge + bin_width

    n_bins = int(round((right_edge - left_edge) / bin_width))
    bin_edges = left_edge + np.arange(n_bins + 1) * bin_width

    zero_idx = np.argmin(np.abs(bin_edges))
    if np.isclose(bin_edges[zero_idx], 0.0, atol=1e-10):
        bin_edges[zero_idx] = 0.0

    return bin_edges


def calc_summary(gain):
    """
    这个函数只用于控制台打印检查信息。
    图中的 Positive / 95% CI / p value 使用 TABLE_S1_STATS 手动写入。
    """
    gain = np.asarray(gain, dtype=float)
    return {
        "N test": len(gain),
        "mean": np.mean(gain),
        "median": np.median(gain),
        "positive": np.mean(gain > 0) * 100
    }


def load_gain_for_earthquake(earthquake):
    Mcut_fusion, Mcut_npp = get_mcut_fusion_npp(earthquake)

    # ===================== 读取 ETAS fixed 3.0 =====================
    etas_file_name = f'results_ETAS_trunc_VP_Mcut{Mcut_etas:.1f}_{earthquake}.csv'
    etas_fp = os.path.join(etas_file_path, etas_file_name)

    etas_temp_pointwise = read_required_column(
        etas_fp,
        'ETAS_trunc_time_pointwise_ll'
    )

    # ===================== 读取 NPP fixed 1.2 / 1.3 =====================
    npp_file_name = f'results_NPP_trunc_Mcut{Mcut_npp:.1f}_{earthquake}_seed1.csv'
    npp_fp = os.path.join(NPP_file_path, npp_file_name)

    npp_temp_pointwise = read_required_column(
        npp_fp,
        'NPP_time_like'
    )

    # ===================== 读取 Fusion fixed 1.2 / 1.3 =====================
    fusion_file_name = (
        f'results_Fusion_trunc_fixed_zhang_Mcut{Mcut_fusion:.1f}_'
        f'{earthquake}_seed1.csv'
    )
    fusion_fp = os.path.join(fusion_file_path, fusion_file_name)

    fusion_temp_pointwise = read_required_column(
        fusion_fp,
        'Fusion_time_like'
    )

    # ===================== 对齐长度 =====================
    L = min(
        len(fusion_temp_pointwise),
        len(etas_temp_pointwise),
        len(npp_temp_pointwise)
    )

    if L == 0:
        raise ValueError(f"Aligned length is zero for {earthquake}. Please check input files.")

    if (
        len(fusion_temp_pointwise) != len(etas_temp_pointwise)
        or len(fusion_temp_pointwise) != len(npp_temp_pointwise)
    ):
        print(
            "[Warning] Length mismatch. Use first L values for alignment. "
            f"Dataset={earthquake}, L={L}, Fusion={len(fusion_temp_pointwise)}, "
            f"ETAS={len(etas_temp_pointwise)}, "
            f"NPP={len(npp_temp_pointwise)}"
        )

    fusion_use = fusion_temp_pointwise[:L]
    etas_use = etas_temp_pointwise[:L]
    npp_use = npp_temp_pointwise[:L]

    # ===================== 计算 event-wise temporal information gain =====================
    gain_fusion_vs_etas = fusion_use - etas_use
    gain_fusion_vs_npp  = fusion_use - npp_use
    gain_npp_vs_etas    = npp_use - etas_use

    return {
        "earthquake": earthquake,
        "Mcut_fusion": Mcut_fusion,
        "Mcut_npp": Mcut_npp,
        "files": {
            "Fusion": fusion_fp,
            "ETAS": etas_fp,
            "NPP": npp_fp
        },
        "gains": {
            "Fusion - ETAS": gain_fusion_vs_etas,
            "Fusion - NPP": gain_fusion_vs_npp,
            "NPP - ETAS": gain_npp_vs_etas
        }
    }


def plot_hist_pdf_count(
    ax_pdf,
    gain,
    title,
    stat_info=None,
    show_xlabel=False,
    show_left_ylabel=True,
    show_right_ylabel=True,
    show_legend=False,
    bins=25,
    count_alpha=0.16
):
    """
    左 y 轴：PDF，由 hist(density=True) 直接计算
    右 y 轴：Event count，由 hist(density=False) 直接计算

    0 被强制设置为一个 bin edge。
    不使用 KDE，不使用插值，不画 target event rug marks。

    stat_info:
        手动传入 Table S1 中的 Positive、95% CI 和 p value。
        不在这里重新计算这些统计量。
    """
    gain = np.asarray(gain, dtype=float)

    if len(gain) < 2:
        ax_pdf.set_title(f"{title} | Not enough data", fontsize=12)
        beautify_main_axis(ax_pdf)
        return

    xmin, xmax = get_x_range(gain)
    bin_edges = make_bin_edges_with_zero(xmin, xmax, bins=bins)

    # ===================== 左轴：PDF =====================
    ax_pdf.hist(
        gain,
        bins=bin_edges,
        density=True,
        histtype='step',
        alpha=1.0,
        color='red',
        linewidth=1.5,
        label='PDF',
        zorder=3
    )

    ax_pdf.axvline(
        0,
        color='black',
        linestyle='--',
        linewidth=1.5,
        label='0 gain',
        zorder=4
    )

    ax_pdf.set_xlim(bin_edges[0], bin_edges[-1])
    if show_left_ylabel:
        ax_pdf.set_ylabel('PDF', fontsize=12)
    else:
        ax_pdf.set_ylabel('')

    ax_pdf.set_title(title, fontsize=12, pad=6)

    if show_xlabel:
        ax_pdf.set_xlabel(
            'Event-wise TIG',
            fontsize=12,
            labelpad=5
        )

    ax_pdf.grid(
        True,
        linestyle='--',
        linewidth=0.7,
        alpha=0.22
    )
    beautify_main_axis(ax_pdf)

    y_min, y_max = ax_pdf.get_ylim()
    y_pad = 0.10 * (y_max - y_min)
    ax_pdf.set_ylim(y_min, y_max + y_pad)

    # ===================== 右轴：Event count =====================
    ax_count = ax_pdf.twinx()
    count_color = '#1f77b4'

    ax_count.hist(
        gain,
        bins=bin_edges,
        density=False,
        histtype='barstacked',
        alpha=count_alpha,
        facecolor=count_color,
        edgecolor=count_color,
        linewidth=0.7,
        zorder=1
    )

    if show_right_ylabel:
        ax_count.set_ylabel('Event count', color=count_color, fontsize=12)
    else:
        ax_count.set_ylabel('')

    y_min, y_max = ax_count.get_ylim()
    y_pad = 0.10 * (y_max - y_min)
    ax_count.set_ylim(y_min, y_max + y_pad)

    ax_count.spines['right'].set_color(count_color)
    beautify_count_axis(ax_count, count_color)

    # 让右轴柱状图位于 PDF 下面，避免视觉上压住红色 PDF
    ax_count.set_zorder(ax_pdf.get_zorder() - 1)
    ax_pdf.patch.set_visible(False)

    # ===================== Table S1 统计信息：手动写入 =====================
    if stat_info is not None:
        stat_text = (
            f"Positive = {stat_info['positive']}\n"
            f"95% CI = {stat_info['ci']}\n"
            f"p = {stat_info['p']}"
        )
    else:
        # 兜底：如果没有传 stat_info，就只显示代码计算的 positive
        # 正常情况下不会走到这里
        summary = calc_summary(gain)
        stat_text = f"Positive = {summary['positive']:.1f}%"

    ax_pdf.text(
        0.025,
        0.955,
        stat_text,
        transform=ax_pdf.transAxes,
        va='top',
        ha='left',
        fontsize=10,
        linespacing=1.20,
        bbox=dict(
            boxstyle='round,pad=0.30',
            facecolor='white',
            alpha=0.82,
            edgecolor='0.55',
            linewidth=0.8
        ),
        zorder=5
    )

    # ===================== 合并图例 =====================
    if show_legend:
        lines1, labels1 = ax_pdf.get_legend_handles_labels()
        lines2, labels2 = ax_count.get_legend_handles_labels()

        ax_pdf.legend(
            lines1 + lines2,
            labels1 + labels2,
            frameon=False,
            loc='upper right',
            fontsize=11
        )


# ===================== 读取数据并打印检查信息 =====================
all_results = {}
for earthquake in plot_earthquakes:
    result = load_gain_for_earthquake(earthquake)
    all_results[earthquake] = result

    print(f"\n========== Files: {earthquake} ==========")
    print(f"Fusion file: {result['files']['Fusion']}")
    print(f"ETAS file:   {result['files']['ETAS']}")
    print(f"NPP file:    {result['files']['NPP']}")

    print(f"\n========== Settings: {earthquake} ==========")
    print(f"Fusion Mcut = {result['Mcut_fusion']:.1f}")
    print(f"ETAS Mcut   = {Mcut_etas:.1f}")
    print(f"NPP Mcut    = {result['Mcut_npp']:.1f}")

    print(f"\n========== Summary: {earthquake} ==========")
    for name, gain in result["gains"].items():
        s = calc_summary(gain)

        table_s1 = TABLE_S1_STATS[(name, earthquake)]

        print(
            f"{name}: "
            f"N test={s['N test']}, "
            f"mean={s['mean']:.4f}, "
            f"median={s['median']:.4f}, "
            f"positive_from_code={s['positive']:.1f}%, "
            f"TableS1_positive={table_s1['positive']}, "
            f"TableS1_95CI={table_s1['ci']}, "
            f"TableS1_p={table_s1['p']}"
        )


# ===================== 绘图：3 * 3 =====================
fig, axes = plt.subplots(
    3,
    3,
    figsize=(A4_WIDTH_INCH, FIG_HEIGHT_INCH),
    sharex=False,
    sharey=False
)
fig.patch.set_facecolor('white')

comparison_rows = [
    "Fusion - ETAS",
    "Fusion - NPP",
    "NPP - ETAS"
]

for col_idx, earthquake in enumerate(plot_earthquakes):
    result = all_results[earthquake]
    Mcut_fusion = result["Mcut_fusion"]
    Mcut_npp = result["Mcut_npp"]

    n_test = calc_summary(result["gains"]["Fusion - ETAS"])["N test"]

    titles = {
        "Fusion - ETAS": (
            f'{earthquake.upper()}(N test={n_test})\n'
            f'Fusion({Mcut_fusion:.1f}) - ETAS({Mcut_etas:.1f})'
        ),
        "Fusion - NPP": (
            f'Fusion({Mcut_fusion:.1f}) - NPP({Mcut_npp:.1f})'
        ),
        "NPP - ETAS": (
            f'NPP({Mcut_npp:.1f}) - ETAS({Mcut_etas:.1f})'
        )
    }

    for row_idx, comparison_name in enumerate(comparison_rows):
        plot_hist_pdf_count(
            ax_pdf=axes[row_idx, col_idx],
            gain=result["gains"][comparison_name],
            title=titles[comparison_name],
            stat_info=TABLE_S1_STATS[(comparison_name, earthquake)],
            show_xlabel=(row_idx == 2),
            show_left_ylabel=(col_idx == 0),
            show_right_ylabel=(col_idx == 2),
            show_legend=(row_idx == 0 and col_idx == 0),
            bins=25,
            count_alpha=0.16
        )

fig.subplots_adjust(
    left=0.060,
    right=0.945,
    bottom=0.075,
    top=0.960,
    wspace=0.25,
    hspace=0.20
)


fname_base = 'figure_5_temp_IG_density_yossi'
out_png = os.path.join(out_dir, f'{fname_base}.png')
out_svg = os.path.join(out_dir, f'{fname_base}.svg')
out_pdf = os.path.join(out_dir, f'{fname_base}.pdf')

plt.savefig(out_svg, transparent=True)
plt.savefig(out_pdf, transparent=True)
plt.savefig(out_png, facecolor='white')
plt.show()

print("Saved to:")
print(out_svg)
print(out_pdf)
print(out_png)