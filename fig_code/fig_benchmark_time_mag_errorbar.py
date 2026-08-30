

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os


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


# ============================================================
# 1. 汇总数据
#
# min Mcut:
#   NPP 和 Fusion
#
# max Mcut = 3.0:
#   仅 ETAS
#
# 数值格式：
#   (mean, std)
# ============================================================
data = {
    "ComCat\n(2.5)": {
        "min": {
            "NPP": {
                "temp": (-2.3815, 0.0039),
                "mag": (-1.1579, 0.0073),
            },
            "Fusion": {
                "temp": (-2.3538, 0.0118),
                "mag": (-0.1171, 0.0110),
            },
        },
        "max": {
            "ETAS": {
                "temp": (-2.4091, 0.0),
                "mag": (-0.1844, 0.0),
            },
            "NPP": {
                "temp": (-2.3977, 0.0133),
                "mag": (-1.1533, 0.0032),
            },
            "Fusion": {
                "temp": (-2.4335, 0.0642),
                "mag": (-0.0597, 0.0464),
            },
        },
    },

    "QTM-SS\n(1.0)": {
        "min": {
            "NPP": {
                "temp": (-4.5061, 0.1071),
                "mag": (-0.0822, 0.0221),
            },
            "Fusion": {
                "temp": (-4.3013, 0.0364),
                "mag": (-0.0721, 0.0133),
            },
        },
        "max": {
            "ETAS": {
                "temp": (-6.7590, 0.0),
                "mag": (0.0429, 0.0),
            },
            "NPP": {
                "temp": (-5.1582, 0.0642),
                "mag": (-0.3916, 0.0490),
            },
            "Fusion": {
                "temp": (-5.1867, 0.1451),
                "mag": (-0.1873, 0.0225),
            },
        },
    },

    "QTM-SJ\n(1.0)": {
        "min": {
            "NPP": {
                "temp": (-6.0370, 0.1597),
                "mag": (-0.8397, 0.0084),
            },
            "Fusion": {
                "temp": (-5.8521, 0.1100),
                "mag": (-0.1644, 0.0099),
            },
        },
        "max": {
            "ETAS": {
                "temp": (-9.3168, 0.0),
                "mag": (0.0782, 0.0),
            },
            "NPP": {
                "temp": (-6.5164, 0.2002),
                "mag": (-2.2434, 0.4829),
            },
            "Fusion": {
                "temp": (-6.4848, 0.2625),
                "mag": (-0.3709, 0.0649),
            },
        },
    },

    "SCEDC\n(2.0)": {
        "min": {
            "NPP": {
                "temp": (-1.3902, 0.0119),
                "mag": (-0.6906, 0.0136),
            },
            "Fusion": {
                "temp": (-1.1627, 0.0086),
                "mag": (-0.2218, 0.0045),
            },
        },
        "max": {
            "ETAS": {
                "temp": (-1.3314, 0.0),
                "mag": (-0.1907, 0.0),
            },
            "NPP": {
                "temp": (-1.3110, 0.0233),
                "mag": (-0.6661, 0.0125),
            },
            "Fusion": {
                "temp": (-1.3838, 0.0928),
                "mag": (-0.1955, 0.0009),
            },
        },
    },

    "WHITE\n(0.6)": {
        "min": {
            "NPP": {
                "temp": (-6.2545, 0.0498),
                "mag": (-1.4502, 0.1714),
            },
            "Fusion": {
                "temp": (-6.1629, 0.0492),
                "mag": (-0.4868, 0.0045),
            },
        },
        "max": {
            "ETAS": {
                "temp": (-7.8313, 0.0),
                "mag": (-0.0686, 0.0),
            },
            "NPP": {
                "temp": (-6.3803, 0.0513),
                "mag": (-1.8911, 0.1372),
            },
            "Fusion": {
                "temp": (-6.1441, 0.0584),
                "mag": (-0.4272, 0.0848),
            },
        },
    },

    "Campotosto\n(1.3)": {
        "min": {
            "NPP": {
                "temp": (-0.27792139363132023, 0.16668420927772876),
                "mag": (-0.2892429949622895, 0.036063010076962936),
            },
            "Fusion": {
                "temp": (-0.20881264241853437, 0.037611460502519165),
                "mag": (-0.1393179076248684, 0.0016484868278453673),
            },
        },
        "max": {
            "ETAS": {
                "temp": (-0.3643473374210569, 0.0),
                "mag": (-0.08436629946213556, 0.0),
            },
            "NPP": {
                "temp": (-0.6503234394093355, 0.05241048029837282),
                "mag": (-0.30652636234426667, 0.03682602796070623),
            },
            "Fusion": {
                "temp": (-0.47239105985562, 0.01027975070092327),
                "mag": (-0.3253796776512, 0.013519508825136957),
            },
        },
    },
}



# ============================================================
# 2. 颜色设置
# ============================================================
colors = {
    "ETAS": {
        "min": "#FFB117",
        "max": "#DE9622",
    },
    "NPP": {
        "min": "#6DB50A",
        "max": "#84BC53",
    },
    "Fusion": {
        "min": "#5F9BED",
        "max": "#74A9E6",
    },
}


# ============================================================
# 3. 基础绘图设置
#
# 时间 LL 和震级 LL 分开绘制：
#   图 1：Temp. LL，2 行 × 3 列
#   图 2：Mag. LL，2 行 × 3 列
# ============================================================
dataset_groups = [
    ["ComCat\n(2.5)", "QTM-SS\n(1.0)", "QTM-SJ\n(1.0)"],
    ["SCEDC\n(2.0)", "WHITE\n(0.6)", "Campotosto\n(1.3)"],
]

models = ["ETAS", "NPP", "Fusion"]
x = np.arange(len(models))


# ============================================================
# 4. 单个子图绘制函数
# ============================================================
def draw_panel(ax, dataset, metric_key):
    """
    绘制一个数据集的一项指标。

    metric_key:
        "temp"：时间 LL
        "mag"：震级 LL
    """

    # --------------------------------------------------------
    # ETAS：只绘制 max Mcut
    # --------------------------------------------------------
    etas_mean = data[dataset]["max"]["ETAS"][metric_key][0]

    ax.scatter(
        x[0],
        etas_mean,
        marker="s",
        s=42,
        facecolors="none",
        edgecolors=colors["ETAS"]["max"],
        linewidths=1.4,
        zorder=4,
    )

    # --------------------------------------------------------
    # NPP 和 Fusion：只绘制 min Mcut
    # --------------------------------------------------------
    for model_index, model in enumerate(
        ["NPP", "Fusion"],
        start=1,
    ):
        min_mean, min_std = data[dataset]["min"][model][metric_key]

        ax.errorbar(
            x[model_index],
            min_mean,
            yerr=min_std,
            fmt="o",
            markersize=5.8,
            markerfacecolor="none",
            markeredgecolor=colors[model]["min"],
            markeredgewidth=1.4,
            ecolor=colors[model]["min"],
            elinewidth=1.2,
            capsize=2.4,
            capthick=1.2,
            linestyle="none",
            zorder=3,
        )

    # --------------------------------------------------------
    # 坐标轴设置
    # --------------------------------------------------------
    ax.set_xticks(x)

    ax.set_xticklabels(
        models,
        fontsize=8,
        fontweight="bold",
    )

    ax.set_xlim(-0.45, 2.45)

    ax.tick_params(
        axis="both",
        direction="out",
        length=2.8,
        width=0.8,
        pad=2,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.30,
    )


# ============================================================
# 5. 图例
# ============================================================
legend_handles = [
    Line2D(
        [0],
        [0],
        marker="s",
        linestyle="none",
        markerfacecolor="none",
        markeredgecolor=colors["ETAS"]["max"],
        markeredgewidth=1.5,
        markersize=7,
        label="ETAS max Mcut",
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor="none",
        markeredgecolor=colors["NPP"]["min"],
        markeredgewidth=1.5,
        markersize=7,
        label="NPP min Mcut",
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor="none",
        markeredgecolor=colors["Fusion"]["min"],
        markeredgewidth=1.5,
        markersize=7,
        label="Fusion min Mcut",
    ),
]


# ============================================================
# 6. 创建并保存单项指标图
# ============================================================
def create_metric_figure(metric_key, ylabel, fname_base):
    """
    创建一个 2 行 × 3 列的图。

    metric_key:
        "temp" 或 "mag"

    ylabel:
        "Temp. LL" 或 "Mag. LL"

    fname_base:
        输出文件名，不包含扩展名
    """

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(8.27, 4.8),
        sharex=False,
        sharey=False,
    )

    # --------------------------------------------------------
    # 绘制 6 个数据集
    # --------------------------------------------------------
    for row, dataset_group in enumerate(dataset_groups):

        for col, dataset in enumerate(dataset_group):

            ax = axes[row, col]

            draw_panel(
                ax,
                dataset,
                metric_key,
            )

            # 数据集标题
            dataset_name, min_mcut = dataset.split("\n")
            min_mcut = min_mcut.strip("()")

            ax.set_title(
                f"{dataset_name} (min Mcut={min_mcut})",
                fontweight="bold",
                pad=2,
            )

    # --------------------------------------------------------
    # 左侧两行分别设置纵坐标标题
    # --------------------------------------------------------
    axes[0, 0].set_ylabel(
        ylabel,
        fontweight="bold",
        labelpad=3,
    )

    axes[1, 0].set_ylabel(
        ylabel,
        fontweight="bold",
        labelpad=3,
    )

    # --------------------------------------------------------
    # 整张图的图例
    # --------------------------------------------------------
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.8,
        labelspacing=0.35,
        borderaxespad=0.0,
    )

    # --------------------------------------------------------
    # 子图间距
    # --------------------------------------------------------
    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.095,
        top=0.82,
        wspace=0.16,
        hspace=0.34,
    )

    # --------------------------------------------------------
    # 保存路径
    # --------------------------------------------------------
    out_svg = os.path.join(
        out_dir,
        f"{fname_base}.svg",
    )

    out_pdf = os.path.join(
        out_dir,
        f"{fname_base}.pdf",
    )

    out_png = os.path.join(
        out_dir,
        f"{fname_base}.png",
    )

    # --------------------------------------------------------
    # 保存图片
    # --------------------------------------------------------
    # fig.savefig(
    #     out_svg,
    #     transparent=True,
    # )

    # fig.savefig(
    #     out_pdf,
    #     transparent=True,
    # )

    # fig.savefig(
    #     out_png,
    #     dpi=300,
    #     facecolor="white",
    # )

    return fig, (out_svg, out_pdf, out_png)


# ============================================================
# 7. 创建输出文件夹
# ============================================================
out_dir = "./figures"
os.makedirs(out_dir, exist_ok=True)


# ============================================================
# 8. 单独生成时间 LL 图
# ============================================================
fig_temp, temp_paths = create_metric_figure(
    metric_key="temp",
    ylabel="Temp. LL",
    fname_base="figure_6_benchmark_temp_ll",
)


# ============================================================
# 9. 单独生成震级 LL 图
# ============================================================
fig_mag, mag_paths = create_metric_figure(
    metric_key="mag",
    ylabel="Mag. LL",
    fname_base="figure_6_benchmark_mag_ll",
)


# ============================================================
# 10. 显示图片
# ============================================================
plt.show()


# ============================================================
# 11. 打印保存路径
# ============================================================
print("Temp. LL figure saved to:")

for path in temp_paths:
    print(path)


print("\nMag. LL figure saved to:")

for path in mag_paths:
    print(path)