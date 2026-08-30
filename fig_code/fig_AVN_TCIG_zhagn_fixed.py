import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from labellines import labelLines
from matplotlib.lines import Line2D
from utils import truncate_catalog_by_threshold, datetime_to_hours


# ===================== 基本设置 =====================
summary_earthquakes = ["Visso", "Norcia", "Campotosto"]
target_M = 3.0
HighestMcut = 3.0
Mcut_etas = 3.0

LINEWIDTH = 1.0
LINE_ALPHA = 0.9
LABEL_FONTSIZE = 12
zero_color = "#6e6e6e"

colors = [
    "#184e77",
    "#1e6091",
    "#1a759f",
    "#168aad",
    "#34a0a4",
    "#52b69a",
    "#76c893",
    "#99d98c",
    "#b5e48c",
    "#d9ed92",
]

# ===================== 路径 =====================
raw_data_file = "./data/Catalogs/Amatrice_CAT5.v20210504_reduced_cols.csv"
etas_file_path = "./results_ETAS_trunc_VP"
NPP_file_path = "./results_NPP_trunc_seed"
fusion_file_path = "./results_Fusion_trunc_VP_seed"

out_dir = "./figures"
os.makedirs(out_dir, exist_ok=True)


# ===================== 地区配置 =====================
def get_earthquake_config(earthquake):
    if earthquake == "Campotosto":
        lowest_mcut = 1.4
        mcut_npp = 1.3
        custom_colors = colors[1:]
    else:
        lowest_mcut = 1.2
        mcut_npp = 1.2
        custom_colors = colors

    mcut_values = [
        round(x, 1)
        for x in np.arange(lowest_mcut, HighestMcut + 0.1, 0.2)
    ]

    return {
        "mcut_npp": mcut_npp,
        "custom_colors": custom_colors,
        "mcut_values": mcut_values,
    }


# ===================== 工具函数：生成 target event 时间轴 =====================
def build_target_event_days(raw_catalog, trunc_mcut, earthquake, target_m):
    truncated_catalog = truncate_catalog_by_threshold(raw_catalog, trunc_mcut)

    times = np.asarray(truncated_catalog["time"])
    mags = np.asarray(truncated_catalog["mw"])
    datetime_truncated = pd.to_datetime(
        truncated_catalog[["year", "month", "day", "hour", "minute", "second"]]
    )

    timeupto = {"Visso": 1200, "Norcia": 1800, "Campotosto": 3600}[earthquake]
    is_test = times >= timeupto

    m_test = mags[is_test]
    datetime_test = datetime_truncated[is_test]
    idx_target_all = np.where(m_test >= target_m)[0]

    if len(idx_target_all) == 0:
        raise ValueError(
            f"No target events found in test set with M >= {target_m} "
            f"under truncation Mcut={trunc_mcut:.1f}, earthquake={earthquake}"
        )

    date_target_all = datetime_test.iloc[idx_target_all].reset_index(drop=True)
    x_days_full = (
        (date_target_all - date_target_all.iloc[0]).dt.total_seconds() / 86400.0
    ).to_numpy(dtype=float)

    return x_days_full


def calculate_ylim(curves):
    all_values = np.concatenate(curves)
    ymin = float(np.min(all_values))
    ymax = float(np.max(all_values))
    ypad = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
    return ymin - ypad, ymax + ypad


# ===================== 读取真实 catalog =====================
AVN_catalog = pd.read_csv(raw_data_file)
AVN_catalog = datetime_to_hours(AVN_catalog).dropna()


# ===================== 计算单个地区结果 =====================
def calculate_earthquake_results(earthquake):
    config = get_earthquake_config(earthquake)
    mcut_npp = config["mcut_npp"]
    mcut_values = config["mcut_values"]
    custom_colors = config["custom_colors"]

    x_days_etas_full = build_target_event_days(
        raw_catalog=AVN_catalog,
        trunc_mcut=Mcut_etas,
        earthquake=earthquake,
        target_m=target_M,
    )
    x_days_npp_full = build_target_event_days(
        raw_catalog=AVN_catalog,
        trunc_mcut=mcut_npp,
        earthquake=earthquake,
        target_m=target_M,
    )

    etas_file_name = f"results_ETAS_trunc_VP_Mcut{Mcut_etas:.1f}_{earthquake}.csv"
    etas_fp = os.path.join(etas_file_path, etas_file_name)
    etas_file = pd.read_csv(etas_fp)
    etas_temp_pointwise_full = etas_file["ETAS_trunc_time_pointwise_ll"].to_numpy(float)

    npp_file_name = f"results_NPP_trunc_Mcut{mcut_npp:.1f}_{earthquake}_seed1.csv"
    npp_fp = os.path.join(NPP_file_path, npp_file_name)
    npp_file = pd.read_csv(npp_fp)
    npp_temp_pointwise_full = npp_file["NPP_time_like"].to_numpy(float)

    mcut_list = []
    cig_fusion_vs_etas_list = []
    cig_fusion_vs_npp_list = []
    x_days_etas_list = []
    x_days_npp_list = []

    for mcut in mcut_values:
        x_days_fusion_full = build_target_event_days(
            raw_catalog=AVN_catalog,
            trunc_mcut=mcut,
            earthquake=earthquake,
            target_m=target_M,
        )

        fusion_file_name = f"results_Fusion_trunc_VP_Mcut{mcut:.1f}_{earthquake}_seed1.csv"
        fusion_fp = os.path.join(fusion_file_path, fusion_file_name)
        fusion_file = pd.read_csv(fusion_fp)
        fusion_temp_pointwise = fusion_file["Fusion_time_like"].to_numpy(float)

        etas_temp_pointwise = etas_temp_pointwise_full.copy()
        npp_temp_pointwise = npp_temp_pointwise_full.copy()

        length = min(
            len(etas_temp_pointwise),
            len(fusion_temp_pointwise),
            len(npp_temp_pointwise),
        )

        etas_use = etas_temp_pointwise[:length]
        fusion_use_for_etas = fusion_temp_pointwise[:length]
        cig_fusion_vs_etas = np.cumsum(fusion_use_for_etas - etas_use)

        npp_use = npp_temp_pointwise[:length]
        fusion_use_for_npp = fusion_temp_pointwise[:length]
        cig_fusion_vs_npp = np.cumsum(fusion_use_for_npp - npp_use)

        x_days_etas = x_days_etas_full[-length:]
        x_days_npp = x_days_npp_full[-length:]
        x_days_fusion = x_days_fusion_full[-length:]

        if not (
            len(x_days_etas) == len(x_days_npp) == len(x_days_fusion) == length
        ):
            raise ValueError(
                "Length mismatch after alignment: "
                f"earthquake={earthquake}, Mcut={mcut:.1f}, "
                f"x_days_etas={len(x_days_etas)}, "
                f"x_days_npp={len(x_days_npp)}, "
                f"x_days_fusion={len(x_days_fusion)}, "
                f"ETAS={len(etas_use)}, NPP={len(npp_use)}, Fusion={len(fusion_use_for_etas)}"
            )

        mcut_list.append(mcut)
        cig_fusion_vs_etas_list.append(cig_fusion_vs_etas)
        cig_fusion_vs_npp_list.append(cig_fusion_vs_npp)
        x_days_etas_list.append(x_days_etas)
        x_days_npp_list.append(x_days_npp)

    return {
        "earthquake": earthquake,
        "mcut_list": mcut_list,
        "custom_colors": custom_colors,
        "x_days_etas_list": x_days_etas_list,
        "x_days_npp_list": x_days_npp_list,
        "cig_fusion_vs_etas_list": cig_fusion_vs_etas_list,
        "cig_fusion_vs_npp_list": cig_fusion_vs_npp_list,
        "ylim_etas": calculate_ylim(cig_fusion_vs_etas_list),
        "ylim_npp": calculate_ylim(cig_fusion_vs_npp_list),
    }


# ===================== 通用绘图函数 =====================
def draw_cig_panel(
    ax,
    x_days_list,
    cig_list,
    mcut_list,
    custom_colors,
    ylim,
    title=None,
    xlabel="Time(days)",
    ylabel="CIG time",
    add_line_labels=True,
):
    data_lines = []

    for idx, mcut in enumerate(mcut_list):
        line, = ax.plot(
            x_days_list[idx],
            cig_list[idx],
            linestyle="-",
            linewidth=LINEWIDTH,
            alpha=LINE_ALPHA,
            color=custom_colors[idx % len(custom_colors)],
            label=f"{mcut:.1f}",
        )
        data_lines.append(line)

    ax.axhline(0, linestyle="--", color=zero_color, linewidth=LINEWIDTH)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    ax.set_ylim(*ylim)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)

    if add_line_labels:
        labelLines(data_lines, align=False, fontsize=LABEL_FONTSIZE)


# ===================== 保存函数 =====================
def save_figure(fig, base_name):
    saved_paths = []
    for ext in ["svg", "pdf", "png"]:
        output_path = os.path.join(out_dir, f"{base_name}.{ext}")
        if ext == "png":
            fig.savefig(output_path, facecolor="white")
        else:
            fig.savefig(output_path, transparent=True)
        saved_paths.append(output_path)
    return saved_paths


# ===================== 2×3 汇总图 =====================
def plot_summary_2x3(all_results):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18.0, 9.0))

    for col_idx, earthquake in enumerate(summary_earthquakes):
        result = all_results[earthquake]

        draw_cig_panel(
            ax=axes[0, col_idx],
            x_days_list=result["x_days_etas_list"],
            cig_list=result["cig_fusion_vs_etas_list"],
            mcut_list=result["mcut_list"],
            custom_colors=result["custom_colors"],
            ylim=result["ylim_etas"],
            title=earthquake,
            xlabel="Time(days)",
            ylabel="CIG time",
            add_line_labels=True,
        )

        draw_cig_panel(
            ax=axes[1, col_idx],
            x_days_list=result["x_days_npp_list"],
            cig_list=result["cig_fusion_vs_npp_list"],
            mcut_list=result["mcut_list"],
            custom_colors=result["custom_colors"],
            ylim=result["ylim_npp"],
            title=None,
            xlabel="Time(days)",
            ylabel="CIG time",
            add_line_labels=True,
        )

    fig.text(0.018, 0.705, "VS ETAS", rotation=90, va="center", ha="center", fontsize=15, fontweight="bold")
    fig.text(0.018, 0.285, "VS NPP", rotation=90, va="center", ha="center", fontsize=15, fontweight="bold")

    all_summary_mcuts = [round(x, 1) for x in np.arange(1.2, HighestMcut + 0.1, 0.2)]
    summary_legend_handles = [
        Line2D(
            [0],
            [0],
            color=colors[idx],
            lw=LINEWIDTH,
            alpha=LINE_ALPHA,
            label=f"{mcut:.1f}",
        )
        for idx, mcut in enumerate(all_summary_mcuts)
    ]

    fig.legend(
        handles=summary_legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.2,
        handletextpad=0.5,
        fontsize=12,
        title="Mcut",
        bbox_to_anchor=(0.5, 0.012),
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.93,
        bottom=0.145,
        wspace=0.22,
        hspace=0.28,
    )

    summary_name = "temporal_CIG_Fusion_multiMcut_summary_2x3_Visso_Norcia_Campotosto_target_events"
    return fig, save_figure(fig, summary_name)


# ===================== 主程序 =====================
def main():
    all_results = {
        earthquake: calculate_earthquake_results(earthquake)
        for earthquake in summary_earthquakes
    }

    summary_fig, summary_paths = plot_summary_2x3(all_results)

    print("\nSaved 2x3 summary figure:")
    for path in summary_paths:
        print(path)

    plt.show()


if __name__ == "__main__":
    main()