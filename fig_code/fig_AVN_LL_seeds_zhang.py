import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker  # 为了右轴刻度格式化
from matplotlib.ticker import MultipleLocator

# ====== 全局样式 ======
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "axes.linewidth": 1.0,
    "savefig.dpi": 300,
    "figure.dpi": 300,
    "legend.frameon": False,
})

LINEWIDTH = 1.0   # 线更细
MS = 8.5          # 点更大
MEW = 0.9         # 点边框更细
BAND_ALPHA = 0.20
MARKER_ALPHA = 0.80

colors = {
    'ETAS VP':   '#ffbe0b',
    'NPP':       '#7fb800',
    'Fusion fixed': '#3a86ff',
    'Fusion VP': '#fb5607',
}
linestyles = {k: '-' for k in colors}
markers = {
    'ETAS VP':   '^',
    'NPP':       '^',
    'Fusion fixed': 'o',
    'Fusion VP': 'o',
}

# ETAS 单结果
file_path = './summary_results'

# NPP 多seed汇总结果（mean/std）
npp_file_path_seed = './summary_results_seed'

# Fusion 多seed汇总结果（mean/std）
fusion_file_path_seed = './summary_results_zhang_seed'

out_dir = './figures'
os.makedirs(out_dir, exist_ok=True)

# ====== 右侧 Training events 轴相关设置 ======
TRAIN_COLOR    = '#757575'
TICK_LABELSIZE = plt.rcParams["font.size"]
BAR_WIDTH      = 0.06


def apply_scaled_ticks_with_multiplier(ax, scale_power=4, nticks_target=4, color=None):
    scale = 10.0 ** scale_power
    ymin, ymax = ax.get_ylim()
    if ymax <= 0:
        ymax = 1.0
        ax.set_ylim(0, ymax)

    scaled_max = ymax / scale
    candidates = [1, 2, 4, 5, 10]
    step_scaled = candidates[-1]
    for c in candidates:
        if scaled_max / c <= nticks_target:
            step_scaled = c
            break

    n = int(np.ceil(scaled_max / step_scaled))
    ticks_scaled = np.arange(0, n + 1) * step_scaled
    ticks = ticks_scaled * scale
    ax.set_yticks(ticks)

    def _fmt(y, pos=None):
        val = y / scale
        if abs(val - round(val)) < 1e-8:
            return f"{int(round(val))}"
        return f"{val:.1f}"

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt))
    ax.yaxis.offsetText.set_visible(False)

    ax.text(1.02, 1.03, rf"$\times 10^{{{scale_power}}}$",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=TICK_LABELSIZE,
            color=(color if color is not None else None))


# ====== 读取 ntrain 数据 ======
ntrain_df = pd.read_csv('./figures/ntrain_pivot_summary.csv')
ntrain_df['Mcut'] = ntrain_df['Mcut'].astype(float)


def load_all(earthquake):
    # ===== ETAS：summary_results =====
    ETAS_VP_file = f'summary_ETAS_VP_{earthquake}.csv'

    # ===== Poisson：summary_results =====
    poiss_file = f'summary_NPP_seed1_{earthquake}.csv'

    # ===== NPP / Fusion：summary_results_seed =====
    NPP_file       = f'summary_NPP_5seed_{earthquake}.csv'
    Fusion_fixed_file = f'summary_Fusion_fixed_zhang_5seed_{earthquake}.csv'
    Fusion_VP_file    = f'summary_Fusion_VP_zhang_5seed_{earthquake}.csv'

    # ETAS
    df_ETAS_VP = pd.read_csv(os.path.join(file_path, ETAS_VP_file))

    # poiss
    df_Poisson = pd.read_csv(os.path.join(file_path, poiss_file))

    # NPP / Fusion
    df_NPP       = pd.read_csv(os.path.join(npp_file_path_seed, NPP_file))
    df_Fusion_fixed = pd.read_csv(os.path.join(fusion_file_path_seed, Fusion_fixed_file))
    df_Fusion_VP = pd.read_csv(os.path.join(fusion_file_path_seed, Fusion_VP_file))

    # 统一成 float
    for df in (df_ETAS_VP, df_Poisson, df_NPP, df_Fusion_VP):
        df['Mcut'] = df['Mcut'].astype(float)

    Mcuts = df_ETAS_VP['Mcut'].to_numpy()

    # 根据 Mcut 对齐训练事件数量
    ntrain_values = (
        ntrain_df.set_index('Mcut')[earthquake]
        .reindex(Mcuts, fill_value=0)
        .to_numpy()
    )

    # 找到 Mcut = 3.0 对应的位置
    idx_etas_30 = np.where(np.isclose(Mcuts, 3.0))[0]
    if len(idx_etas_30) == 0:
        raise ValueError(f'Cannot find Mcut=3.0 for earthquake={earthquake}')
    idx_etas_30 = idx_etas_30[0]

    data = dict(
        Mcuts=Mcuts,

        # baseline
        poiss_LL=df_Poisson['Poisson_LL'].to_numpy(),

        # ===== ETAS（单结果）=====
        temp_ETAS_VP=df_ETAS_VP['ETAS_VP_time_LL'].to_numpy(),
        mag_ETAS_VP=df_ETAS_VP['ETAS_VP_mag_LL'].to_numpy(),

        # ===== NPP（mean/std）=====
        temp_NPP_mean=df_NPP['NPP_time_LL_mean'].to_numpy(),
        temp_NPP_std=df_NPP['NPP_time_LL_std'].to_numpy(),
        mag_NPP_mean=df_NPP['NPP_mag_LL_mean'].to_numpy(),
        mag_NPP_std=df_NPP['NPP_mag_LL_std'].to_numpy(),

        # ===== Fusion（mean/std）=====
        temp_Fusion_VP_mean=df_Fusion_VP['Fusion_time_LL_mean'].to_numpy(),
        temp_Fusion_VP_std=df_Fusion_VP['Fusion_time_LL_std'].to_numpy(),
        mag_Fusion_VP_mean=df_Fusion_VP['Fusion_mag_LL_mean'].to_numpy(),
        mag_Fusion_VP_std=df_Fusion_VP['Fusion_mag_LL_std'].to_numpy(),

        # ===== Fusion fixed（mean/std）=====
        temp_Fusion_fixed_mean=df_Fusion_fixed['Fusion_time_LL_mean'].to_numpy(),
        temp_Fusion_fixed_std=df_Fusion_fixed['Fusion_time_LL_std'].to_numpy(),
        mag_Fusion_fixed_mean=df_Fusion_fixed['Fusion_mag_LL_mean'].to_numpy(),
        mag_Fusion_fixed_std=df_Fusion_fixed['Fusion_mag_LL_std'].to_numpy(),

        # ===== ETAS 3.0 横线 =====
        etas30_temp=(df_ETAS_VP['ETAS_VP_time_LL'].to_numpy()[idx_etas_30]
                     - df_Poisson['Poisson_LL'].to_numpy()[idx_etas_30]),
        etas30_mag=df_ETAS_VP['ETAS_VP_mag_LL'].to_numpy()[idx_etas_30],

        ntrain=ntrain_values,
    )

    print(f'poiss LL = {data["poiss_LL"]}')
    return data


def compute_column_ylims(datasets, pad_ratio=0.06):
    # ===== 左列：Temporal =====
    temporal_vals = []
    for data in datasets:
        temporal_vals.extend([
            data['temp_ETAS_VP'] - data['poiss_LL'],
            data['temp_NPP_mean'] - data['poiss_LL'],
            data['temp_NPP_mean'] - data['poiss_LL'] - data['temp_NPP_std'],
            data['temp_NPP_mean'] - data['poiss_LL'] + data['temp_NPP_std'],
            data['temp_Fusion_VP_mean'] - data['poiss_LL'],
            data['temp_Fusion_VP_mean'] - data['poiss_LL'] - data['temp_Fusion_VP_std'],
            data['temp_Fusion_VP_mean'] - data['poiss_LL'] + data['temp_Fusion_VP_std'],
            data['temp_Fusion_fixed_mean'] - data['poiss_LL'],
            data['temp_Fusion_fixed_mean'] - data['poiss_LL'] - data['temp_Fusion_fixed_std'],
            data['temp_Fusion_fixed_mean'] - data['poiss_LL'] + data['temp_Fusion_fixed_std'],
            np.array([data['etas30_temp']]),
        ])

    tmin = min(np.min(v) for v in temporal_vals)
    tmax = max(np.max(v) for v in temporal_vals)
    tpad = (tmax - tmin) * pad_ratio if tmax > tmin else 1.0
    temp_ylim = (tmin - tpad, tmax + tpad)

    # ===== 右列：Magnitude =====
    mag_vals = []
    for data in datasets:
        mag_vals.extend([
            data['mag_ETAS_VP'],
            data['mag_NPP_mean'],
            data['mag_NPP_mean'] - data['mag_NPP_std'],
            data['mag_NPP_mean'] + data['mag_NPP_std'],
            data['mag_Fusion_VP_mean'],
            data['mag_Fusion_VP_mean'] - data['mag_Fusion_VP_std'],
            data['mag_Fusion_VP_mean'] + data['mag_Fusion_VP_std'],
            data['mag_Fusion_fixed_mean'],
            data['mag_Fusion_fixed_mean'] - data['mag_Fusion_fixed_std'],
            data['mag_Fusion_fixed_mean'] + data['mag_Fusion_fixed_std'],
            np.array([data['etas30_mag']]),
        ])

    mmin = min(np.min(v) for v in mag_vals)
    mmax = max(np.max(v) for v in mag_vals)
    mpad = (mmax - mmin) * pad_ratio if mmax > mmin else 1.0
    mag_ylim = (mmin - mpad, mmax + mpad)

    return temp_ylim, mag_ylim

def plot_two(ax_temp, ax_mag, data, temp_ylim=None, mag_ylim=None):
    x = data['Mcuts']

    def draw_series(ax, y_mean, key, label, y_std=None):
        ax.plot(
            x, y_mean,
            color=colors[key],
            linestyle=linestyles[key],
            linewidth=LINEWIDTH,
            marker=markers[key],
            markersize=MS,
            markeredgewidth=MEW,
            markeredgecolor=colors[key],
            markerfacecolor='none',
            label=label,
            alpha=MARKER_ALPHA
        )
        if y_std is not None and len(y_std) == len(y_mean):
            ax.fill_between(
                x, y_mean - y_std, y_mean + y_std,
                color=colors[key], alpha=BAND_ALPHA, linewidth=0
            )

    # ======================
    #      左：Temporal
    # ======================
    draw_series(ax_temp, data['temp_ETAS_VP'] - data['poiss_LL'], 'ETAS VP', 'ETAS VP')

    draw_series(
        ax_temp,
        data['temp_NPP_mean'] - data['poiss_LL'],
        'NPP', 'NPP',
        data['temp_NPP_std']
    )

    draw_series(
        ax_temp,
        data['temp_Fusion_VP_mean'] - data['poiss_LL'],
        'Fusion VP', 'Fusion VP',
        data['temp_Fusion_VP_std']
    )

    draw_series(
        ax_temp,
        data['temp_Fusion_fixed_mean'] - data['poiss_LL'],
        'Fusion fixed', 'Fusion fixed',
        data['temp_Fusion_fixed_std']
    )

    # ===== ETAS 3.0 横向参考线 =====
    ax_temp.axhline(
        y=data['etas30_temp'],
        color='black',
        linestyle='--',
        linewidth=1.2,
        alpha=0.9
    )

    ax_temp.set_xlabel('Mcut')
    ax_temp.set_ylabel('Temporal Likelihood')
    if temp_ylim is not None:
        ax_temp.set_ylim(temp_ylim)

    ax_temp.yaxis.set_major_locator(MultipleLocator(0.5))
    
    ax_temp_bar = ax_temp.twinx()
    ax_temp_bar.bar(
        x, data['ntrain'],
        width=BAR_WIDTH,
        color='#bdbdbd',
        alpha=0.35,
        edgecolor='none',
        zorder=0
    )
    ax_temp_bar.set_ylim(0, max(data['ntrain']) * 1.15)
    ax_temp_bar.grid(False)
    ax_temp_bar.set_ylabel('')
    ax_temp_bar.tick_params(
        axis='y',
        which='both',
        right=False, labelright=False,
        left=False, labelleft=False
    )
    if 'right' in ax_temp_bar.spines:
        ax_temp_bar.spines['right'].set_visible(False)

    # ======================
    #      右：Magnitude
    # ======================
    draw_series(ax_mag, data['mag_ETAS_VP'], 'ETAS VP', 'ETAS VP')

    draw_series(
        ax_mag,
        data['mag_NPP_mean'],
        'NPP', 'NPP',
        data['mag_NPP_std']
    )

    draw_series(
        ax_mag,
        data['mag_Fusion_VP_mean'],
        'Fusion VP', 'Fusion VP',
        data['mag_Fusion_VP_std']
    )

    draw_series(
        ax_mag,
        data['mag_Fusion_fixed_mean'],
        'Fusion fixed', 'Fusion fixed',
        data['mag_Fusion_fixed_std']
    )

    # ===== ETAS 3.0 横向参考线 =====
    ax_mag.axhline(
        y=data['etas30_mag'],
        color='black',
        linestyle='--',
        linewidth=1.2,
        alpha=0.9
    )

    ax_mag.set_xlabel('Mcut')
    ax_mag.set_ylabel('Magnitude Likelihood')
    if mag_ylim is not None:
        ax_mag.set_ylim(mag_ylim)
    
    ax_mag.yaxis.set_major_locator(MultipleLocator(0.2))

    ax_right = ax_mag.twinx()
    ax_right.bar(
        x, data['ntrain'],
        width=BAR_WIDTH,
        color='#bdbdbd',
        alpha=0.35,
        edgecolor='none',
        zorder=0
    )
    ax_right.set_ylim(0, max(data['ntrain']) * 1.15)
    ax_right.grid(False)

    ax_right.set_ylabel('Training events',
                        fontsize=plt.rcParams["axes.labelsize"],
                        color=TRAIN_COLOR)
    ax_right.tick_params(axis='y',
                         which='both',
                         labelsize=TICK_LABELSIZE,
                         pad=2,
                         color=TRAIN_COLOR,
                         labelcolor=TRAIN_COLOR)

    apply_scaled_ticks_with_multiplier(ax_right,
                                       scale_power=4,
                                       nticks_target=4,
                                       color=TRAIN_COLOR)


# ===== 3×2 布局 =====
fig = plt.figure(figsize=(9, 9))
gs = GridSpec(3, 2, figure=fig, wspace=0.3, hspace=0.30)

visso = load_all('Visso')
norcia = load_all('Norcia')
campo = load_all('Campotosto')

ax_v1 = fig.add_subplot(gs[0, 0]); ax_v2 = fig.add_subplot(gs[0, 1])
plot_two(ax_v1, ax_v2, visso)

ax_n1 = fig.add_subplot(gs[1, 0]); ax_n2 = fig.add_subplot(gs[1, 1])
plot_two(ax_n1, ax_n2, norcia)

ax_c1 = fig.add_subplot(gs[2, 0]); ax_c2 = fig.add_subplot(gs[2, 1])
plot_two(ax_c1, ax_c2, campo)

# === 每一行左侧竖版行标题 ===
y_visso  = (ax_v1.get_position().y0 + ax_v1.get_position().y1) / 2.0
y_norcia = (ax_n1.get_position().y0 + ax_n1.get_position().y1) / 2.0
y_campo  = (ax_c1.get_position().y0 + ax_c1.get_position().y1) / 2.0
x_left = ax_v1.get_position().x0
x_title = x_left - 0.1
dy = 0.15

fig.text(x_title, y_visso + dy, 'Visso',
         ha='right', va='center',
         fontsize=16, fontweight='bold',
         rotation=90, rotation_mode='anchor')
fig.text(x_title, y_norcia + dy, 'Norcia',
         ha='right', va='center',
         fontsize=16, fontweight='bold',
         rotation=90, rotation_mode='anchor')
fig.text(x_title, y_campo + dy, 'Campotosto',
         ha='right', va='center',
         fontsize=16, fontweight='bold',
         rotation=90, rotation_mode='anchor')

# 图例（去重）
handles, labels = ax_n2.get_legend_handles_labels()
uniq = dict(zip(labels, handles))
fig.legend(list(uniq.values()), list(uniq.keys()),
           loc='lower center', ncol=3)

plt.subplots_adjust(top=0.95, bottom=0.18)

fname_base = 'LL_poiss_5seed_no_fixed_zhang'
out_svg = os.path.join(out_dir, f'{fname_base}.svg')
out_pdf = os.path.join(out_dir, f'{fname_base}.pdf')
out_png = os.path.join(out_dir, f'{fname_base}.png')

plt.savefig(out_svg, transparent=True)
plt.savefig(out_pdf, transparent=True)
plt.savefig(out_png, facecolor='white')
plt.show()

print("Saved to:")
print(out_svg)
print(out_pdf)
print(out_png)