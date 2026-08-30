import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils import truncate_catalog_by_threshold, datetime_to_hours

plot_earthquake = "Visso"   # "Visso", "Norcia", "Campotosto"

if plot_earthquake == 'Campotosto':
    plot_Mcut = 1.3
else:
    plot_Mcut = 1.2

plot_etas_Mcut = 3.0
target_M = 3.0

LINEWIDTH = 1.5
LINE_ALPHA = 0.8

# ["#efc7c2","#ffe5d4","#bfd3c1"]
# ["#ffa69e","#faf3dd","#b8f2e6","#aed9e0"]
# ["#d4e09b","#f6f4d2","#cbdfbd","#f19c79"]
colors = {
    'ETAS':   '#5bc0eb',
    'NPP':    '#9bc53d',
    'Fusion': '#ef767a',
    'star':   '#ffd670',
}

linestyles = {
    'ETAS': '-',
    'NPP': '-',
    'Fusion': '-',
}
def select_training_testing_partition(earthquake):
    return {"Visso": 1200, "Norcia": 1800, "Campotosto": 3600}[earthquake]

# ===================== 读取三个模型输出 =====================
etas_file_path = './etas_inputs_trunc_VP'
NPP_file_path = './pred_NPP_trunc_seed'
fusion_file_path = './pred_Fusion_trunc_fixed_zhang_seed'

etas_file_name   = f'ETAS_trunc_VP_test_time_Mcut{plot_etas_Mcut}_params{plot_etas_Mcut}_{plot_earthquake}.csv'
NPP_file_name    = f'pred_NPP_trunc_Mcut{plot_Mcut}_{plot_earthquake}_seed1.csv'
fusion_file_name = f'pred_Fusion_trunc_fixed_zhang_Mcut{plot_Mcut}_{plot_earthquake}_seed1.csv'

print("Reading files:")
print(os.path.join(etas_file_path, etas_file_name))
print(os.path.join(NPP_file_path, NPP_file_name))
print(os.path.join(fusion_file_path, fusion_file_name))

etas_file   = pd.read_csv(os.path.join(etas_file_path, etas_file_name))
NPP_file    = pd.read_csv(os.path.join(NPP_file_path, NPP_file_name))
fusion_file = pd.read_csv(os.path.join(fusion_file_path, fusion_file_name))

etas_temp_rate   = etas_file['time rate full'].to_numpy(float)
NPP_temp_rate    = NPP_file['nn_temp_rate'].to_numpy(float)
fusion_temp_rate = fusion_file['fusion_temp_rate'].to_numpy(float)

print('ETAS raw length  :', len(etas_temp_rate))
print('NPP raw length   :', len(NPP_temp_rate))
print('Fusion raw length:', len(fusion_temp_rate))

# ===================== 读取真实测试事件 =====================
raw_data_file = './data/Catalogs/Amatrice_CAT5.v20210504_reduced_cols.csv'
AVN_catalog = pd.read_csv(raw_data_file)
AVN_catalog = datetime_to_hours(AVN_catalog).dropna()
truncated_catalog = truncate_catalog_by_threshold(AVN_catalog, plot_Mcut)

times = np.array(truncated_catalog['time'])
mags  = np.array(truncated_catalog['mw'])
datetime_truncated = pd.to_datetime(
    truncated_catalog[['year', 'month', 'day', 'hour', 'minute', 'second']]
)

timeupto = select_training_testing_partition(plot_earthquake)
is_test = (times >= timeupto)

M_test = mags[is_test]
datetime_test = datetime_truncated[is_test]

print('test event length:', len(M_test))

# ===================== 尾部长度对齐 =====================
L_raw = min(len(NPP_temp_rate), len(fusion_temp_rate))

npp_use    = NPP_temp_rate[-L_raw:]
fusion_use = fusion_temp_rate[-L_raw:]

M_use = M_test[-L_raw:]
datetime_use = datetime_test.iloc[-L_raw:].reset_index(drop=True)

# ===================== 只保留目标事件 =====================
idx_target = np.where(M_use >= target_M)[0]

if len(idx_target) == 0:
    raise ValueError(f'No target events with M >= {target_M} found in test set.')

date_target   = datetime_use.iloc[idx_target].reset_index(drop=True)
npp_target    = npp_use[idx_target]
fusion_target = fusion_use[idx_target]
Mw_target     = M_use[idx_target]
print(f'number of target events (M >= {target_M}):', len(idx_target))

L = len(Mw_target)
etas_target = etas_temp_rate[-L:]

print('ETAS target length  :', len(etas_target))
print('NPP target length   :', len(npp_target))
print('Fusion target length:', len(fusion_target))

# 横坐标：从第0个目标事件开始，按天计数
x_days = ((date_target - date_target.iloc[0]).dt.total_seconds() / 86400.0).to_numpy(dtype=float)

# ===================== 找到 Mw 最大的事件 =====================
idx_mw_max = np.argmax(Mw_target)
x_mw_max = x_days[idx_mw_max]
mw_max_event = Mw_target[idx_mw_max]
date_mw_max = date_target.iloc[idx_mw_max]

print(f'Max Mw event: Mw={mw_max_event:.4f}, x_days={x_mw_max:.4f}, date={date_mw_max}')

# ===================== 左轴范围（temporal rate） =====================
all_min = min(np.min(etas_target), np.min(npp_target), np.min(fusion_target))
all_max = max(np.max(etas_target), np.max(npp_target), np.max(fusion_target))
pad = 0.05 * (all_max - all_min) if all_max > all_min else 1e-6
ymin = all_min - pad
ymax = all_max + pad

# ===================== 右轴范围（Mw） =====================
mw_min = min(target_M - 0.2, np.min(Mw_target) - 0.1)
mw_max = np.max(Mw_target) + 0.3

# 修改右侧Y轴的刻度颜色
# ===================== 绘图：主图 =====================
fig, axes = plt.subplots(3, 1, figsize=(6.9, 5.3), sharex=True)

axes[0].plot(x_days, etas_target,
             label=f'ETAS({plot_etas_Mcut})', color=colors['ETAS'],
             linestyle=linestyles['ETAS'],
             linewidth=LINEWIDTH, alpha=LINE_ALPHA,
             marker='o', markersize=4.5,
             markerfacecolor='none', markeredgecolor=colors['ETAS'])

axes[1].plot(x_days, npp_target,
             label=f'NPP({plot_Mcut})', color=colors['NPP'],
             linestyle=linestyles['NPP'],
             linewidth=LINEWIDTH, alpha=LINE_ALPHA,
             marker='o', markersize=4.5,
             markerfacecolor='none', markeredgecolor=colors['NPP'])

axes[2].plot(x_days, fusion_target,
             label=f'Fusion({plot_Mcut})', color=colors['Fusion'],
             linestyle=linestyles['Fusion'],
             linewidth=LINEWIDTH, alpha=LINE_ALPHA,
             marker='o', markersize=4.5,
             markerfacecolor='none', markeredgecolor=colors['Fusion'])

# ===== 在主图中添加zoom区域的框 =====
# campotosto: 5.5~6.0; Norcia: 0.8~2.0; Visso: 25.5~27.0
# campotosto: 50; Norcia: 200; Visso: 75
if plot_earthquake == "Campotosto":
    zoom_xmin = 5.5
    zoom_xmax = 6.05
    REF_Y = 50
    xlim_gap = 0.1
elif plot_earthquake == "Norcia":
    zoom_xmin = 0.8
    zoom_xmax = 1.6
    REF_Y = 200        
    xlim_gap = 0.2
elif plot_earthquake == "Visso":
    zoom_xmin = 25.5
    zoom_xmax = 27.0
    REF_Y = 75  
    xlim_gap = 0.5

rect_width = zoom_xmax - zoom_xmin
rect_height = ymax - ymin

# 在每个子图上添加矩形框
for ax in axes:
    ax.add_patch(plt.Rectangle(
        (zoom_xmin, ymin),  # 左下角坐标
        rect_width,          # 宽度
        rect_height,         # 高度
        linewidth=1,         # 线宽
        edgecolor='#7b7b7b', # 边框颜色
        facecolor='#7b7b7b', # 填充颜色
        alpha=0.3           # 透明度
    ))

for ax in axes:
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Temp. Rate')
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    ax.tick_params(axis='both', which='both', direction='in',
                   bottom=True, top=True, left=True, right=False)
# ===== 每个子图都叠加 Mw 右轴 =====
ax_mw0 = axes[0].twinx()
ax_mw0.scatter(
    x_mw_max, mw_max_event,
    marker='*', s=140,
    color=colors['star'], edgecolors='gray', linewidths=0.6, 
    label=f'Max Mw={mw_max_event:.1f}', zorder=5
)
ax_mw0.set_ylabel('Mw', color='#7b7b7b') 
ax_mw0.set_ylim(mw_min, mw_max)
ax_mw0.tick_params(axis='y', which='both', direction='in',
                   right=True, labelcolor='#7b7b7b')

ax_mw1 = axes[1].twinx()
ax_mw1.scatter(
    x_mw_max, mw_max_event,
    marker='*', s=140,
    color=colors['star'], edgecolors='gray', linewidths=0.6, 
    label=f'Max Mw={mw_max_event:.1f}', zorder=5
)
ax_mw1.set_ylabel('Mw', color='#7b7b7b')
ax_mw1.set_ylim(mw_min, mw_max)
ax_mw1.tick_params(axis='y', which='both', direction='in',
                   right=True, labelcolor='#7b7b7b')


ax_mw2 = axes[2].twinx()
ax_mw2.scatter(
    x_mw_max, mw_max_event,
    marker='*', s=140,
    color=colors['star'], edgecolors='gray', linewidths=0.6, 
    label=f'Max Mw={mw_max_event:.1f}', zorder=5
)
ax_mw2.set_ylabel('Mw', color='#7b7b7b') 
ax_mw2.set_ylim(mw_min, mw_max)
ax_mw2.tick_params(axis='y', which='both', direction='in',
                   right=True, labelcolor='#7b7b7b')


# ===== 主图图例 =====
# 第一个子图：显示 ETAS + Mw + Max Mw
lines1, labels1 = axes[0].get_legend_handles_labels()
lines2, labels2 = ax_mw0.get_legend_handles_labels()
leg0 = axes[0].legend(
    lines1 + lines2, labels1 + labels2,
    loc='upper right', frameon=False
)
leg0.get_frame().set_alpha(0.0)
leg0.get_frame().set_linewidth(0.0)

# 第二个子图：只显示 NPP
lines1, labels1 = axes[1].get_legend_handles_labels()
leg1 = axes[1].legend(
    lines1, labels1,
    loc='upper right', frameon=False
)
leg1.get_frame().set_alpha(0.0)
leg1.get_frame().set_linewidth(0.0)

# 第三个子图：只显示 Fusion
lines1, labels1 = axes[2].get_legend_handles_labels()
leg2 = axes[2].legend(
    lines1, labels1,
    loc='upper right', frameon=False
)
leg2.get_frame().set_alpha(0.0)
leg2.get_frame().set_linewidth(0.0)

axes[-1].set_xlabel('Time(Days)', fontsize=12)

fig.suptitle(f'{plot_earthquake}', y=0.98, fontsize=12)
fig.tight_layout()

out_dir = './figures'
os.makedirs(out_dir, exist_ok=True)

fname_base = f'temp_rate_zhang_{plot_earthquake}_Mcut{plot_Mcut}_Mt{target_M}'

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

# ===================== 局部放大图 =====================
mask_zoom = (x_days >= zoom_xmin) & (x_days <= zoom_xmax)

x_zoom = x_days[mask_zoom]
etas_zoom = etas_target[mask_zoom]
npp_zoom = npp_target[mask_zoom]
fusion_zoom = fusion_target[mask_zoom]
Mw_zoom = Mw_target[mask_zoom]

# ===== 判断最大 Mw 事件是否落在 zoom 区间 =====
has_max_in_zoom = mask_zoom[idx_mw_max]
if has_max_in_zoom:
    x_mw_max_zoom = x_days[idx_mw_max]
    mw_max_event_zoom = Mw_target[idx_mw_max]

if len(x_zoom) == 0:
    raise ValueError(f'No target events found in zoom window [{zoom_xmin}, {zoom_xmax}].')

def beautify_zoom_axes(ax, ydata, ref_y=None):
    y_min = np.min(ydata)
    y_max = np.max(ydata)

    if ref_y is not None:
        y_min = min(y_min, ref_y)
        y_max = max(y_max, ref_y)

    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1e-6

    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_ylabel('Temp. Rate')
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    ax.tick_params(axis='both', which='both', direction='in',
               bottom=True, top=False, left=True, right=True)
    ax.spines['top'].set_visible(False)

fig_zoom, axes_zoom = plt.subplots(
    3, 1, figsize=(6.9, 5.3), sharex=True,
    gridspec_kw={'hspace': 0.12}
)

# ===== 先画三条 zoom 曲线 =====
axes_zoom[0].plot(
    x_zoom, etas_zoom,
    label=f'ETAS(Mcut={plot_etas_Mcut})', color=colors['ETAS'],
    linestyle=linestyles['ETAS'],
    linewidth=LINEWIDTH, alpha=LINE_ALPHA,
    marker='o', markersize=4.5,
    markerfacecolor='none', markeredgecolor=colors['ETAS']
)

axes_zoom[1].plot(
    x_zoom, npp_zoom,
    label=f'NPP(Mcut={plot_Mcut})', color=colors['NPP'],
    linestyle=linestyles['NPP'],
    linewidth=LINEWIDTH, alpha=LINE_ALPHA,
    marker='o', markersize=4.5,
    markerfacecolor='none', markeredgecolor=colors['NPP']
)

axes_zoom[2].plot(
    x_zoom, fusion_zoom,
    label=f'Fusion(Mcut={plot_Mcut})', color=colors['Fusion'],
    linestyle=linestyles['Fusion'],
    linewidth=LINEWIDTH, alpha=LINE_ALPHA,
    marker='o', markersize=4.5,
    markerfacecolor='none', markeredgecolor=colors['Fusion']
)

# ===== 每个子图单独设置 y 轴，并加入统一参考线 =====
beautify_zoom_axes(axes_zoom[0], etas_zoom, REF_Y)
beautify_zoom_axes(axes_zoom[1], npp_zoom, REF_Y)
beautify_zoom_axes(axes_zoom[2], fusion_zoom, REF_Y)

for ax in axes_zoom:
    ax.axhline(REF_Y, color='black', linestyle='--', linewidth=1.0, alpha=0.9)

    ax.set_xticks(np.arange(min(x_days), max(x_days), xlim_gap))  # 设置X轴刻度间隔为0.2
    ax.xaxis.set_major_locator(ticker.MultipleLocator(xlim_gap))   # 确保主要刻度间隔为0.2
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: '{:.1f}'.format(val)))  # 格式化为x.0, x.2


# ===== 只保留一套右轴 =====
ax_mwz0 = axes_zoom[0].twinx()
if has_max_in_zoom:
    ax_mwz0.scatter(
        x_mw_max_zoom, mw_max_event_zoom,
        marker='*', s=140,
        color=colors['star'], edgecolors='gray', linewidths=0.6,
        label='Max Mw', zorder=5
    )
ax_mwz0.set_ylabel('Mw')
ax_mwz0.set_ylim(mw_min, mw_max)
ax_mwz0.tick_params(axis='y', which='both', direction='in', right=True)
ax_mwz0.spines['top'].set_visible(False)

ax_mwz1 = axes_zoom[1].twinx()
if has_max_in_zoom:
    ax_mwz1.scatter(
        x_mw_max_zoom, mw_max_event_zoom,
        marker='*', s=140,
        color=colors['star'], edgecolors='gray', linewidths=0.6,
        label='Max Mw', zorder=5
    )
ax_mwz1.set_ylabel('Mw')
ax_mwz1.set_ylim(mw_min, mw_max)
ax_mwz1.tick_params(axis='y', which='both', direction='in', right=True)
ax_mwz1.spines['top'].set_visible(False)

ax_mwz2 = axes_zoom[2].twinx()
if has_max_in_zoom:
    ax_mwz2.scatter(
        x_mw_max_zoom, mw_max_event_zoom,
        marker='*', s=140,
        color=colors['star'], edgecolors='gray', linewidths=0.6,
        label='Max Mw', zorder=5
    )
ax_mwz2.set_ylabel('Mw')
ax_mwz2.set_ylim(mw_min, mw_max)
ax_mwz2.tick_params(axis='y', which='both', direction='in', right=True)
ax_mwz2.spines['top'].set_visible(False)

axes_zoom[-1].set_xlabel('Time(Days)', fontsize=12)

fig_zoom.suptitle(f'{plot_earthquake}', fontsize=12)
fig_zoom.align_ylabels(axes_zoom)
fig_zoom.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.10)

fname_zoom = f'temp_rate_zhang_zoom_{plot_earthquake}_Mcut{plot_Mcut}_Mt{target_M}'

out_svg_zoom = os.path.join(out_dir, f'{fname_zoom}.svg')
out_pdf_zoom = os.path.join(out_dir, f'{fname_zoom}.pdf')
out_png_zoom = os.path.join(out_dir, f'{fname_zoom}.png')

plt.savefig(out_svg_zoom, transparent=True, bbox_inches='tight')
plt.savefig(out_pdf_zoom, transparent=True, bbox_inches='tight')
plt.savefig(out_png_zoom, facecolor='white', bbox_inches='tight')
plt.show()

print("Zoom figure saved to:")
print(out_svg_zoom)
print(out_pdf_zoom)
print(out_png_zoom)