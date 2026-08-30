import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

# =========================
# 全局样式
# =========================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "axes.unicode_minus": False,
    "svg.fonttype": "none",   # 保存 SVG 时保留文字可编辑
    "figure.dpi": 300,
    "savefig.dpi": 600,
})

# =========================
# 直线：经过 (5,2) 和 (7,1)
# y = 4.5 - 0.5x
# =========================
x = np.linspace(3.0, 8.0, 400)
y = 4.5 - 0.5 * x

# 两个指定交点
x1, y1 = 5, 2
x2, y2 = 7, 1

# =========================
# 创建画布
# =========================
fig, ax = plt.subplots(figsize=(4.2, 3.2))

# 红色主线
ax.plot(x, y, color="#e53935", lw=2.0, solid_capstyle="round", zorder=3)

# 灰色辅助虚线
ax.vlines([x1, x2], ymin=0, ymax=[y1, y2],
          colors="0.7", linestyles=(0, (3, 3)), lw=1.0, zorder=1)
ax.hlines([y1, y2], xmin=0, xmax=[x1, x2],
          colors="0.7", linestyles=(0, (3, 3)), lw=1.0, zorder=1)

# =========================
# 坐标范围与刻度
# =========================
ax.set_xlim(3.0, 8.2)
ax.set_ylim(0.0, 3.4)

ax.set_xticks([5, 7, 8])
ax.set_yticks([1, 2, 3])

# 只保留左、下刻度，但不显示原始边框
for spine in ax.spines.values():
    spine.set_visible(False)

ax.tick_params(axis='both', which='major',
               direction='out', length=3.5, width=1.0,
               colors='0.25', labelsize=11)

# =========================
# 手动画箭头坐标轴
# =========================
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

# x轴箭头
ax.annotate(
    "", xy=(xmax, 0), xytext=(xmin, 0),
    arrowprops=dict(
        arrowstyle="-|>",
        lw=1.2,
        edgecolor="0.25",
        facecolor="0.25",
        shrinkA=0,
        shrinkB=0,
        mutation_scale=10
    ),
    clip_on=False,
    zorder=4
)

# y轴箭头
ax.annotate(
    "", xy=(xmin, ymax), xytext=(xmin, 0),
    arrowprops=dict(
        arrowstyle="-|>",
        lw=1.2,
        edgecolor="0.25",
        facecolor="0.25",
        shrinkA=0,
        shrinkB=0,
        mutation_scale=10
    ),
    clip_on=False,
    zorder=4
)

# =========================
# 标签
# =========================
ax.set_xlabel("Magnitude", fontsize=13, color="0.2", labelpad=4)
ax.set_ylabel("Log of annual number of earthquakes", fontsize=13, color="0.2", labelpad=10)

# 让 xlabel 稍微居中靠下
ax.xaxis.set_label_coords(0.55, -0.14)
ax.yaxis.set_label_coords(-0.06, 0.5)

# =========================
# 右上角文字
# =========================
ax.text(7.15, 3.02, "G-R law", ha="left", va="bottom",
        fontsize=14, fontweight="bold", color="0.2")
ax.text(6.8, 2.72, r"Log $N$ = a - b$M$", ha="left", va="bottom",
        fontsize=13, color="0.2")

# =========================
# 灰色小箭头（示意）
# =========================
ax.annotate(
    "", xy=(6.75, 1.35), xytext=(6.25, 1.7),
    arrowprops=dict(
        arrowstyle="simple",
        fc="0.75",
        ec="0.75",
        alpha=0.9,
        mutation_scale=18
    ),
    zorder=2
)

fig.tight_layout(pad=0.35)

out_dir = './figures'
os.makedirs(out_dir, exist_ok=True)

fname = 'diag_GR_law'
fig.savefig(os.path.join(out_dir, f'{fname}.svg'), transparent=True)

plt.show()

# print(svg_path)
# print(png_path)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "figure.dpi": 300,
    "savefig.dpi": 600,
})

# -------------------------
# Modified Omori law
# -------------------------
t = np.linspace(0, 8, 600)
K = 2.5
c = 0.22
p = 1.15
rate = K / (t + c)**p

fig, ax = plt.subplots(figsize=(4.2, 3.2))

# New color style based on the user's reference image
COL_CURVE = "#d94841"   # red line
COL_AXIS  = "#4a4748"   # dark gray axes
COL_DASH  = "#a9a9a9"   # light gray dashed lines
COL_ARROW = "#bdbdbd"   # helper arrow
COL_TEXT  = "#333333"   # dark text

# Main curve
ax.plot(t, rate, color=COL_CURVE, lw=2.3, solid_capstyle="round", zorder=3)

# Reference dashed lines
t1 = 1.2
t2 = 2.4
r1 = K / (t1 + c)**p
r2 = K / (t2 + c)**p

ax.vlines([t1, t2], ymin=0, ymax=[r1, r2],
          colors=COL_DASH, linestyles=(0, (3, 3)), lw=1.1, zorder=1)
ax.hlines([r1, r2], xmin=0, xmax=[t1, t2],
          colors=COL_DASH, linestyles=(0, (3, 3)), lw=1.1, zorder=1)

# Axis limits
ax.set_xlim(0, 8.2)
ax.set_ylim(0, rate.max() * 1.10)

# Remove spines and ticks
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

# Arrow axes
ax.annotate(
    "", xy=(xmax, 0), xytext=(0, 0),
    arrowprops=dict(
        arrowstyle="-|>",
        lw=1.4,
        edgecolor=COL_AXIS,
        facecolor=COL_AXIS,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=13
    ),
    clip_on=False, zorder=4
)
ax.annotate(
    "", xy=(0, ymax), xytext=(0, 0),
    arrowprops=dict(
        arrowstyle="-|>",
        lw=1.4,
        edgecolor=COL_AXIS,
        facecolor=COL_AXIS,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=13
    ),
    clip_on=False, zorder=4
)

# Labels
ax.text(-0.08, ymax * 1.02, "Aftershock rate",
        ha="left", va="bottom", fontsize=13, color=COL_TEXT)
ax.text(xmax * 0.88, -0.06 * ymax, "time",
        ha="center", va="top", fontsize=12, color=COL_TEXT)

# Title and formula
ax.text(4.95, ymax * 0.91, "Omori law",
        ha="left", va="bottom", fontsize=14, fontweight="bold", color=COL_TEXT)
ax.text(4.05, ymax * 0.78, r"$n(t)=\dfrac{K}{(t+c)^p}$",
        ha="left", va="bottom", fontsize=13, color=COL_TEXT)

# Gray helper arrow
ax.annotate(
    "", xy=(2.95, 1.75), xytext=(2.45, 2.15),
    arrowprops=dict(
        arrowstyle="simple",
        fc=COL_ARROW,
        ec=COL_ARROW,
        alpha=0.9,
        mutation_scale=18
    ),
    zorder=2
)

plt.tight_layout()

# out_dir = './figures'
# os.makedirs(out_dir, exist_ok=True)

# fname = 'diag_omori_law'
# fig.savefig(os.path.join(out_dir, f'{fname}.svg'), transparent=True)

plt.show()
