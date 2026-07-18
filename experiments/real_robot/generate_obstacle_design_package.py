#!/usr/bin/env python3
"""Generate supplier-facing drawings for the 6.5 m real-robot obstacle kit.

The YAML file is the single source of truth.  This script produces raster/vector
drawings, one multi-page PDF, CSV placement tables and an editable multi-page
draw.io document.  It does not change controller or simulator configuration.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches, transforms
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "real_robot" / "obstacle_kit_6p5m.yaml"
DEFAULT_OUTPUT = ROOT / "docs" / "real_robot" / "obstacle_kit_6p5m"
CJK_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("/mnt/c/Windows/Fonts/msyh.ttc"),
)

COLORS = {
    "wall": "#D55E00",
    "prism": "#CC79A7",
    "dynamic": "#E69F00",
    "floor": "#56B4E9",
    "start": "#0072B2",
    "goal": "#009E73",
    "active": "#4D4D4D",
}


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("design config must contain a mapping")
    return config


def _validate_config(config: dict) -> None:
    assumptions = config["design_assumptions"]
    height = float(assumptions["obstacle_height"])
    beam_min, beam_max = map(float, assumptions["compatible_lidar_beam_height_range"])
    margin = float(assumptions["minimum_lidar_vertical_margin"])
    if beam_min < margin or beam_max + margin > height:
        raise ValueError("configured obstacle height does not cover the declared LiDAR beam range")
    if str(config["materials"]["wall_and_base"]).startswith("transparent"):
        raise ValueError("transparent acrylic is prohibited for a LaserScan obstacle")
    for key, module in config["modules"].items():
        if module["kind"] == "wall":
            length, base_depth, module_height = map(float, module["dimensions"])
            if abs(base_depth - float(module["base_depth"])) > 1e-9:
                raise ValueError("%s plan depth must equal the fabricated base depth" % key)
            if abs(module_height - height) > 1e-9:
                raise ValueError("%s height must match the static obstacle height" % key)
            panel_length, panel_height, panel_thickness = map(float, module["panel_dimensions"])
            base_length, fabricated_base_depth, base_thickness = map(float, module["base_dimensions"])
            if abs(panel_length - length) > 1e-9 or abs(base_length - length) > 1e-9:
                raise ValueError("%s panel and base lengths must match the finished width" % key)
            if abs(fabricated_base_depth - base_depth) > 1e-9:
                raise ValueError("%s base blank depth must match the finished footprint" % key)
            if abs(panel_height + base_thickness - module_height) > 1e-9:
                raise ValueError("%s panel blank plus base thickness must equal finished height" % key)
            if abs(panel_thickness - float(module["panel_thickness"])) > 1e-9:
                raise ValueError("%s panel thickness metadata is inconsistent" % key)
            if int(module["gusset_count"]) != 2 * int(module["gusset_pair_count"]):
                raise ValueError("%s gussets must be installed as front/back pairs" % key)
            horizontal_leg = float(module["gusset_horizontal_leg"])
            if horizontal_leg + panel_thickness / 2.0 > base_depth / 2.0 - 0.005:
                raise ValueError("%s gusset does not leave 5 mm base-edge clearance" % key)
            stations = [float(value) for value in module["gusset_station_x"]]
            if len(stations) != int(module["gusset_pair_count"]):
                raise ValueError("%s gusset station count is inconsistent" % key)
            if any(value <= 0.0 or value >= length for value in stations):
                raise ValueError("%s gusset stations must be inside the panel ends" % key)
        if module["kind"] == "octagonal_prism" and int(module.get("sides", 0)) != 8:
            raise ValueError("%s must remain an eight-sided prism for the frozen design" % key)


def _setup_font() -> None:
    for font_path in CJK_FONT_CANDIDATES:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "figure.titlesize": 16,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _dimension(ax, p1, p2, label, offset=(0.0, 0.0), color="#333333"):
    x1, y1 = p1[0] + offset[0], p1[1] + offset[1]
    x2, y2 = p2[0] + offset[0], p2[1] + offset[1]
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="<->", color=color, linewidth=1.0),
    )
    ax.text((x1 + x2) / 2.0, (y1 + y2) / 2.0, label, ha="center", va="bottom", color=color)


def _draw_cuboid(ax, x0, x1, y0, y1, z0, z1, facecolor, edgecolor="#7A3100", alpha=0.86):
    vertices = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    faces = [
        [vertices[i] for i in (0, 1, 2, 3)],
        [vertices[i] for i in (4, 5, 6, 7)],
        [vertices[i] for i in (0, 1, 5, 4)],
        [vertices[i] for i in (1, 2, 6, 5)],
        [vertices[i] for i in (2, 3, 7, 6)],
        [vertices[i] for i in (3, 0, 4, 7)],
    ]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=facecolor, edgecolors=edgecolor, linewidths=0.7, alpha=alpha))


def _draw_triangular_prism(ax, x_center, yz_points, thickness=6.0, facecolor="#FFD2B8"):
    x0, x1 = x_center - thickness / 2.0, x_center + thickness / 2.0
    front = [(x0, y, z) for y, z in yz_points]
    back = [(x1, y, z) for y, z in yz_points]
    faces = [front, back]
    for index in range(3):
        nxt = (index + 1) % 3
        faces.append([front[index], front[nxt], back[nxt], back[index]])
    ax.add_collection3d(Poly3DCollection(faces, facecolors=facecolor, edgecolors="#7A3100", linewidths=0.7, alpha=0.94))


def _format_wall_3d_axis(ax, length, title):
    ax.set_xlim(-25, length + 25)
    ax.set_ylim(-185, 185)
    ax.set_zlim(0, 485)
    ax.set_box_aspect((1.7, 1.0, 1.25))
    ax.view_init(elev=19, azim=-58)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.set_title(title, fontsize=13, fontweight="bold", pad=6)


def _draw_wall_3d(ax, module, exploded=False):
    length, base_depth, _ = [1000.0 * float(value) for value in module["dimensions"]]
    _, panel_height, panel_thickness = [1000.0 * float(value) for value in module["panel_dimensions"]]
    _, _, base_thickness = [1000.0 * float(value) for value in module["base_dimensions"]]
    horizontal_leg = 1000.0 * float(module["gusset_horizontal_leg"])
    vertical_leg = 1000.0 * float(module["gusset_vertical_leg"])
    stations = [1000.0 * float(value) for value in module["gusset_station_x"]]

    _draw_cuboid(ax, 0, length, -base_depth / 2.0, base_depth / 2.0, 0, base_thickness, "#F6D5C1")
    panel_z = base_thickness + (105.0 if exploded else 0.0)
    _draw_cuboid(
        ax,
        0,
        length,
        -panel_thickness / 2.0,
        panel_thickness / 2.0,
        panel_z,
        panel_z + panel_height,
        "#F2A477",
    )
    gusset_y_shift = 48.0 if exploded else 0.0
    gusset_z_shift = 30.0 if exploded else 0.0
    z0 = base_thickness + gusset_z_shift
    for station in stations:
        _draw_triangular_prism(
            ax,
            station,
            [
                (panel_thickness / 2.0 + gusset_y_shift, z0),
                (panel_thickness / 2.0 + gusset_y_shift + horizontal_leg, z0),
                (panel_thickness / 2.0 + gusset_y_shift, z0 + vertical_leg),
            ],
            thickness=panel_thickness,
        )
        _draw_triangular_prism(
            ax,
            station,
            [
                (-panel_thickness / 2.0 - gusset_y_shift, z0),
                (-panel_thickness / 2.0 - gusset_y_shift - horizontal_leg, z0),
                (-panel_thickness / 2.0 - gusset_y_shift, z0 + vertical_leg),
            ],
            thickness=panel_thickness,
        )

    if exploded:
        ax.text(length * 0.52, 8, panel_z + panel_height * 0.60, "A  主墙板\n600×344×6", ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(length * 0.55, -92, 8, "B  底板\n600×160×6", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(length * 0.52, 145, z0 + 58, "C  加强肋\n前后成对，共6件", ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text2D(0.01, 0.77, "A  主墙板：1件", transform=ax.transAxes, fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", facecolor="#F2A477", edgecolor="#7A3100"))
        ax.text2D(0.01, 0.66, "B  底板：1件", transform=ax.transAxes, fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", facecolor="#F6D5C1", edgecolor="#7A3100"))
        ax.text2D(0.01, 0.55, "C  加强肋：6件", transform=ax.transAxes, fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFD2B8", edgecolor="#7A3100"))
    else:
        ax.text(length * 0.50, 5, panel_height * 0.63, "成品总高 350\n板长 600\n底板深 160", ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text2D(0.02, 0.82, "成品外形：600 × 160 × 350", transform=ax.transAxes, fontsize=9.5, fontweight="bold", zorder=100, clip_on=False, bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#7A3100"))
        ax.text2D(0.02, 0.72, "C：前后成对，共6件", transform=ax.transAxes, fontsize=9.5, fontweight="bold", zorder=100, clip_on=False, bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#7A3100"))


def _module_shape(config: dict, item: dict):
    module = config["modules"][item["module"]]
    kind = module["kind"]
    if kind == "wall":
        length, thickness, _ = module["dimensions"]
        return kind, float(length), float(thickness)
    if kind in ("octagonal_prism", "dynamic_octagonal_shell"):
        diameter = float(module["diameter"])
        return kind, diameter, diameter
    raise ValueError("unsupported obstacle module: %s" % kind)


def _draw_obstacle(ax, config: dict, item: dict, annotate=True):
    kind, width, depth = _module_shape(config, item)
    cx, cy = map(float, item["center"])
    yaw = float(item.get("yaw_deg", 0.0))
    if kind == "wall":
        shape = patches.Rectangle(
            (cx - width / 2.0, cy - depth / 2.0),
            width,
            depth,
            facecolor=COLORS["wall"],
            edgecolor="#7A3100",
            linewidth=1.2,
            alpha=0.82,
        )
        shape.set_transform(transforms.Affine2D().rotate_deg_around(cx, cy, yaw) + ax.transData)
    else:
        color = COLORS["dynamic"] if kind == "dynamic_octagonal_shell" else COLORS["prism"]
        shape = patches.RegularPolygon(
            (cx, cy),
            numVertices=int(config["modules"][item["module"]].get("sides", 8)),
            radius=width / 2.0,
            orientation=math.radians(22.5),
            facecolor=color,
            edgecolor="#6B3659",
            linewidth=1.2,
            alpha=0.85,
        )
    ax.add_patch(shape)
    if annotate:
        ax.text(cx, cy, item["id"], ha="center", va="center", fontsize=7, fontweight="bold")


def _draw_field(ax, config: dict, title: str, show_path=True):
    field = config["field"]
    width, height = field["size"]
    xmin, xmax, ymin, ymax = field["active_bounds"]
    ax.add_patch(patches.Rectangle((0, 0), width, height, fill=False, edgecolor="#111111", linewidth=2.0))
    ax.add_patch(
        patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            facecolor="#F7F7F7",
            edgecolor=COLORS["active"],
            linewidth=1.5,
            linestyle="--",
        )
    )
    start = field["start"]
    goal = field["goal"]
    radius = float(config["design_assumptions"]["robot_collision_radius"])
    ax.add_patch(patches.Circle(start, radius, facecolor=COLORS["start"], edgecolor="white", linewidth=1.2))
    ax.arrow(start[0], start[1], 0.34, 0.0, width=0.025, head_width=0.13, color=COLORS["start"])
    ax.plot(goal[0], goal[1], marker="*", markersize=17, color=COLORS["goal"], markeredgecolor="#005B43")
    ax.text(start[0], start[1] - 0.42, "S 起点", ha="center", color=COLORS["start"], fontweight="bold")
    ax.text(goal[0], goal[1] + 0.30, "G 目标", ha="center", color=COLORS["goal"], fontweight="bold")
    if show_path:
        ax.plot([start[0], goal[0]], [start[1], goal[1]], color="#999999", linestyle=":", linewidth=1.2)
    ax.text(0.5, 6.15, "1.0 m 周边安全带（人员/急停区）", fontsize=8, color="#555555")
    ax.set_xlim(-0.15, width + 0.15)
    ax.set_ylim(-0.15, height + 0.15)
    ax.set_aspect("equal")
    ax.set_xticks([i / 2.0 for i in range(14)])
    ax.set_yticks([i / 2.0 for i in range(14)])
    ax.grid(True, color="#DDDDDD", linewidth=0.5)
    ax.set_xlabel("场地 X / m")
    ax.set_ylabel("场地 Y / m")
    ax.set_title(title, fontweight="bold")


def _scene_figure(config: dict, scene_name: str):
    scene = config["scenes"][scene_name]
    fig, ax = plt.subplots(figsize=(8.27, 11.69), constrained_layout=True)
    _draw_field(ax, config, "%s（%s）" % (scene["display_name"], scene_name))
    for item in scene.get("obstacles", []):
        _draw_obstacle(ax, config, item)
    dynamic = scene.get("dynamic_obstacle")
    if dynamic:
        x1, y1 = dynamic["path_start"]
        x2, y2 = dynamic["path_end"]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="<->", color=COLORS["dynamic"], linewidth=3.0),
        )
        middle = {"id": dynamic["id"], "module": dynamic["module"], "center": [(x1 + x2) / 2, (y1 + y2) / 2]}
        _draw_obstacle(ax, config, middle)
        ax.text((x1 + x2) / 2, y1 + 0.35, "横穿轨迹：0.10 / 0.20 m/s", ha="center", color="#9C6500")
    floor_patch = scene.get("floor_patch")
    if floor_patch:
        module = config["modules"][floor_patch["module"]]
        length, width, _ = module["dimensions"]
        cx, cy = floor_patch["center"]
        yaw = float(floor_patch["yaw_deg"])
        patch = patches.Rectangle(
            (cx - length / 2, cy - width / 2), length, width,
            facecolor=COLORS["floor"], edgecolor="#0072B2", hatch="//", alpha=0.28,
        )
        patch.set_transform(transforms.Affine2D().rotate_deg_around(cx, cy, yaw) + ax.transData)
        ax.add_patch(patch)
        ax.text(cx, cy, "可更换地面材料", ha="center", va="center", color="#00598C")
    note = "机器人安全包络 D=0.50 m；最小通行宽度 0.85 m；正式加工前必须按实车复测"
    ax.text(
        3.25, 0.34, note, ha="center", va="center", fontsize=8, color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#AAAAAA", alpha=0.9),
    )
    return fig


def _site_figure(config: dict):
    fig, ax = plt.subplots(figsize=(8.27, 11.69), constrained_layout=True)
    _draw_field(ax, config, "6.5 m × 6.5 m 场地与坐标系统", show_path=False)
    origin = config["field"]["experiment_origin_in_field"]
    ax.plot(origin[0], origin[1], marker="+", markersize=18, markeredgewidth=2.5, color="#000000")
    ax.text(origin[0] + 0.15, origin[1] + 0.15, "实验坐标原点 (0,0)")
    _dimension(ax, (0, -0.03), (6.5, -0.03), "6500 mm", offset=(0, -0.08))
    _dimension(ax, (-0.03, 0), (-0.03, 6.5), "6500 mm", offset=(-0.08, 0))
    ax.text(
        3.25,
        0.45,
        "场地坐标 = 实验坐标 + (1.5, 1.5) m\n当前 goal=(3,3) 对应场地 (4.5,4.5) m",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#777777"),
    )
    return fig


def _wall_figure(config: dict):
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3), constrained_layout=True)
    fig.suptitle("低矮亚克力墙体模块加工图（单位：mm）", fontweight="bold")
    for ax, key in zip(axes[0], ("W600", "W300")):
        module = config["modules"][key]
        length, _, height = [1000 * float(v) for v in module["dimensions"]]
        panel_t = 1000 * float(module["panel_thickness"])
        gusset_leg = 1000 * float(module["gusset_leg"])
        panel_y = 100
        ax.add_patch(
            patches.FancyBboxPatch(
                (100, panel_y), length, height,
                boxstyle="round,pad=0,rounding_size=20",
                facecolor="#F5B18B", edgecolor="#7A3100", linewidth=1.5,
            )
        )
        gusset_x = [100 + 35, 100 + length - 35]
        if int(module["gusset_count"]) == 3:
            gusset_x.insert(1, 100 + length / 2)
        panel_center = 100 + length / 2
        for xpos in gusset_x:
            if xpos > panel_center:
                triangle = [(xpos + panel_t / 2, panel_y), (xpos - gusset_leg * 0.55, panel_y), (xpos + panel_t / 2, panel_y + gusset_leg)]
            else:
                triangle = [(xpos - panel_t / 2, panel_y), (xpos + gusset_leg * 0.55, panel_y), (xpos - panel_t / 2, panel_y + gusset_leg)]
            ax.add_patch(
                patches.Polygon(
                    triangle,
                    closed=True, facecolor="#FFD2B8", edgecolor="#7A3100", linewidth=0.8,
                )
            )
        ax.text(100 + length / 2, panel_y + height * 0.58, key, ha="center", va="center", fontsize=16, fontweight="bold")
        _dimension(ax, (100, 70), (100 + length, 70), "%d" % length)
        _dimension(ax, (70, panel_y), (70, panel_y + height), "%d" % height)
        ax.set_xlim(0, max(800, length + 200)); ax.set_ylim(0, 560); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("%s 正视图：%d mm PMMA / 加强肋 %d 件" % (key, panel_t, module["gusset_count"]))
    ax = axes[1, 0]
    ax.add_patch(patches.Rectangle((120, 160), 600, 160, facecolor="#F9D5C1", edgecolor="#7A3100"))
    ax.add_patch(patches.Rectangle((120, 237), 600, 6, facecolor="#D55E00", edgecolor="#7A3100"))
    for xpos in (155, 420, 685):
        ax.add_patch(patches.Rectangle((xpos - 3, 180), 6, 120, facecolor="#FFD2B8", edgecolor="#7A3100"))
    _dimension(ax, (120, 125), (720, 125), "600")
    _dimension(ax, (80, 160), (80, 320), "160")
    ax.text(420, 340, "顶视图：600×160×6 底板；主板居中插接/粘接", ha="center", va="center", fontweight="bold")
    ax.text(420, 95, "底面贴 2 mm 防滑橡胶；任何连接件不得超出 160 mm 轮廓", ha="center", fontsize=9)
    ax.set_xlim(0, 850); ax.set_ylim(50, 410); ax.set_aspect("equal"); ax.axis("off")
    ax = axes[1, 1]
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        "结构与安全要求\n\n"
        "• 主板/底板：6 mm 浇铸 PMMA；总高 350 mm，底座深 160 mm。\n"
        "• W600 配 3 个、W300 配 2 个 120×120×6 mm 三角加强肋。\n"
        "• 插槽按实测板厚+0.2～0.4 mm 试配；正式件不得强行压入。\n"
        "• 外露上角 R≥20 mm；全部边缘倒钝并安装透明硅胶 U 形护边。\n"
        "• 双面使用不透明哑光橙色板或哑光贴膜；禁用透明/镜面/亮黑。\n"
        "• 底面贴防滑垫；若首件推力试验位移>10 mm，再加低位内置配重。\n"
        "• 直连/90°/T 形连接件不得扩大既定 160 mm 平面占地。\n"
        "• 正反面标注模块 ID、几何中心线和朝向线。",
        va="top",
        fontsize=11,
        linespacing=1.55,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFF8F1", edgecolor="#D55E00"),
    )
    return fig


def _wall_assembly_figure(config: dict):
    module = config["modules"]["W600"]
    fig = plt.figure(figsize=(11.7, 8.3), constrained_layout=True)
    fig.suptitle("低矮亚克力墙体——供应商装配总图（单位：mm）", fontsize=17, fontweight="bold")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.35, 0.85))

    exploded = fig.add_subplot(grid[0, 0], projection="3d")
    _draw_wall_3d(exploded, module, exploded=True)
    _format_wall_3d_axis(exploded, 600, "① 爆炸图：A、B、C 三种板件如何组合")

    assembled = fig.add_subplot(grid[0, 1], projection="3d")
    _draw_wall_3d(assembled, module, exploded=False)
    _format_wall_3d_axis(assembled, 600, "② 成品图：加强肋必须前后成对")

    parts_ax = fig.add_subplot(grid[1, 0])
    parts_ax.axis("off")
    parts_ax.set_title("W600 单件物料（W300 差异见下一页）", fontsize=12, fontweight="bold", pad=8)
    rows = [
        ["A", "主墙板", "600 × 344 × 6", "1", "上角 R20；下角 R5"],
        ["B", "底板", "600 × 160 × 6", "1", "刻中心线及肋板定位线"],
        ["C", "三角加强肋", "70 × 100 × 6", "6", "3 个位置，每处前后各 1 件"],
    ]
    table = parts_ax.table(
        cellText=rows,
        colLabels=["代号", "零件", "下料尺寸 / 外形", "数量", "说明"],
        colWidths=[0.08, 0.18, 0.24, 0.09, 0.41],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)
    for col in range(5):
        table[(0, col)].set_facecolor("#F6D5C1")
        table[(0, col)].set_text_props(weight="bold")

    steps_ax = fig.add_subplot(grid[1, 1])
    steps_ax.axis("off")
    steps_ax.text(
        0.02,
        0.98,
        "装配顺序（不得自行改成通长开槽）\n\n"
        "1. 在底板 B 上刻长向中心线和加强肋位置线。\n"
        "2. 主墙板 A 沿中心线立放，用直角夹具校正为 90°。\n"
        "3. 沿 A/B 接缝两侧连续施加 PMMA 专用溶剂胶。\n"
        "4. 每个定位位置粘 2 件 C：墙板前后各 1 件。\n"
        "5. 按胶黏剂说明固化后，再装 U 形护边和防滑垫。\n\n"
        "关键尺寸：344 + 6 = 成品总高 350；底板中心线两侧各 80。\n"
        "70 mm 肋脚不会超出底板，外缘保留约 7 mm 余量。\n\n"
        "先做 1 件 W600 首件，确认粘接、抗倾倒和雷达回波后再批量。",
        va="top",
        fontsize=10.2,
        linespacing=1.42,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFF8F1", edgecolor="#D55E00", linewidth=1.4),
    )
    return fig


def _wall_parts_figure(config: dict):
    module = config["modules"]["W600"]
    stations = [1000.0 * float(value) for value in module["gusset_station_x"]]
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3), constrained_layout=True)
    fig.suptitle("低矮亚克力墙体——零件图与定位尺寸（单位：mm）", fontsize=17, fontweight="bold")

    ax = axes[0, 0]
    ax.add_patch(patches.Rectangle((90, 70), 600, 344, facecolor="#F2A477", edgecolor="#7A3100", linewidth=1.5))
    _dimension(ax, (90, 42), (690, 42), "600")
    _dimension(ax, (55, 70), (55, 414), "344")
    ax.text(390, 242, "A  主墙板", ha="center", va="center", fontsize=15, fontweight="bold")
    ax.annotate("上角 R20（2处）", xy=(90, 414), xytext=(190, 455), arrowprops=dict(arrowstyle="->", color="#333333"), ha="center")
    ax.text(390, 18, "厚 6；下角 R5；全部外露边倒钝并装透明硅胶 U 形护边", ha="center", fontsize=8.5)
    ax.set_xlim(0, 760); ax.set_ylim(0, 485); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("A  主墙板下料图（W600）", fontweight="bold")

    ax = axes[0, 1]
    ax.add_patch(patches.Rectangle((80, 120), 600, 160, facecolor="#F6D5C1", edgecolor="#7A3100", linewidth=1.5))
    ax.add_patch(patches.Rectangle((80, 197), 600, 6, facecolor="#D55E00", edgecolor="#7A3100"))
    for station in stations:
        x = 80 + station
        ax.add_patch(patches.Rectangle((x - 3, 127), 6, 70, facecolor="#FFD2B8", edgecolor="#7A3100"))
        ax.add_patch(patches.Rectangle((x - 3, 203), 6, 70, facecolor="#FFD2B8", edgecolor="#7A3100"))
        ax.text(x, 300, "X=%d" % station, ha="center", fontsize=8)
    _dimension(ax, (80, 88), (680, 88), "600")
    _dimension(ax, (42, 120), (42, 280), "160")
    ax.text(380, 335, "棕色中线 = A 主墙板；浅色短条 = C 加强肋的顶视投影", ha="center", fontsize=9)
    ax.text(380, 55, "B  底板厚 6；不开通长槽；加强肋中心 X=60 / 300 / 540", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xlim(0, 760); ax.set_ylim(30, 370); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("B  底板顶视定位图（W600）", fontweight="bold")

    ax = axes[1, 0]
    ax.add_patch(patches.Rectangle((90, 55), 160, 6, facecolor="#F6D5C1", edgecolor="#7A3100", linewidth=1.3))
    ax.add_patch(patches.Rectangle((167, 61), 6, 344, facecolor="#F2A477", edgecolor="#7A3100", linewidth=1.3))
    ax.add_patch(patches.Polygon([(173, 61), (243, 61), (173, 161)], closed=True, facecolor="#FFD2B8", edgecolor="#7A3100"))
    ax.add_patch(patches.Polygon([(167, 61), (97, 61), (167, 161)], closed=True, facecolor="#FFD2B8", edgecolor="#7A3100"))
    _dimension(ax, (90, 28), (250, 28), "160")
    _dimension(ax, (55, 55), (55, 405), "350")
    _dimension(ax, (272, 61), (272, 161), "100")
    _dimension(ax, (173, 177), (243, 177), "70")
    ax.text(170, 432, "侧视装配剖面：A 居中，C 前后各 1 件", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(170, 8, "底板左右各 80；A 与 B 为 90°", ha="center", fontsize=8.5)
    ax.set_xlim(0, 325); ax.set_ylim(0, 465); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("成品侧视图（最关键装配关系）", fontweight="bold")

    ax = axes[1, 1]
    ax.add_patch(patches.Polygon([(90, 160), (160, 160), (90, 260)], closed=True, facecolor="#FFD2B8", edgecolor="#7A3100", linewidth=1.5))
    _dimension(ax, (90, 138), (160, 138), "70")
    _dimension(ax, (60, 160), (60, 260), "100")
    ax.text(125, 292, "C  三角加强肋（厚 6）", ha="center", fontsize=11, fontweight="bold")
    ax.annotate("外露尖角 R5", xy=(160, 160), xytext=(230, 192), arrowprops=dict(arrowstyle="->", color="#333333"), fontsize=9)
    rows = [
        ["W600", "A 600×344×6", "B 600×160×6", "X=60/300/540", "C×6（3对）"],
        ["W300", "A 300×344×6", "B 300×160×6", "X=60/240", "C×4（2对）"],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["模块", "主墙板", "底板", "加强肋中心位置", "加强肋数量"],
        colWidths=[0.12, 0.22, 0.22, 0.27, 0.17],
        cellLoc="center",
        bbox=(0.02, 0.02, 0.96, 0.27),
    )
    table.auto_set_font_size(False); table.set_fontsize(8.2); table.scale(1, 1.35)
    for col in range(5):
        table[(0, col)].set_facecolor("#F6D5C1"); table[(0, col)].set_text_props(weight="bold")
    ax.text(0.02, 0.33, "通用材料：不透明哑光浇铸 PMMA；尺寸公差 ±3；胶缝不得有贯穿气泡。", transform=ax.transAxes, fontsize=8.5)
    ax.set_xlim(0, 420); ax.set_ylim(0, 320); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("C  加强肋下料图 + 两种模块差异表", fontweight="bold")
    return fig


def _octagon_xy(diameter, orientation=math.radians(22.5)):
    radius = diameter / 2.0
    return [
        (radius * math.cos(orientation + 2.0 * math.pi * index / 8.0), radius * math.sin(orientation + 2.0 * math.pi * index / 8.0))
        for index in range(8)
    ]


def _draw_octagonal_base_3d(ax, diameter, thickness=6.0):
    lower = [(x, y, 0.0) for x, y in _octagon_xy(diameter)]
    upper = [(x, y, thickness) for x, y in _octagon_xy(diameter)]
    faces = [lower, upper]
    for index in range(8):
        nxt = (index + 1) % 8
        faces.append([lower[index], lower[nxt], upper[nxt], upper[index]])
    ax.add_collection3d(Poly3DCollection(faces, facecolors="#EFD2E4", edgecolors="#6B3659", linewidths=0.7, alpha=0.90))


def _draw_octagonal_ring_3d(ax, outer_diameter, width, z0, thickness=6.0):
    outer = _octagon_xy(outer_diameter)
    inner = _octagon_xy(max(outer_diameter - 2.0 * width / math.cos(math.pi / 8.0), 1.0))
    faces = []
    for index in range(8):
        nxt = (index + 1) % 8
        faces.append([
            (outer[index][0], outer[index][1], z0),
            (outer[nxt][0], outer[nxt][1], z0),
            (inner[nxt][0], inner[nxt][1], z0),
            (inner[index][0], inner[index][1], z0),
        ])
        faces.append([
            (outer[index][0], outer[index][1], z0 + thickness),
            (outer[nxt][0], outer[nxt][1], z0 + thickness),
            (inner[nxt][0], inner[nxt][1], z0 + thickness),
            (inner[index][0], inner[index][1], z0 + thickness),
        ])
    ax.add_collection3d(Poly3DCollection(faces, facecolors="#C989B3", edgecolors="#6B3659", linewidths=0.65, alpha=0.92))


def _draw_octagonal_prism_3d(ax, diameter, height, side_thickness, exploded=False):
    base_thickness = 6.0
    ring_thickness = 6.0
    ring_width = 30.0
    vertices = _octagon_xy(diameter)
    _draw_octagonal_base_3d(ax, diameter, base_thickness)
    panel_z0 = base_thickness + (38.0 if exploded else 0.0)
    radial_shift = 46.0 if exploded else 0.0
    colors = ("#E8B8D6", "#DFA8CC")
    for index in range(8):
        nxt = (index + 1) % 8
        x1, y1 = vertices[index]
        x2, y2 = vertices[nxt]
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        norm = max(math.hypot(mx, my), 1.0)
        sx, sy = radial_shift * mx / norm, radial_shift * my / norm
        face = [
            (x1 + sx, y1 + sy, panel_z0),
            (x2 + sx, y2 + sy, panel_z0),
            (x2 + sx, y2 + sy, panel_z0 + height - base_thickness),
            (x1 + sx, y1 + sy, panel_z0 + height - base_thickness),
        ]
        ax.add_collection3d(Poly3DCollection([face], facecolors=colors[index % 2], edgecolors="#6B3659", linewidths=0.8, alpha=0.84))

    inner_outer_diameter = diameter - 2.0 * side_thickness / math.cos(math.pi / 8.0)
    ring_z = height - ring_thickness if not exploded else height + 72.0
    _draw_octagonal_ring_3d(ax, inner_outer_diameter, ring_width, ring_z, ring_thickness)
    limit = diameter * 0.85 + radial_shift
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_zlim(0, height + 120)
    ax.set_box_aspect((1.0, 1.0, 1.15))
    ax.view_init(elev=23, azim=-48)
    ax.set_proj_type("ortho"); ax.set_axis_off()


def _octagonal_assembly_figure(config: dict):
    reference = config["modules"]["C320"]
    diameter = 1000.0 * float(reference["diameter"])
    height = 1000.0 * float(reference["height"])
    side_thickness = 1000.0 * float(reference["panel_thickness"])
    fig = plt.figure(figsize=(11.7, 8.3), constrained_layout=True)
    fig.suptitle("八边形亚克力柱——供应商装配总图（单位：mm）", fontsize=17, fontweight="bold")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.25, 0.85))

    ax = fig.add_subplot(grid[0, 0], projection="3d")
    _draw_octagonal_prism_3d(ax, diameter, height, side_thickness, exploded=True)
    ax.set_title("① C320 爆炸图：8块侧板 + 底板 + 内嵌上口环", fontsize=12, fontweight="bold")
    ax.text2D(0.02, 0.77, "A  等宽侧板 × 8", transform=ax.transAxes, fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", facecolor="#E8B8D6", edgecolor="#6B3659"))
    ax.text2D(0.02, 0.66, "B  八边形底板 × 1", transform=ax.transAxes, fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", facecolor="#EFD2E4", edgecolor="#6B3659"))
    ax.text2D(0.02, 0.55, "C  上口加强环 × 1", transform=ax.transAxes, fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", facecolor="#C989B3", edgecolor="#6B3659"))

    ax = fig.add_subplot(grid[0, 1], projection="3d")
    _draw_octagonal_prism_3d(ax, diameter, height, side_thickness, exploded=False)
    ax.set_title("② C320 成品图：D 为外接圆直径", fontsize=12, fontweight="bold")
    ax.text2D(0.03, 0.83, "成品外形：D320 × H350", transform=ax.transAxes, fontsize=9.5, fontweight="bold", zorder=100, clip_on=False, bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#6B3659"))
    ax.text2D(0.03, 0.73, "上口环嵌在内部，顶面齐平", transform=ax.transAxes, fontsize=9.5, fontweight="bold", zorder=100, clip_on=False, bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#6B3659"))

    table_ax = fig.add_subplot(grid[1, 0])
    table_ax.axis("off")
    rows = []
    base_thickness = 1000.0 * float(config["design_assumptions"]["acrylic_prism_base_thickness"])
    for key in ("C240", "C320", "C560", "D320"):
        module = config["modules"][key]
        d = 1000.0 * float(module["diameter"])
        h = 1000.0 * float(module["height"])
        t = 1000.0 * float(module["panel_thickness"])
        side_width = d * math.sin(math.pi / 8.0)
        rows.append([key, "D%d × H%d" % (round(d), round(h)), "%.1f × %d × %d" % (side_width, round(h - base_thickness), round(t)), "8", str(module["quantity"])])
    table = table_ax.table(
        cellText=rows,
        colLabels=["模块", "成品外形", "A 单块侧板有效尺寸", "每柱侧板数", "成品数量"],
        colWidths=[0.13, 0.20, 0.34, 0.18, 0.15],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False); table.set_fontsize(8.8); table.scale(1, 1.5)
    for col in range(5):
        table[(0, col)].set_facecolor("#EFD2E4"); table[(0, col)].set_text_props(weight="bold")
    table_ax.set_title("规格差异表（侧板高度 = 成品高 − 6 mm 底板）", fontsize=11, fontweight="bold", pad=8)

    note_ax = fig.add_subplot(grid[1, 1])
    note_ax.axis("off")
    note_ax.text(
        0.02,
        0.98,
        "装配顺序\n\n"
        "1. 8 块 A 围成正八边形；竖缝采用 22.5° 对拼斜边，或由供应商给出等效内衬方案。\n"
        "2. 侧板底边粘在 6 mm 八边形底板 B 上；B 的 D 等于成品标称 D。\n"
        "3. 30 mm 宽、6 mm 厚的 C 从上口嵌入，顶面与侧板齐平，不增加成品高度。\n"
        "4. 顶边装透明硅胶护边；底面贴 2 mm 防滑垫。\n\n"
        "侧板表中宽度为外表面有效宽度；正式批量前先做 C320 首件，按实测板厚修正斜边和上口环外径。\n"
        "D320 为可选动态外壳，必须软边、快拆、低速并配独立急停。",
        va="top",
        fontsize=9.7,
        linespacing=1.42,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFF7FC", edgecolor="#6B3659", linewidth=1.4),
    )
    return fig


def _octagonal_figure(config: dict):
    keys = ("C240", "C320", "C560", "D320")
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3), constrained_layout=True)
    fig.suptitle("亚克力八边形柱障碍加工图（单位：mm）", fontweight="bold")
    for ax, key in zip(axes.flat, keys):
        module = config["modules"][key]
        diameter = 1000 * float(module["diameter"])
        height = 1000 * float(module["height"])
        sides = int(module.get("sides", 8))
        side_width = diameter * math.sin(math.pi / sides)
        polygon = patches.RegularPolygon(
            (120 + diameter / 2, 100 + diameter / 2),
            numVertices=sides,
            radius=diameter / 2,
            orientation=math.radians(22.5),
            facecolor="#F5D7EA", edgecolor="#6B3659", linewidth=1.2,
        )
        ax.add_patch(polygon)
        ax.add_patch(patches.Rectangle((120, 100 + diameter + 55), diameter, height, facecolor="#E8B8D6", edgecolor="#6B3659"))
        _dimension(ax, (120, 70), (120 + diameter, 70), "D%d" % diameter)
        _dimension(ax, (85, 100 + diameter + 55), (85, 100 + diameter + 55 + height), "%d" % height)
        ax.text(120 + diameter / 2, 100 + diameter / 2, key + " 顶视", ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(120 + diameter / 2, 100 + diameter + 55 + height / 2, key + " 正视", ha="center", va="center", fontsize=11, fontweight="bold")
        ax.set_xlim(0, max(850, diameter + 260)); ax.set_ylim(0, max(860, diameter + height + 240)); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("%s：8×侧板约 %.0f×%.0f×%.0f" % (key, side_width, height, 1000 * float(module["panel_thickness"])), fontsize=10)
    fig.text(
        0.5,
        0.01,
        "每个模块由 8 块 4 mm 侧板、6 mm 八边形底板和上口加强环粘接；D 为外接圆直径。动态 D320 必须加硅胶软边条、快拆安装、独立急停，先以 0.10 m/s 验收。",
        ha="center",
        fontsize=9,
    )
    return fig


def _bom_figure(config: dict):
    fig, ax = plt.subplots(figsize=(11.7, 8.3), constrained_layout=True)
    ax.axis("off")
    ax.set_title("建议一次性采购清单与验收项目", fontsize=17, fontweight="bold", pad=20)
    rows = []
    for key, module in config["modules"].items():
        if "dimensions" in module:
            dims = " × ".join(str(int(round(float(v) * 1000))) for v in module["dimensions"]) + " mm"
        else:
            dims = "D%d × H%d mm" % (round(module["diameter"] * 1000), round(module["height"] * 1000))
        rows.append([key, dims, str(module["quantity"]), module["description"]])
    table = ax.table(
        cellText=rows,
        colLabels=["编号", "外形尺寸", "数量", "用途"],
        colWidths=[0.12, 0.24, 0.10, 0.48],
        cellLoc="left",
        loc="upper center",
    )
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.65)
    for col in range(4):
        table[(0, col)].set_facecolor("#D9EAF7"); table[(0, col)].set_text_props(weight="bold")
    ax.text(
        0.02,
        0.36,
        "材料：墙板/底座 6 mm、八边形侧板 4 mm 不透明哑光浇铸 PMMA；外露边安装透明硅胶 U 形护边。\n\n"
        "连接件：直连 12、90° 6、T 形 2。另购 50 mm 黑/黄哑光地贴、卷尺、激光测距仪、AprilTag/ArUco ID、防滑垫。\n\n"
        "下单前强制复测：①车体最大外接轮廓；②雷达光束离地高度（当前兼容 100–250 mm）；③材料样片回波；④急停距离；⑤首件推力/倾倒稳定性。\n\n"
        "供应商验收：尺寸公差、圆角倒钝、哑光不透明表面、粘接质量、护边、防滑、连接缝、无突出支脚。",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        linespacing=1.55,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFFBEA", edgecolor="#C49A00"),
    )
    return fig


def _save_figure(fig, output_dir: Path, stem: str, pdf: PdfPages):
    png = output_dir / (stem + ".png")
    svg = output_dir / (stem + ".svg")
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _write_bom(config: dict, output_dir: Path):
    with (output_dir / "bill_of_materials.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["module_id", "kind", "dimensions_mm", "quantity", "description"])
        for key, module in config["modules"].items():
            if "dimensions" in module:
                dims = "x".join(str(int(round(float(v) * 1000))) for v in module["dimensions"])
            else:
                dims = "D%dxH%d" % (round(module["diameter"] * 1000), round(module["height"] * 1000))
            writer.writerow([key, module["kind"], dims, module["quantity"], module["description"]])


def _write_cut_list(config: dict, output_dir: Path):
    path = output_dir / "acrylic_cut_list.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["part_id", "module_id", "material", "thickness_mm", "blank_or_profile_mm", "quantity", "fabrication_notes"]
        )
        for key, module in config["modules"].items():
            quantity = int(module["quantity"])
            kind = module["kind"]
            if kind == "wall":
                panel_length, panel_height, thickness = [1000.0 * float(v) for v in module["panel_dimensions"]]
                base_length, base_depth, base_thickness = [1000.0 * float(v) for v in module["base_dimensions"]]
                horizontal_leg = 1000.0 * float(module["gusset_horizontal_leg"])
                vertical_leg = 1000.0 * float(module["gusset_vertical_leg"])
                stations = "/".join(str(round(1000.0 * float(value))) for value in module["gusset_station_x"])
                writer.writerow([key + "-A", key, "opaque matte cast PMMA", thickness, "%dx%d main panel" % (round(panel_length), round(panel_height)), quantity, "top corners R20; lower corners R5; deburr; silicone U-edge"])
                writer.writerow([key + "-B", key, "opaque matte cast PMMA", base_thickness, "%dx%d base" % (round(base_length), round(base_depth)), quantity, "NO THROUGH SLOT; etch centerline and gusset stations X=" + stations + "; nonslip pads below"])
                writer.writerow([key + "-C", key, "opaque matte cast PMMA", thickness, "%dx%d right-triangle gusset" % (round(horizontal_leg), round(vertical_leg)), quantity * int(module["gusset_count"]), "install as front/back pairs; R5 exposed corner; solvent bond"])
            elif kind in ("octagonal_prism", "dynamic_octagonal_shell"):
                diameter = 1000.0 * float(module["diameter"])
                height = 1000.0 * float(module["height"])
                sides = int(module.get("sides", 8))
                thickness = 1000.0 * float(module["panel_thickness"])
                base_thickness = 1000.0 * float(config["design_assumptions"]["acrylic_prism_base_thickness"])
                side_width = diameter * math.sin(math.pi / sides)
                inner_diameter = diameter - 2.0 * thickness / math.cos(math.pi / sides)
                writer.writerow([key + "-A", key, "opaque matte cast PMMA", thickness, "%.1fx%d side panel" % (side_width, round(height - base_thickness)), quantity * sides, "eight equal panels; 22.5-deg vertical miter or supplier-equivalent inner seam; silicone edge at top"])
                writer.writerow([key + "-B", key, "opaque matte cast PMMA", base_thickness, "regular octagon D%d" % round(diameter), quantity, "full base; D is circumscribed diameter; nonslip pads below"])
                writer.writerow([key + "-C", key, "opaque matte cast PMMA", 6, "30-wide octagonal ring, outer D approx %.1f" % inner_diameter, quantity, "fit inside side panels; top flush; freeze after C320 first-article fit"])
        connector_specs = (
            ("J-S", "straight_internal", "160x60 splice plate"),
            ("J-L", "right_angle_internal", "160x160 L plate"),
            ("J-T", "tee_internal", "240x160 T plate"),
        )
        for part_id, connector_key, profile in connector_specs:
            writer.writerow([part_id, "CONNECTOR", "opaque matte cast PMMA", 6, profile, int(config["connectors"][connector_key]["quantity"]), "final slot/hole pattern frozen after W600 first-article fit"])


def _write_placements(config: dict, output_dir: Path):
    ox, oy = config["field"]["experiment_origin_in_field"]
    with (output_dir / "scene_placements.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["scene", "role", "obstacle_id", "module", "field_x_m", "field_y_m", "experiment_x_m", "experiment_y_m", "yaw_deg"]
        )
        for scene_name, scene in config["scenes"].items():
            for item in scene.get("obstacles", []):
                x, y = item["center"]
                writer.writerow(
                    [scene_name, scene["role"], item["id"], item["module"], "%.3f" % x, "%.3f" % y, "%.3f" % (x - ox), "%.3f" % (y - oy), "%.1f" % item.get("yaw_deg", 0.0)]
                )


def _drawio_cell(root, cell_id, value, style, x, y, width, height, parent="1"):
    cell = ET.SubElement(root, "mxCell", id=str(cell_id), value=str(value), style=style, vertex="1", parent=parent)
    ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(width), height=str(height), **{"as": "geometry"})


def _drawio_edge(root, cell_id, source, target, value=""):
    cell = ET.SubElement(
        root,
        "mxCell",
        id=str(cell_id),
        value=value,
        style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=classic;strokeWidth=2;strokeColor=#7A3100;",
        edge="1",
        parent="1",
        source=str(source),
        target=str(target),
    )
    ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})


def _drawio_wall_assembly_page(config: dict):
    diagram = ET.Element("diagram", name="01 墙体爆炸装配图")
    model = ET.SubElement(diagram, "mxGraphModel", dx="1600", dy="900", grid="1", gridSize="10", page="1", pageScale="1", pageWidth="1169", pageHeight="827")
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")
    _drawio_cell(root, 2, "W600 亚克力低墙：零件与装配关系（mm）", "text;html=1;align=center;verticalAlign=middle;fontSize=24;fontStyle=1;", 180, 25, 800, 55)
    _drawio_cell(root, 3, "A  主墙板&lt;br&gt;600 × 344 × 6&lt;br&gt;1件", "rounded=1;whiteSpace=wrap;html=1;fillColor=#F2A477;strokeColor=#7A3100;fontStyle=1;fontSize=16;", 70, 130, 250, 110)
    _drawio_cell(root, 4, "B  底板&lt;br&gt;600 × 160 × 6&lt;br&gt;1件；不开通长槽", "rounded=1;whiteSpace=wrap;html=1;fillColor=#F6D5C1;strokeColor=#7A3100;fontStyle=1;fontSize=16;", 70, 285, 250, 110)
    _drawio_cell(root, 5, "C  三角加强肋&lt;br&gt;70 × 100 × 6&lt;br&gt;6件；前后3对", "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFD2B8;strokeColor=#7A3100;fontStyle=1;fontSize=16;", 70, 440, 250, 110)
    _drawio_cell(root, 6, "W600 成品：600 × 160 × 350", "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#7A3100;strokeWidth=2;dashed=1;verticalAlign=top;fontStyle=1;fontSize=17;container=1;pointerEvents=0;", 520, 110, 500, 450)
    _drawio_cell(root, 7, "A 主墙板", "rounded=0;whiteSpace=wrap;html=1;fillColor=#F2A477;strokeColor=#7A3100;fontStyle=1;fontSize=18;", 105, 65, 300, 190, parent="6")
    _drawio_cell(root, 8, "C 加强肋：X=60 / 300 / 540；每处前后各1件", "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFD2B8;strokeColor=#7A3100;fontStyle=1;fontSize=14;", 105, 270, 300, 45, parent="6")
    _drawio_cell(root, 9, "B 底板（中心线两侧各80）", "rounded=0;whiteSpace=wrap;html=1;fillColor=#F6D5C1;strokeColor=#7A3100;fontStyle=1;fontSize=15;", 65, 335, 380, 60, parent="6")
    _drawio_edge(root, 12, 3, 6)
    _drawio_edge(root, 13, 4, 6)
    _drawio_edge(root, 14, 5, 6)
    steps = [
        "1  B上刻中心线和X=60/300/540定位线",
        "2  A沿中心线立放，夹具校正90°",
        "3  A/B接缝两侧连续施溶剂胶",
        "4  每处前后各粘1件C，固化后装护边",
    ]
    for index, value in enumerate(steps):
        _drawio_cell(root, 20 + index, value, "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF8F1;strokeColor=#D55E00;fontSize=13;", 65 + index * 270, 625, 245, 105)
        if index:
            _drawio_edge(root, 30 + index, 19 + index, 20 + index)
    return diagram


def _drawio_page(config: dict, name: str, scene: dict | None = None):
    diagram = ET.Element("diagram", name=name)
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        dx="1600",
        dy="900",
        grid="1",
        gridSize="10",
        page="1",
        pageScale="1",
        pageWidth="1169",
        pageHeight="827",
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")
    scale = 100.0
    margin = 80.0
    field_w, field_h = config["field"]["size"]
    _drawio_cell(root, 2, "6.5 m × 6.5 m 场地", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#111111;strokeWidth=2;verticalAlign=top;fontStyle=1;container=1;pointerEvents=0;", margin, margin, field_w * scale, field_h * scale)
    xmin, xmax, ymin, ymax = config["field"]["active_bounds"]
    _drawio_cell(root, 3, "4.5 m × 4.5 m 有效实验区", "rounded=0;whiteSpace=wrap;html=1;fillColor=#f7f7f7;strokeColor=#666666;dashed=1;verticalAlign=top;container=1;pointerEvents=0;", xmin * scale, (field_h - ymax) * scale, (xmax - xmin) * scale, (ymax - ymin) * scale, parent="2")
    sx, sy = config["field"]["start"]
    gx, gy = config["field"]["goal"]
    _drawio_cell(root, 4, "S", "ellipse;whiteSpace=wrap;html=1;fillColor=#0072B2;strokeColor=#004A75;fontColor=#ffffff;fontStyle=1;", (sx - xmin - 0.25) * scale, (ymax - sy - 0.25) * scale, 50, 50, parent="3")
    _drawio_cell(root, 5, "G", "ellipse;whiteSpace=wrap;html=1;fillColor=#009E73;strokeColor=#006849;fontColor=#ffffff;fontStyle=1;", (gx - xmin - 0.18) * scale, (ymax - gy - 0.18) * scale, 36, 36, parent="3")
    next_id = 6
    if scene:
        for item in scene.get("obstacles", []):
            kind, width, depth = _module_shape(config, item)
            cx, cy = item["center"]
            yaw = float(item.get("yaw_deg", 0.0))
            if kind == "wall":
                style = "rounded=0;whiteSpace=wrap;html=1;fillColor=#D55E00;strokeColor=#7A3100;rotation=%s;fontStyle=1;" % yaw
            else:
                style = "ellipse;whiteSpace=wrap;html=1;fillColor=#CC79A7;strokeColor=#6B3659;fontStyle=1;dashed=1;"
            _drawio_cell(root, next_id, item["id"], style, (cx - xmin - width / 2) * scale, (ymax - cy - depth / 2) * scale, width * scale, depth * scale, parent="3")
            next_id += 1
        dynamic = scene.get("dynamic_obstacle")
        if dynamic:
            x1, y1 = dynamic["path_start"]; x2, y2 = dynamic["path_end"]
            edge = ET.SubElement(root, "mxCell", id=str(next_id), value="动态横穿 0.10/0.20 m/s", style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;startArrow=classic;endArrow=classic;strokeColor=#E69F00;strokeWidth=3;", edge="1", parent="3")
            geom = ET.SubElement(edge, "mxGeometry", relative="1", **{"as": "geometry"})
            ET.SubElement(geom, "mxPoint", x=str((x1 - xmin) * scale), y=str((ymax - y1) * scale), **{"as": "sourcePoint"})
            ET.SubElement(geom, "mxPoint", x=str((x2 - xmin) * scale), y=str((ymax - y2) * scale), **{"as": "targetPoint"})
    return diagram


def _write_drawio(config: dict, output_dir: Path):
    mxfile = ET.Element("mxfile", host="drawio", version="26.0.0")
    mxfile.append(_drawio_wall_assembly_page(config))
    mxfile.append(_drawio_page(config, "00 场地坐标"))
    for scene_name, scene in config["scenes"].items():
        mxfile.append(_drawio_page(config, scene_name, scene))
    tree = ET.ElementTree(mxfile)
    tree.write(output_dir / "obstacle_kit_6p5m.drawio", encoding="utf-8", xml_declaration=True)


def _write_supplier_zip(output_dir: Path) -> None:
    zip_path = output_dir.parent / "obstacle_kit_6p5m_supplier_package.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in {".pdf", ".png", ".svg", ".csv", ".drawio", ".md", ".xlsx"}:
                archive.write(path, arcname="obstacle_kit_6p5m/%s" % path.name)


def _remove_stale_numbered_drawings(output_dir: Path) -> None:
    """Remove only generator-owned numbered PNG/SVG pages before regeneration."""
    for suffix in ("png", "svg"):
        for path in output_dir.glob("[0-9][0-9]_*." + suffix):
            path.unlink()


def generate(config_path: Path, output_dir: Path) -> None:
    config = _load(config_path)
    _validate_config(config)
    _setup_font()
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_numbered_drawings(output_dir)
    scene_order = [
        "clean_dynamics",
        "single_obstacle",
        "straight_corridor_0900",
        "u_trap",
        "lab_complex",
        "dynamic_crossing",
        "friction_patch",
    ]
    with PdfPages(output_dir / "obstacle_kit_6p5m_supplier_drawings.pdf") as pdf:
        _save_figure(_site_figure(config), output_dir, "00_field_coordinate_system", pdf)
        _save_figure(_wall_assembly_figure(config), output_dir, "01_wall_assembly_exploded", pdf)
        _save_figure(_wall_parts_figure(config), output_dir, "02_wall_part_drawings", pdf)
        _save_figure(_octagonal_assembly_figure(config), output_dir, "03_octagonal_assembly_exploded", pdf)
        for index, scene_name in enumerate(scene_order, start=4):
            _save_figure(_scene_figure(config, scene_name), output_dir, "%02d_scene_%s" % (index, scene_name), pdf)
        _save_figure(_bom_figure(config), output_dir, "11_bill_of_materials_and_acceptance", pdf)
    _write_bom(config, output_dir)
    _write_cut_list(config, output_dir)
    _write_placements(config, output_dir)
    _write_drawio(config, output_dir)
    _write_supplier_zip(output_dir)
    print("generated obstacle design package at %s" % output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.config.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
