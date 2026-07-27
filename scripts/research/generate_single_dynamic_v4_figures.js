#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const sharp = require("sharp");
const { chromium } = require("playwright");

const REPO = path.resolve(__dirname, "..", "..");
const ANALYSIS = path.join(
  REPO,
  "research_artifacts",
  "single_dynamic_obstacle_paper_v4_analysis_20260725",
  "analysis"
);
const LANG_ARG = process.argv.slice(2).find((arg) => arg.startsWith("--lang="));
const LANG = LANG_ARG ? LANG_ARG.split("=", 2)[1].toLowerCase() : "en";
if (!["en", "zh"].includes(LANG)) {
  throw new Error(`Unsupported language: ${LANG}; expected en or zh`);
}
const OUTPUT_ARG = process.argv.slice(2).find((arg) => !arg.startsWith("--"));
const OUTPUT = OUTPUT_ARG
  ? path.resolve(OUTPUT_ARG)
  : path.join(
      process.env.USERPROFILE || REPO,
      "Desktop",
      LANG === "zh"
        ? "single_dynamic_v4_figures_zh_20260727"
        : "single_dynamic_v4_figures_20260727"
    );

const W = 1600;
const H = 1000;
const IS_ZH = LANG === "zh";
const COLORS = {
  ink: "#17202A",
  muted: "#5D6D7E",
  grid: "#D9E2EC",
  paper: "#FFFFFF",
  baseline: "#7A8793",
  full: "#0072B2",
  learning: "#E69F00",
  probability: "#009E73",
  success: "#009E73",
  collision: "#D55E00",
  ci: "#4C566A",
  softBlue: "#E8F3F8",
  softOrange: "#FBEFE8",
  softGreen: "#E8F5F0",
  danger: "#CC3311",
};

function tr(en, zh) {
  return IS_ZH ? zh : en;
}

function esc(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        cell += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        cell += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(cell);
      cell = "";
    } else if (ch === "\n") {
      row.push(cell.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += ch;
    }
  }
  if (cell.length || row.length) {
    row.push(cell.replace(/\r$/, ""));
    rows.push(row);
  }
  const header = rows.shift();
  return rows
    .filter((r) => r.some((v) => v !== ""))
    .map((r) => Object.fromEntries(header.map((h, i) => [h, r[i] ?? ""])));
}

function readJson(name) {
  return JSON.parse(fs.readFileSync(path.join(ANALYSIS, name), "utf8"));
}

function readCsv(name) {
  return parseCsv(fs.readFileSync(path.join(ANALYSIS, name), "utf8"));
}

function parseCi(value) {
  return JSON.parse(value.replaceAll("'", '"'));
}

function pLabel(p) {
  if (p < 0.001) return "p < 0.001";
  return `p = ${p.toFixed(3)}`;
}

function pct(value, digits = 1) {
  return `${(100 * value).toFixed(digits)}%`;
}

function pp(value, digits = 1) {
  const x = 100 * value;
  return `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(digits)} pp`;
}

function text(x, y, value, options = {}) {
  const {
    size = 28,
    weight = 400,
    fill = COLORS.ink,
    anchor = "start",
    family = "Arial, 'Microsoft YaHei', sans-serif",
    rotate = null,
    opacity = 1,
  } = options;
  const transform = rotate === null ? "" : ` transform="rotate(${rotate} ${x} ${y})"`;
  return `<text x="${x}" y="${y}" font-family="${family}" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}" opacity="${opacity}"${transform}>${esc(value)}</text>`;
}

function line(x1, y1, x2, y2, options = {}) {
  const { stroke = COLORS.grid, width = 2, dash = "", opacity = 1 } = options;
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${width}" ${dash ? `stroke-dasharray="${dash}"` : ""} opacity="${opacity}"/>`;
}

function rect(x, y, width, height, options = {}) {
  const {
    fill = "none",
    stroke = "none",
    strokeWidth = 0,
    radius = 0,
    opacity = 1,
  } = options;
  return `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="${radius}" fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}" opacity="${opacity}"/>`;
}

function circle(cx, cy, r, options = {}) {
  const { fill = COLORS.full, stroke = COLORS.paper, strokeWidth = 3 } = options;
  return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}"/>`;
}

function baseSvg(titleValue, subtitle) {
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`,
    rect(0, 0, W, H, { fill: COLORS.paper }),
    text(80, 76, titleValue, { size: 42, weight: 700 }),
    text(80, 118, subtitle, { size: 23, fill: COLORS.muted }),
    line(80, 145, 1520, 145, { stroke: COLORS.ink, width: 2 }),
  ].join("\n");
}

function footer(
  left = tr(
    "Single dynamic obstacle v4 · sealed formal analysis · 360 paired core seeds",
    "单动态障碍 v4 · 封存正式分析 · 360 个配对核心种子"
  ),
  right = tr(
    "Error bars: paired Tango two-sided 95% CI",
    "误差线：配对 Tango 双侧 95% 置信区间"
  )
) {
  return [
    line(80, 930, 1520, 930, { stroke: COLORS.grid, width: 2 }),
    text(80, 965, left, {
      size: 18,
      fill: COLORS.muted,
    }),
    text(1520, 965, right, {
      size: 18,
      fill: COLORS.muted,
      anchor: "end",
    }),
    "</svg>",
  ].join("\n");
}

function drawRatePanel({ x, y, width, height, titleValue, max, rates, delta, p, color }) {
  const out = [];
  out.push(rect(x, y, width, height, { fill: "#FBFCFE", stroke: COLORS.grid, strokeWidth: 2, radius: 18 }));
  out.push(text(x + 34, y + 54, titleValue, { size: 30, weight: 700 }));
  const plot = { x: x + 90, y: y + 110, w: width - 140, h: height - 270 };
  for (let i = 0; i <= 5; i += 1) {
    const value = (max * i) / 5;
    const yy = plot.y + plot.h - (value / max) * plot.h;
    out.push(line(plot.x, yy, plot.x + plot.w, yy, { stroke: COLORS.grid, width: 1.5 }));
    out.push(text(plot.x - 18, yy + 7, `${Math.round(100 * value)}%`, {
      size: 18,
      fill: COLORS.muted,
      anchor: "end",
    }));
  }
  const barW = 135;
  const gap = 80;
  const start = plot.x + (plot.w - (2 * barW + gap)) / 2;
  rates.forEach((item, i) => {
    const bh = (item.value / max) * plot.h;
    const bx = start + i * (barW + gap);
    const by = plot.y + plot.h - bh;
    out.push(rect(bx, by, barW, bh, { fill: item.color, radius: 8 }));
    out.push(text(bx + barW / 2, by - 16, pct(item.value), {
      size: 25,
      weight: 700,
      anchor: "middle",
      fill: item.color,
    }));
    out.push(text(bx + barW / 2, plot.y + plot.h + 40, item.label, {
      size: 22,
      weight: 600,
      anchor: "middle",
    }));
  });
  out.push(rect(x + 34, y + height - 88, width - 68, 54, {
    fill: color === COLORS.collision ? COLORS.softOrange : COLORS.softBlue,
    radius: 12,
  }));
  out.push(text(
    x + width / 2,
    y + height - 51,
    `${tr("Full − Baseline", "完整方法 − 基线")}: ${pp(delta)}  ·  ${pLabel(p)}`,
    {
    size: 22,
    weight: 700,
    anchor: "middle",
    fill: color,
    }
  ));
  return out.join("\n");
}

function figurePrimary(primary) {
  const pooled = primary.by_split.pooled;
  return [
    baseSvg(
      tr(
        "Primary outcomes: Full reduces collisions, not completion failures",
        "主要结果：完整方法显著降低碰撞，但未显著提高任务完成率"
      ),
      tr(
        "Strong nominal MPPI (Baseline) versus proposed Full system; pooled ID + OOD results",
        "强标称 MPPI（基线）与完整方法对比；合并 ID 与 OOD 的正式结果"
      )
    ),
    drawRatePanel({
      x: 90,
      y: 180,
      width: 680,
      height: 700,
      titleValue: tr("Safe success rate", "安全成功率"),
      max: 1,
      rates: [
        { label: tr("Baseline", "基线"), value: pooled.success_rate.control_rate, color: COLORS.baseline },
        { label: tr("Full", "完整方法"), value: pooled.success_rate.treatment_rate, color: COLORS.full },
      ],
      delta: pooled.success_rate.treatment_minus_control,
      p: pooled.success_rate.exact_mcnemar_two_sided_p,
      color: COLORS.full,
    }),
    drawRatePanel({
      x: 830,
      y: 180,
      width: 680,
      height: 700,
      titleValue: tr("Collision rate", "碰撞率"),
      max: 0.30,
      rates: [
        { label: tr("Baseline", "基线"), value: pooled.collision_rate.control_rate, color: COLORS.baseline },
        { label: tr("Full", "完整方法"), value: pooled.collision_rate.treatment_rate, color: COLORS.collision },
      ],
      delta: pooled.collision_rate.treatment_minus_control,
      p: pooled.collision_rate.exact_mcnemar_two_sided_p,
      color: COLORS.collision,
    }),
    footer(
      tr(
        "Single dynamic obstacle v4 · sealed formal analysis · 360 paired core seeds",
        "单动态障碍 v4 · 封存正式分析 · 360 个配对核心种子"
      ),
      tr("Paired exact McNemar tests", "配对精确 McNemar 检验")
    ),
  ].join("\n");
}

function drawForestPanel({
  x,
  y,
  width,
  height,
  titleValue,
  rows,
  domain,
  favorableDirection,
  xLabel = tr(
    "Paired Full − Baseline difference (percentage points)",
    "配对差值：完整方法 − 基线（百分点）"
  ),
}) {
  const out = [];
  out.push(rect(x, y, width, height, { fill: "#FBFCFE", stroke: COLORS.grid, strokeWidth: 2, radius: 18 }));
  out.push(text(x + 34, y + 54, titleValue, { size: 29, weight: 700 }));
  out.push(text(x + width - 34, y + 54, favorableDirection, {
    size: 18,
    fill: COLORS.muted,
    anchor: "end",
  }));
  const plot = { x: x + 145, y: y + 105, w: width - 205, h: height - 210 };
  const scale = (v) => plot.x + ((v - domain[0]) / (domain[1] - domain[0])) * plot.w;
  const ticks = 6;
  for (let i = 0; i <= ticks; i += 1) {
    const v = domain[0] + ((domain[1] - domain[0]) * i) / ticks;
    const xx = scale(v);
    out.push(line(xx, plot.y, xx, plot.y + plot.h, { stroke: COLORS.grid, width: 1.5 }));
    out.push(text(xx, plot.y + plot.h + 34, `${v > 0 ? "+" : ""}${v.toFixed(0)}`, {
      size: 17,
      fill: COLORS.muted,
      anchor: "middle",
    }));
  }
  out.push(line(scale(0), plot.y - 10, scale(0), plot.y + plot.h, { stroke: COLORS.ink, width: 2.5 }));
  rows.forEach((row, i) => {
    const yy = plot.y + 55 + i * 125;
    out.push(text(plot.x - 24, yy + 7, row.label, { size: 23, weight: 600, anchor: "end" }));
    out.push(line(scale(row.ci[0]), yy, scale(row.ci[1]), yy, { stroke: COLORS.ci, width: 6 }));
    out.push(line(scale(row.ci[0]), yy - 12, scale(row.ci[0]), yy + 12, { stroke: COLORS.ci, width: 4 }));
    out.push(line(scale(row.ci[1]), yy - 12, scale(row.ci[1]), yy + 12, { stroke: COLORS.ci, width: 4 }));
    out.push(circle(scale(row.effect), yy, 12, { fill: row.color, stroke: COLORS.paper, strokeWidth: 3 }));
    out.push(text(x + width - 18, yy + 7, `${row.effect >= 0 ? "+" : ""}${row.effect.toFixed(1)} pp`, {
      size: 21,
      weight: 700,
      fill: row.color,
      anchor: "end",
    }));
  });
  out.push(text(plot.x + plot.w / 2, y + height - 36, xLabel, {
    size: 19,
    fill: COLORS.muted,
    anchor: "middle",
  }));
  return out.join("\n");
}

function figureSplitEffects(primary) {
  const labels = [["pooled", tr("Pooled", "合并")], ["id", "ID"], ["ood", "OOD"]];
  const successRows = labels.map(([key, label]) => {
    const d = primary.by_split[key].success_rate;
    return {
      label,
      effect: 100 * d.treatment_minus_control,
      ci: d.tango_two_sided_95ci.map((v) => 100 * v),
      color: COLORS.success,
    };
  });
  const collisionRows = labels.map(([key, label]) => {
    const d = primary.by_split[key].collision_rate;
    return {
      label,
      effect: 100 * d.treatment_minus_control,
      ci: d.tango_two_sided_95ci.map((v) => 100 * v),
      color: COLORS.collision,
    };
  });
  return [
    baseSvg(
      tr(
        "Paired effects are consistent across ID and OOD splits",
        "ID 与 OOD 分层中的配对效应总体一致"
      ),
      tr(
        "Raw Full − Baseline differences; points are paired effects and bars are Tango 95% confidence intervals",
        "横轴为完整方法减基线；圆点为配对效应，误差线为 Tango 95% 置信区间"
      )
    ),
    drawForestPanel({
      x: 80,
      y: 190,
      width: 710,
      height: 690,
      titleValue: tr("Safe success difference", "安全成功率差值"),
      rows: successRows,
      domain: [-16, 16],
      favorableDirection: tr("Right favors Full", "越右越有利于完整方法"),
    }),
    drawForestPanel({
      x: 830,
      y: 190,
      width: 690,
      height: 690,
      titleValue: tr("Collision-rate difference", "碰撞率差值"),
      rows: collisionRows,
      domain: [-16, 6],
      favorableDirection: tr("Left favors Full", "越左越有利于完整方法"),
    }),
    footer(),
  ].join("\n");
}

function figureFactorial(core) {
  const names = {
    B00_strong_nominal_mppi: tr("B00\nNominal", "B00\n强标称基线"),
    B10_learning_only: tr("B10\nLearning only", "B10\n仅学习包"),
    B01_probability_only: tr("B01\nProbability only", "B01\n仅概率包"),
    B11_full_proposed: tr("B11\nFull", "B11\n完整方法"),
  };
  const colors = {
    B00_strong_nominal_mppi: COLORS.baseline,
    B10_learning_only: COLORS.learning,
    B01_probability_only: COLORS.probability,
    B11_full_proposed: COLORS.full,
  };
  const arms = Object.keys(names);
  const lookup = {};
  core
    .filter((r) => r.split === "pooled" && ["safe_success", "collision"].includes(r.endpoint))
    .forEach((r) => {
      lookup[`${r.arm}:${r.endpoint}`] = Number(r.mean_or_rate);
    });

  function panel(x, titleValue, endpoint, max) {
    const width = 700;
    const height = 690;
    const out = [
      rect(x, 190, width, height, { fill: "#FBFCFE", stroke: COLORS.grid, strokeWidth: 2, radius: 18 }),
      text(x + 34, 244, titleValue, { size: 29, weight: 700 }),
    ];
    const plot = { x: x + 80, y: 300, w: width - 130, h: 430 };
    for (let i = 0; i <= 5; i += 1) {
      const value = (max * i) / 5;
      const yy = plot.y + plot.h - (value / max) * plot.h;
      out.push(line(plot.x, yy, plot.x + plot.w, yy, { stroke: COLORS.grid, width: 1.5 }));
      out.push(text(plot.x - 15, yy + 7, `${Math.round(value * 100)}%`, {
        size: 17,
        fill: COLORS.muted,
        anchor: "end",
      }));
    }
    const gap = 24;
    const barW = (plot.w - gap * 5) / 4;
    arms.forEach((arm, i) => {
      const value = lookup[`${arm}:${endpoint}`];
      const bh = (value / max) * plot.h;
      const bx = plot.x + gap + i * (barW + gap);
      const by = plot.y + plot.h - bh;
      out.push(rect(bx, by, barW, bh, { fill: colors[arm], radius: 7 }));
      out.push(text(bx + barW / 2, by - 13, pct(value), {
        size: 21,
        weight: 700,
        anchor: "middle",
        fill: colors[arm],
      }));
      const [l1, l2] = names[arm].split("\n");
      out.push(text(bx + barW / 2, plot.y + plot.h + 40, l1, {
        size: 19,
        weight: 700,
        anchor: "middle",
      }));
      out.push(text(bx + barW / 2, plot.y + plot.h + 66, l2, {
        size: 17,
        fill: COLORS.muted,
        anchor: "middle",
      }));
    });
    return out.join("\n");
  }

  return [
    baseSvg(
      tr(
        "Factorial mechanism comparison exposes the safety–completion trade-off",
        "四个核心 Arm 揭示安全性与完成率之间的权衡"
      ),
      tr(
        "All four core arms use the same 360 paired seeds; L = learning package, P = probabilistic temporal package",
        "四个 Arm 共用 360 个配对种子；L 表示学习包，P 表示概率时序包"
      )
    ),
    panel(80, tr("Safe success rate", "安全成功率"), "safe_success", 1),
    panel(820, tr("Collision rate", "碰撞率"), "collision", 0.25),
    rect(80, 900, 1440, 48, { fill: COLORS.softGreen, radius: 12 }),
    text(800, 932, tr(
      "Probability-only is safest but conservative; adding learning recovers completion while retaining most collision reduction.",
      "仅概率包最安全但较保守；加入学习包可恢复完成率，同时保留大部分碰撞下降收益。"
    ), {
      size: 20,
      weight: 600,
      anchor: "middle",
      fill: "#176B4D",
    }),
    "</svg>",
  ].join("\n");
}

function figureAblations(ablation) {
  const labelMap = {
    B11_full_proposed_minus_A_full_ordinary_imm: tr("Ordinary IMM", "普通 IMM"),
    B11_full_proposed_minus_A_no_icode: tr("No ICODE", "移除 ICODE"),
    B11_full_proposed_minus_A_fixed_hss: tr("Fixed HSS", "固定 HSS"),
  };
  function rows(endpoint) {
    return ablation
      .filter((r) => r.split === "pooled" && r.endpoint === endpoint)
      .map((r) => ({
        label: labelMap[r.contrast],
        effect: 100 * Number(r.treatment_minus_control),
        ci: parseCi(r.tango_two_sided_95ci).map((v) => 100 * v),
        color: endpoint === "safe_success" ? COLORS.full : COLORS.collision,
      }));
  }
  return [
    baseSvg(
      tr(
        "Full-system ablations: directional gains, but wide paired intervals",
        "完整系统消融：效应方向有利，但配对置信区间仍较宽"
      ),
      tr(
        "B11 Full minus each ablation on the shared 140-seed subset; negative collision difference favors Full",
        "在共享的 140 种子子集上计算 B11 完整方法减各消融；碰撞差值为负时有利于完整方法"
      )
    ),
    drawForestPanel({
      x: 80,
      y: 190,
      width: 710,
      height: 690,
      titleValue: tr("Safe success: Full − ablation", "安全成功率：完整方法 − 消融"),
      rows: rows("safe_success"),
      domain: [-8, 14],
      favorableDirection: tr("Right favors Full", "越右越有利于完整方法"),
      xLabel: tr(
        "Paired Full − ablation difference (percentage points)",
        "配对差值：完整方法 − 消融（百分点）"
      ),
    }),
    drawForestPanel({
      x: 830,
      y: 190,
      width: 690,
      height: 690,
      titleValue: tr("Collision: Full − ablation", "碰撞率：完整方法 − 消融"),
      rows: rows("collision"),
      domain: [-12, 6],
      favorableDirection: tr("Left favors Full", "越左越有利于完整方法"),
      xLabel: tr(
        "Paired Full − ablation difference (percentage points)",
        "配对差值：完整方法 − 消融（百分点）"
      ),
    }),
    footer(
      tr(
        "Single dynamic obstacle v4 · full-system ablations · shared 140-seed subset",
        "单动态障碍 v4 · 完整系统消融 · 共享 140 种子子集"
      ),
      tr(
        "Error bars: paired Tango two-sided 95% CI",
        "误差线：配对 Tango 双侧 95% 置信区间"
      )
    ),
  ].join("\n");
}

function figureMechanismTradeoff(continuous) {
  const wanted = [
    ["minimum_clearance", tr("Minimum clearance", "最小净空"), tr("cm", "厘米"), 100],
    ["conflict_q05_clearance", tr("Conflict-window q05 clearance", "冲突窗口 q05 净空"), tr("cm", "厘米"), 100],
    ["stuck_steps", tr("Stuck steps", "停滞步数"), tr("steps", "步"), 1],
    ["release_delay_max_s", tr("Maximum release delay", "最大风险解除延迟"), tr("s", "秒"), 1],
  ];
  const lookup = Object.fromEntries(
    continuous.filter((r) => r.split === "pooled").map((r) => [r.endpoint, r])
  );
  const panels = [];
  wanted.forEach(([endpoint, label, unit, multiplier], index) => {
    const row = lookup[endpoint];
    const col = index % 2;
    const rix = Math.floor(index / 2);
    const x = 90 + col * 740;
    const y = 190 + rix * 340;
    const width = 680;
    const height = 290;
    const a = Number(row.control_mean) * multiplier;
    const b = Number(row.treatment_mean) * multiplier;
    const max = Math.max(a, b, 0.01) * 1.28;
    const scale = (v) => x + 105 + (v / max) * (width - 200);
    const yy = y + 150;
    panels.push(rect(x, y, width, height, { fill: "#FBFCFE", stroke: COLORS.grid, strokeWidth: 2, radius: 18 }));
    panels.push(text(x + 32, y + 48, label, { size: 27, weight: 700 }));
    panels.push(line(x + 105, yy, x + width - 95, yy, { stroke: COLORS.grid, width: 8 }));
    panels.push(line(scale(a), yy, scale(b), yy, { stroke: COLORS.full, width: 8 }));
    panels.push(circle(scale(a), yy, 14, { fill: COLORS.baseline, stroke: COLORS.paper, strokeWidth: 3 }));
    panels.push(circle(scale(b), yy, 14, { fill: COLORS.full, stroke: COLORS.paper, strokeWidth: 3 }));
    const clearanceUnit = unit === "cm" || unit === "厘米";
    panels.push(text(scale(a), yy + 48, `${tr("Baseline", "基线")} ${a.toFixed(clearanceUnit ? 1 : 2)}`, {
      size: 18,
      fill: COLORS.baseline,
      weight: 700,
      anchor: "middle",
    }));
    panels.push(text(scale(b), yy - 30, `${tr("Full", "完整方法")} ${b.toFixed(clearanceUnit ? 1 : 2)}`, {
      size: 18,
      fill: COLORS.full,
      weight: 700,
      anchor: "middle",
    }));
    const diff = b - a;
    panels.push(text(x + width / 2, y + height - 28, `${tr("Change", "变化")}: ${diff >= 0 ? "+" : "−"}${Math.abs(diff).toFixed(clearanceUnit ? 1 : 2)} ${unit}`, {
      size: 20,
      weight: 700,
      anchor: "middle",
      fill: index < 2 ? COLORS.success : COLORS.danger,
    }));
  });
  return [
    baseSvg(
      tr(
        "Mechanism diagnostics explain why safety did not raise safe completion",
        "机制诊断解释了安全性提升为何没有转化为更高完成率"
      ),
      tr(
        "Full creates substantially more clearance, but also increases post-conflict delay and stuck behavior",
        "完整方法显著扩大净空，但同时增加冲突后的解除延迟与停滞行为"
      )
    ),
    panels.join("\n"),
    rect(90, 880, 1420, 48, { fill: COLORS.softOrange, radius: 12 }),
    text(800, 912, tr(
      "Interpretation: collision avoidance improved; some rescued episodes were converted into safe non-completion.",
      "解释：避碰能力得到提升，但部分避免碰撞的 episode 被转化为“安全但未完成”。"
    ), {
      size: 20,
      weight: 600,
      anchor: "middle",
      fill: "#9C3E0A",
    }),
    "</svg>",
  ].join("\n");
}

async function writeFigure(browser, filename, svg) {
  const svgPath = path.join(OUTPUT, `${filename}.svg`);
  const pngPath = path.join(OUTPUT, `${filename}.png`);
  const pdfPath = path.join(OUTPUT, `${filename}.pdf`);
  fs.writeFileSync(svgPath, svg, "utf8");
  await sharp(Buffer.from(svg)).png({ quality: 100 }).toFile(pngPath);
  const page = await browser.newPage({ viewport: { width: W, height: H } });
  await page.setContent(
    `<!doctype html><html><head><style>html,body{margin:0;padding:0;width:${W}px;height:${H}px;background:white}</style></head><body>${svg}</body></html>`,
    { waitUntil: "load" }
  );
  await page.pdf({
    path: pdfPath,
    width: `${W}px`,
    height: `${H}px`,
    printBackground: true,
    margin: { top: "0", right: "0", bottom: "0", left: "0" },
  });
  await page.close();
  return [svgPath, pngPath, pdfPath];
}

function sha256(file) {
  return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}

async function main() {
  fs.mkdirSync(OUTPUT, { recursive: true });
  const primary = readJson("primary_confirmatory_analysis.json");
  const core = readCsv("core_descriptive_table.csv");
  const ablation = readCsv("full_system_ablation_effects.csv");
  const continuous = readCsv("continuous_full_vs_baseline.csv");

  const figures = [
    ["01_primary_outcomes", figurePrimary(primary)],
    ["02_id_ood_paired_effects", figureSplitEffects(primary)],
    ["03_factorial_mechanism_arms", figureFactorial(core)],
    ["04_full_system_ablations", figureAblations(ablation)],
    ["05_safety_mobility_tradeoff", figureMechanismTradeoff(continuous)],
  ];

  const edge = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
  const browser = await chromium.launch({ executablePath: edge, headless: true });
  const written = [];
  try {
    for (const [name, svg] of figures) {
      written.push(...(await writeFigure(browser, name, svg)));
    }
  } finally {
    await browser.close();
  }

  const sourceFiles = [
    "primary_confirmatory_analysis.json",
    "core_descriptive_table.csv",
    "full_system_ablation_effects.csv",
    "continuous_full_vs_baseline.csv",
    "analysis_integrity_audit.json",
  ].map((name) => {
    const file = path.join(ANALYSIS, name);
    return { name, sha256: sha256(file) };
  });

  const manifest = {
    generated_at: new Date().toISOString(),
    language: LANG,
    source_analysis: ANALYSIS,
    source_files: sourceFiles,
    figures: written.map((file) => ({
      file: path.basename(file),
      sha256: sha256(file),
    })),
  };
  fs.writeFileSync(
    path.join(OUTPUT, "figure_manifest.json"),
    JSON.stringify(manifest, null, 2) + "\n",
    "utf8"
  );

  const pooled = primary.by_split.pooled;
  const readme = `# 单动态障碍 v4 正式实验可视化

数据来源：封存后的正式分析包（共 1860 个 episode jobs；核心 B00/B10/B01/B11 为 360 个严格配对 seeds）。

## 主要结果

- Safe success：Baseline ${pct(pooled.success_rate.control_rate)}，Full ${pct(pooled.success_rate.treatment_rate)}，差值 ${pp(pooled.success_rate.treatment_minus_control)}，${pLabel(pooled.success_rate.exact_mcnemar_two_sided_p)}。
- Collision rate：Baseline ${pct(pooled.collision_rate.control_rate)}，Full ${pct(pooled.collision_rate.treatment_rate)}，差值 ${pp(pooled.collision_rate.treatment_minus_control)}，${pLabel(pooled.collision_rate.exact_mcnemar_two_sided_p)}。
- Full 显著降低了碰撞，并扩大了冲突净空；但 safe success 没有得到统计确认的提升。
- 机制诊断显示，Full 同时增加了 stuck steps 和风险解除后的 release delay，因此部分“避免碰撞”的 episode 转化成安全未完成。
- 正式实时性结论仍需独立、独占资源的 timing cohort；这里不把行为矩阵中的并发耗时作为论文实时性证据。

## 文件

1. 01_primary_outcomes：主终点（成功率、碰撞率）
2. 02_id_ood_paired_effects：ID/OOD 配对效应与 95% CI
3. 03_factorial_mechanism_arms：四个核心机制 Arm
4. 04_full_system_ablations：ordinary IMM / no ICODE / fixed HSS 消融
5. 05_safety_mobility_tradeoff：净空收益与机动性代价

每张图均提供 PNG（展示）、SVG（矢量编辑）和 PDF（论文插图）。figure_manifest.json 记录输入与输出 SHA-256。
`;
  fs.writeFileSync(path.join(OUTPUT, "README.md"), readme, "utf8");
  console.log(JSON.stringify({ output: OUTPUT, figures: figures.length, files: written.length + 2 }, null, 2));
}

main().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
