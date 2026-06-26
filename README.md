[econ_chart_extractor_v2.py](https://github.com/user-attachments/files/29365866/econ_chart_extractor_v2.py)
# -V2#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EconChartExtractor V2 — 多引擎共识 + 置信度分级的图表提取器

改进要点（基于 GPT-5 方案 + approval-seal-analyzer 多引擎思路）：
  L1: 几何结构引擎 — 横线/竖线/曲线/边框检测（原版改进：动态 kernel）
  L2: 颜色/图像区域引擎 — 填充区域、阴影、色块、点阵检测（新增）
  L3: 文本布局引擎 — 图题、轴标签、图号检测（新增，启发式）
  L4: 深度视觉网络 — ResNet-Lite + SE 注意力（升级原版 ChartPatchNet）
  L5: 置信度评分 + 三级分流 — trusted / uncertain / unreliable

依赖:
    pip install pymupdf opencv-python numpy pillow torch tqdm scikit-image
可选:
    pip install ultralytics    # YOLO/RT-DETR
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, random_split
except Exception:
    torch = None
    nn = None
    F = None
    Dataset = object
    DataLoader = None
    random_split = None

try:
    from skimage.feature import graycomatrix, graycoprops
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# ========================================================================
# 数据结构
# ========================================================================

@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float = 0.0
    source: str = "cv"
    engine_votes: int = 0          # 有多少引擎投票此框
    confidence: float = 0.0        # 0-100 综合置信度
    confidence_tier: str = ""      # trusted / uncertain / unreliable
    feat: dict = field(default_factory=dict)

    def clamp(self, w: int, h: int) -> "Box":
        self.x1 = max(0, min(int(self.x1), w - 1))
        self.y1 = max(0, min(int(self.y1), h - 1))
        self.x2 = max(1, min(int(self.x2), w))
        self.y2 = max(1, min(int(self.y2), h))
        if self.x2 <= self.x1:
            self.x2 = min(w, self.x1 + 1)
        if self.y2 <= self.y1:
            self.y2 = min(h, self.y1 + 1)
        return self

    @property
    def w(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def h(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def expand(self, pad: int, w: int, h: int) -> "Box":
        return Box(self.x1 - pad, self.y1 - pad, self.x2 + pad, self.y2 + pad,
                   self.score, self.source, self.engine_votes).clamp(w, h)

    def to_xywh(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.w, self.h


def iou(a: Box, b: Box) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    return inter / float(a.area + b.area - inter + 1e-9)


def giou(a: Box, b: Box) -> float:
    """Generalized IoU — 对非重叠框更友好，负值表示框之间有间隙。"""
    x1_i = max(a.x1, b.x1)
    y1_i = max(a.y1, b.y1)
    x2_i = min(a.x2, b.x2)
    y2_i = min(a.y2, b.y2)
    inter = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    if inter == 0:
        iou_v = 0.0
    else:
        iou_v = inter / float(a.area + b.area - inter + 1e-9)

    # enclosing box
    x1_e = min(a.x1, b.x1)
    y1_e = min(a.y1, b.y1)
    x2_e = max(a.x2, b.x2)
    y2_e = max(a.y2, b.y2)
    enclosing = max(1, (x2_e - x1_e) * (y2_e - y1_e))

    return iou_v - (enclosing - (a.area + b.area - inter)) / enclosing


def contained_ratio(inner: Box, outer: Box) -> float:
    x1, y1 = max(inner.x1, outer.x1), max(inner.y1, outer.y1)
    x2, y2 = min(inner.x2, outer.x2), min(inner.y2, outer.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / float(inner.area + 1e-9)


def nms(boxes: Sequence[Box], iou_thr: float = 0.30) -> List[Box]:
    boxes = sorted(boxes, key=lambda b: (b.score, b.area), reverse=True)
    keep: List[Box] = []
    for b in boxes:
        discard = False
        for k in keep:
            if iou(b, k) > iou_thr:
                discard = True
                break
            if contained_ratio(b, k) > 0.88 and k.area >= b.area:
                discard = True
                break
        if not discard:
            keep.append(b)
    return keep


# ========================================================================
# PDF 渲染
# ========================================================================

def render_pdf_pages(pdf_path: Path, dpi: int = 360,
                     first: int = 1, last: Optional[int] = None
                     ) -> Iterable[Tuple[int, np.ndarray]]:
    doc = fitz.open(str(pdf_path))
    last_page = len(doc) if last is None else min(last, len(doc))
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page_idx in range(max(1, first) - 1, last_page):
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        yield page_idx + 1, arr.copy()
    doc.close()


# ========================================================================
# 图像预处理增强
# ========================================================================

def enhance_image(rgb: np.ndarray) -> np.ndarray:
    """CLAHE + 轻微锐化，增强低对比度扫描页。"""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ========================================================================
# L1: 几何结构引擎 (改进版)
# ========================================================================

def binarize_ink(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adap = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 41, 18)
    ink = cv2.bitwise_or(otsu, adap)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    return ink


def line_masks(ink: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = ink.shape[:2]
    hk = max(34, w // 55)
    vk = max(34, h // 55)
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))
    edges = cv2.Canny(ink, 80, 160)
    curve = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    return horizontal, vertical, curve


def remove_page_rules(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    out = mask.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw > 0.72 * w and ch < max(8, 0.008 * h):
            cv2.rectangle(out, (x, y), (x + cw, y + ch), 0, -1)
    return out


def _max_component_extent(mask: np.ndarray, horizontal: bool) -> int:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        best = max(best, w if horizontal else h)
    return int(best)


def estimate_chart_density(ink: np.ndarray) -> float:
    """估算页面中图表区域的密度，用于动态调整膨胀 kernel。"""
    h, w = ink.shape[:2]
    # 分块统计
    grid_h, grid_w = max(1, h // 200), max(1, w // 200)
    density_map = np.zeros((grid_h, grid_w))
    for i in range(grid_h):
        for j in range(grid_w):
            y1, y2 = i * 200, min((i + 1) * 200, h)
            x1, x2 = j * 200, min((j + 1) * 200, w)
            density_map[i, j] = np.count_nonzero(ink[y1:y2, x1:x2]) / ((y2 - y1) * (x2 - x1))
    # 中位数以上块的密度（排除正文密集区）
    nonzero = density_map[density_map > 0.001]
    if len(nonzero) == 0:
        return 0.3
    return float(max(0.2, min(0.8, np.median(nonzero) * 3)))


def geometric_engine(rgb: np.ndarray, min_size_px: int = 42,
                     max_area_frac: float = 0.36) -> List[Box]:
    """
    L1 引擎：基于几何结构的候选框生成。
    改进点：动态 kernel、两步膨胀策略、密度感知。
    """
    h, w = rgb.shape[:2]
    enhanced = enhance_image(rgb)
    ink = binarize_ink(enhanced)
    hmask, vmask, curve = line_masks(ink)
    hmask = remove_page_rules(hmask)
    structural = cv2.bitwise_or(hmask, vmask)

    # 动态 kernel：根据页面密度调整
    density = estimate_chart_density(ink)
    base_k = max(13, min(45, w // 95))
    # 密度高 → kernel 稍小（图表多，避免合并）；密度低 → kernel 稍大（图表稀疏）
    dyn_k = int(base_k * (1.3 - density))
    dyn_k = max(8, min(50, dyn_k))

    kx = dyn_k
    ky = dyn_k

    # 两步膨胀：先小膨胀合并轴+标签，再大膨胀识别图表范围
    group1 = cv2.dilate(structural,
                        cv2.getStructuringElement(cv2.MORPH_RECT, (kx // 2, ky // 2)),
                        iterations=1)
    group1 = cv2.morphologyEx(group1, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (kx // 2, ky // 2)),
                              iterations=1)

    # 合并曲线边缘
    structural2 = cv2.bitwise_or(group1, curve)
    grouped = cv2.dilate(structural2,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)),
                         iterations=1)
    grouped = cv2.morphologyEx(grouped, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)),
                               iterations=1)

    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Box] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        b = Box(x, y, x + bw, y + bh, source="geom")
        if b.area <= 0 or b.area > max_area_frac * w * h:
            continue
        if b.w > 0.70 * w and b.h < 0.06 * h:
            continue
        if b.y1 < 0.18 * h and b.w > 0.35 * w and b.h < 0.18 * h:
            continue
        feat = box_features(b, ink, hmask, vmask, curve)
        score = heuristic_score_v2(feat)
        if score >= 0.12:  # 降低阈值，后续由多引擎共识过滤
            b.score = score
            b.feat = feat
            boxes.append(b)

    # 第二阶段：dense 区域补充
    dense_k = max(5, kx // 2)
    dense = cv2.morphologyEx(ink, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (dense_k, dense_k)),
                             iterations=1)
    contours, _ = cv2.findContours(dense, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        b = Box(x, y, x + bw, y + bh, source="geom_dense")
        if b.area < 0.00012 * w * h or b.area > 0.10 * w * h:
            continue
        if b.w < 24 or b.h < 24:
            continue
        feat = box_features(b, ink, hmask, vmask, curve)
        if feat["curve_pixels"] > 80 and (feat["line_density"] > 0.005 or feat["ink_density"] > 0.20):
            b.score = max(0.12, heuristic_score_v2(feat))
            b.feat = feat
            boxes.append(b)

    return nms(boxes, iou_thr=0.20)


def box_features(box: Box, ink: np.ndarray, hmask: np.ndarray,
                 vmask: np.ndarray, curve: np.ndarray) -> dict:
    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    area = max(1, box.area)
    ink_roi = ink[y1:y2, x1:x2]
    h_roi = hmask[y1:y2, x1:x2]
    v_roi = vmask[y1:y2, x1:x2]
    c_roi = curve[y1:y2, x1:x2]
    ink_density = float(np.count_nonzero(ink_roi)) / area
    line_density = float(np.count_nonzero(h_roi) + np.count_nonzero(v_roi)) / area
    curve_density = float(np.count_nonzero(c_roi)) / area
    aspect = box.w / max(1, box.h)
    hd = cv2.dilate(h_roi, np.ones((3, 3), np.uint8), iterations=1)
    vd = cv2.dilate(v_roi, np.ones((3, 3), np.uint8), iterations=1)
    intersections = int(np.count_nonzero(cv2.bitwise_and(hd, vd)))
    max_h_extent = _max_component_extent(h_roi, horizontal=True)
    max_v_extent = _max_component_extent(v_roi, horizontal=False)
    return {
        "ink_density": ink_density,
        "line_density": line_density,
        "curve_density": curve_density,
        "aspect": aspect,
        "area_frac": area / float(ink.shape[0] * ink.shape[1]),
        "h_pixels": int(np.count_nonzero(h_roi)),
        "v_pixels": int(np.count_nonzero(v_roi)),
        "curve_pixels": int(np.count_nonzero(c_roi)),
        "intersections": intersections,
        "max_h_extent": max_h_extent,
        "max_v_extent": max_v_extent,
        "max_h_ratio": max_h_extent / max(1, box.w),
        "max_v_ratio": max_v_extent / max(1, box.h),
    }


def heuristic_score_v2(feat: dict) -> float:
    """改进版启发式评分：更连续的加权，减少硬阈值。"""
    score = 0.0
    # 线条密度（核心信号，sigmoid-like曲线）
    score += 0.35 * (1 - math.exp(-feat["line_density"] * 30))

    # 曲线/边缘密度
    score += 0.08 * (1 - math.exp(-feat["curve_density"] * 5))

    # 坐标轴检测：连续分值而非硬阈值
    axis_score = 0.0
    if feat["max_h_ratio"] > 0.15 and feat["max_v_ratio"] > 0.15:
        axis_score += 0.25 * min(1.0, (feat["max_h_ratio"] - 0.15) / 0.5) * min(1.0, (feat["max_v_ratio"] - 0.15) / 0.5)
    if feat["intersections"] > 2:
        axis_score += 0.15 * min(1.0, feat["intersections"] / 15)
    if feat["h_pixels"] > 50 and feat["v_pixels"] > 50:
        axis_score += 0.05
    score += axis_score

    # 宽高比（经济学图通常在 0.3-4.2 范围）
    if 0.30 <= feat["aspect"] <= 4.2:
        score += 0.08
    elif 0.15 <= feat["aspect"] <= 7.0:
        score += 0.04

    # 面积占比（太小或太大扣分，用高斯型曲线）
    area_z = (feat["area_frac"] - 0.08) / 0.10
    score += 0.08 * math.exp(-area_z * area_z)

    # 正文惩罚（高ink密度但低line密度）
    if feat["ink_density"] > 0.15 and feat["line_density"] < 0.012:
        score -= 0.30
    elif feat["ink_density"] > 0.10 and feat["line_density"] < 0.008:
        score -= 0.15

    # 极小图惩罚
    if feat["h_pixels"] < 80 and feat["v_pixels"] < 80:
        score -= 0.20

    # 孤立组件惩罚
    if feat["intersections"] <= 1 and feat["max_h_ratio"] < 0.40 and feat["max_v_ratio"] < 0.40:
        score -= 0.18

    return float(max(0.0, min(1.0, score)))


# ========================================================================
# L2: 颜色/图像区域引擎 (新增)
# ========================================================================

def color_region_engine(rgb: np.ndarray, min_size_px: int = 42,
                        max_area_frac: float = 0.36) -> List[Box]:
    """
    L2 引擎：检测填充区域、阴影、色块、点阵。
    策略：颜色聚类 + 纹理分析 + 区域连通。
    """
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # 1. 检测非文字的大面积均匀区域（颜色/灰度块）
    # 用大kernel均值滤波，找到大面积平滑区域
    smooth = cv2.GaussianBlur(gray, (21, 21), 0)
    laplacian = cv2.Laplacian(smooth, cv2.CV_64F)
    lap_abs = np.abs(laplacian).astype(np.uint8)
    # 低梯度区域 = 大面积填充/阴影
    _, low_texture = cv2.threshold(lap_abs, 8, 255, cv2.THRESH_BINARY_INV)

    # 2. 检测中等灰度填充区域（常见于经济学图的阴影、填充）
    # 用自适应阈值找中等亮度块
    thresh_block = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 61, 12)
    # 反转找暗色填充
    thresh_block_dark = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 61, 12)

    # 3. 纹理分析：检测点阵/网格填充（用局部标准差）
    block_size = 15
    local_std = np.zeros_like(gray, dtype=np.float32)
    for y in range(0, h - block_size, block_size // 2):
        for x in range(0, w - block_size, block_size // 2):
            patch = gray[y:y + block_size, x:x + block_size]
            local_std[y:y + block_size, x:x + block_size] = patch.std()
    # 中等标准差 = 点阵/纹理填充（文字区很高，空白区很低）
    texture_fill = ((local_std > 25) & (local_std < 80)).astype(np.uint8) * 255
    texture_fill = cv2.morphologyEx(texture_fill, cv2.MORPH_CLOSE,
                                    np.ones((5, 5), np.uint8), iterations=2)

    # 4. 合并所有填充区域信号
    region_mask = cv2.bitwise_or(low_texture, thresh_block)
    region_mask = cv2.bitwise_or(region_mask, thresh_block_dark)
    region_mask = cv2.bitwise_or(region_mask, texture_fill)

    # 形态学闭合：把散落的填充小块连成区域
    k_close = max(15, min(40, w // 80))
    region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close)),
                                   iterations=2)

    # 膨胀让相邻区域合并
    region_mask = cv2.dilate(region_mask,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (k_close // 2, k_close // 2)),
                             iterations=1)

    # 5. 提取候选框
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Box] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        b = Box(x, y, x + bw, y + bh, source="color")
        if b.area <= 0:
            continue
        if b.w < min_size_px or b.h < min_size_px:
            continue
        if b.area > max_area_frac * w * h:
            continue
        # 排除纯文字块（文字区域通常不是纯填充）
        patch = gray[b.y1:b.y2, b.x1:b.x2]
        if patch.size == 0:
            continue
        # 如果区域内拉普拉斯方差太低（太平），且没有结构线，可能是空白/背景
        lap_var = cv2.Laplacian(patch, cv2.CV_64F).var()
        if lap_var < 5:
            continue

        # L2 引擎的评分：基于填充面积和纹理复杂度
        fill_density = np.count_nonzero(region_mask[b.y1:b.y2, b.x1:b.x2]) / max(1, b.area)
        score = 0.15 + 0.25 * fill_density + 0.10 * min(1.0, lap_var / 50)
        b.score = float(min(1.0, score))
        boxes.append(b)

    return nms(boxes, iou_thr=0.15)


# ========================================================================
# L3: 文本布局引擎 (新增)
# ========================================================================

def text_layout_engine(rgb: np.ndarray, geometric_boxes: List[Box]) -> List[Box]:
    """
    L3 引擎：基于文本布局的图表辅助定位。
    检测 Figure/图表编号、轴标签等文本提示，给邻近的几何候选框加分。
    不单独产生新框，而是增强/筛选已有框。
    """
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # 用 MSER 或简单形态学检测文字块
    # 简化版：大kernel形态学找大文字块
    text_k = max(8, min(20, w // 100))
    text_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 31, 8)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (text_k, text_k // 2)),
                                 iterations=1)

    contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        tb = Box(x, y, x + bw, y + bh, source="text")
        if tb.w < 24 or tb.h < 8:
            continue
        if tb.w > 0.60 * w:
            continue  # 不是标题
        text_boxes.append(tb)

    enhanced_boxes: List[Box] = []
    for gbox in geometric_boxes:
        gbox = Box(gbox.x1, gbox.y1, gbox.x2, gbox.y2,
                   gbox.score, gbox.source, gbox.engine_votes, gbox.confidence,
                   gbox.confidence_tier, gbox.feat)

        # 在图表框上方/下方附近找文字块（图题通常在下方）
        text_nearby = 0
        for tb in text_boxes:
            # 文字块在框下方30px内
            if tb.y1 >= gbox.y2 and tb.y1 < gbox.y2 + max(30, h * 0.03):
                if abs(tb.x1 - gbox.x1) < gbox.w * 0.5:
                    text_nearby += 1
                    break
            # 文字块在框上方30px内
            if tb.y2 <= gbox.y1 and tb.y2 > gbox.y1 - max(30, h * 0.03):
                if abs(tb.x1 - gbox.x1) < gbox.w * 0.5:
                    text_nearby += 1
                    break

        if text_nearby > 0:
            gbox.score *= 1.15  # 有图题暗示，加分
            gbox.score = min(1.0, gbox.score)
            gbox.source += "+text"

        enhanced_boxes.append(gbox)

    return enhanced_boxes


# ========================================================================
# 边框精修与分割 (改进版)
# ========================================================================

def refine_bbox_v2(rgb: np.ndarray, box: Box, crop_padding: int = 10) -> Box:
    """
    改进版边界精修：
    1. 在候选框附近找连通墨迹
    2. 检测内部间隙（判断是否需要分割）
    3. 用投影法找最佳边界
    """
    H, W = rgb.shape[:2]
    grow = int(max(10, min(W, H) * 0.015))
    outer = box.expand(grow, W, H)
    roi = rgb[outer.y1:outer.y2, outer.x1:outer.x2]
    if roi.size == 0:
        return box

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    ink = binarize_ink(roi)
    hmask, vmask, curve = line_masks(ink)
    structure = cv2.bitwise_or(cv2.bitwise_or(hmask, vmask), curve)
    st = cv2.dilate(structure, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)
    contours, _ = cv2.findContours(st, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    core = Box(box.x1 - outer.x1, box.y1 - outer.y1,
               box.x2 - outer.x1, box.y2 - outer.y1).clamp(outer.w, outer.h)
    union_mask = np.zeros_like(ink)
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        cb = Box(x, y, x + bw, y + bh)
        if iou(cb, core) > 0.01 or contained_ratio(cb, core) > 0.20 or contained_ratio(core, cb) > 0.20:
            cv2.drawContours(union_mask, [c], -1, 255, -1)

    if np.count_nonzero(union_mask) == 0:
        union_mask = np.ones_like(ink) * 255

    # 适度膨胀纳入标签
    near = cv2.dilate(union_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)), iterations=1)
    selected_ink = cv2.bitwise_and(ink, near)

    ys, xs = np.where(selected_ink > 0)
    if len(xs) == 0:
        return box
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    refined = Box(outer.x1 + x1 - crop_padding,
                  outer.y1 + y1 - crop_padding,
                  outer.x1 + x2 + crop_padding,
                  outer.y1 + y2 + crop_padding,
                  score=box.score, source=box.source).clamp(W, H)

    if refined.area > max(box.area * 5.0, 0.45 * W * H):
        return box.expand(crop_padding, W, H)
    return refined


def detect_internal_gaps(rgb: np.ndarray, box: Box) -> List[Tuple[int, int]]:
    """
    检测候选框内部的水平/垂直间隙。
    返回可能的切分位置列表 [(pos, axis)]，axis=0为水平，axis=1为垂直。
    用于判断是否需要将一个大框分割为多个独立图表。
    """
    H, W = rgb.shape[:2]
    gray = cv2.cvtColor(rgb[box.y1:box.y2, box.x1:box.x2], cv2.COLOR_RGB2GRAY)
    bh, bw = gray.shape

    gaps = []

    # 水平投影：找大段空白行
    h_proj = np.sum(gray < 200, axis=1).astype(np.float32)  # 暗像素
    h_proj = h_proj / max(1, bw)
    # 空白行 = 暗像素密度 < 0.02
    blank_rows = h_proj < 0.02
    # 找连续空白行段
    blank_runs = []
    start = None
    for i, is_blank in enumerate(blank_rows):
        if is_blank and start is None:
            start = i
        elif not is_blank and start is not None:
            run_len = i - start
            if run_len > max(15, bh * 0.03):  # 至少占框高3%的空白
                blank_runs.append((start, i - 1, run_len))
            start = None
    if start is not None:
        run_len = len(blank_rows) - start
        if run_len > max(15, bh * 0.03):
            blank_runs.append((start, len(blank_rows) - 1, run_len))

    for rs, re, rl in blank_runs:
        mid = (rs + re) // 2
        # 确保不在框的边缘
        if mid > bh * 0.15 and mid < bh * 0.85:
            gaps.append((box.y1 + mid, 0))  # axis=0 = 水平切分

    # 垂直投影：找大段空白列
    v_proj = np.sum(gray < 200, axis=0).astype(np.float32)
    v_proj = v_proj / max(1, bh)
    blank_cols = v_proj < 0.02
    blank_runs_v = []
    start = None
    for i, is_blank in enumerate(blank_cols):
        if is_blank and start is None:
            start = i
        elif not is_blank and start is not None:
            run_len = i - start
            if run_len > max(15, bw * 0.03):
                blank_runs_v.append((start, i - 1, run_len))
            start = None
    if start is not None:
        run_len = len(blank_cols) - start
        if run_len > max(15, bw * 0.03):
            blank_runs_v.append((start, len(blank_cols) - 1, run_len))

    for rs, re, rl in blank_runs_v:
        mid = (rs + re) // 2
        if mid > bw * 0.15 and mid < bw * 0.85:
            gaps.append((box.x1 + mid, 1))  # axis=1 = 垂直切分

    return gaps


def split_by_gaps(box: Box, gaps: List[Tuple[int, int]]) -> List[Box]:
    """根据检测到的间隙将框分割为子框。"""
    if not gaps:
        return [box]

    # 按位置排序，分别处理水平和垂直切分
    h_splits = sorted([p for p, a in gaps if a == 0])
    v_splits = sorted([p for p, a in gaps if a == 1])

    result = [box]
    for split_y in h_splits:
        new_result = []
        for b in result:
            if b.y1 < split_y < b.y2:
                upper = Box(b.x1, b.y1, b.x2, split_y, b.score, b.source + "+split")
                lower = Box(b.x1, split_y, b.x2, b.y2, b.score, b.source + "+split")
                if upper.h > 30:
                    new_result.append(upper)
                if lower.h > 30:
                    new_result.append(lower)
            else:
                new_result.append(b)
        result = new_result

    for split_x in v_splits:
        new_result = []
        for b in result:
            if b.x1 < split_x < b.x2:
                left = Box(b.x1, b.y1, split_x, b.y2, b.score, b.source + "+split")
                right = Box(split_x, b.y1, b.x2, b.y2, b.score, b.source + "+split")
                if left.w > 30:
                    new_result.append(left)
                if right.w > 30:
                    new_result.append(right)
            else:
                new_result.append(b)
        result = new_result

    return result


# ========================================================================
# L4: 深度视觉网络 (ResNet-Lite + SE Attention)
# ========================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 注意力块"""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(x).view(x.size(0), -1, 1, 1)
        return x * scale


class ResidualBlock(nn.Module):
    """轻量残差块 with SE attention"""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expansion: float = 1.0):
        super().__init__()
        mid_ch = int(out_ch * expansion)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class ChartResNetLite(nn.Module):
    """
    ResNet-Lite 图表分类器。
    比原版 ChartPatchNet 更深，带 SE attention 和残差连接。
    输出 4 分类：chart / not_chart / decorative / text_block
    """
    def __init__(self, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Stage 1
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 48, stride=1, expansion=1.5),
            ResidualBlock(48, 48, stride=1, expansion=1.5),
        )
        # Stage 2
        self.stage2 = nn.Sequential(
            ResidualBlock(48, 96, stride=2, expansion=1.5),
            ResidualBlock(96, 96, stride=1, expansion=1.5),
        )
        # Stage 3
        self.stage3 = nn.Sequential(
            ResidualBlock(96, 192, stride=2, expansion=1.5),
            ResidualBlock(192, 192, stride=1, expansion=1.5),
        )
        # Stage 4
        self.stage4 = nn.Sequential(
            ResidualBlock(192, 320, stride=2, expansion=1.5),
            ResidualBlock(320, 320, stride=1, expansion=1.5),
        )

        # Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(320, num_classes)

        # 质量回归头
        self.quality_head = nn.Sequential(
            nn.Linear(320, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, return_features: bool = False):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        features = self.pool(x)
        feats = features.view(features.size(0), -1)
        dropped = self.dropout(feats)
        logits = self.fc(dropped)
        quality = self.quality_head(feats)
        if return_features:
            return logits, quality, feats
        return logits, quality


# ---- 兼容原版 -----

class ChartPatchNet(nn.Module):
    """原版小型 CNN（保持向后兼容）"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(96, 160, 3, padding=1), nn.BatchNorm2d(160), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(160, 2))

    def forward(self, x):
        return self.classifier(self.features(x))


# ---- 训练与推理 ----

def preprocess_patch(rgb_patch: np.ndarray, size: int = 224) -> "torch.Tensor":
    img = Image.fromarray(rgb_patch).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


class PatchFolder(Dataset):
    def __init__(self, label_dir: Path, size: int = 224):
        self.items: List[Tuple[Path, int]] = []
        self.size = size
        for cls, label in [("not_chart", 0), ("chart", 1), ("decorative", 2), ("text_block", 3)]:
            d = label_dir / cls
            if not d.exists():
                continue
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                self.items.extend((p, label) for p in d.glob(ext))
        if not self.items:
            # 至少要有 chart 和 not_chart
            for cls, label in [("not_chart", 0), ("chart", 1)]:
                d = label_dir / cls
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                    self.items.extend((p, label) for p in d.glob(ext))
        if not self.items:
            raise RuntimeError(f"未在 {label_dir}/chart 与 {label_dir}/not_chart 中找到训练图片。")
        random.shuffle(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        p, y = self.items[idx]
        img = np.asarray(Image.open(p).convert("RGB"))
        return preprocess_patch(img, self.size), torch.tensor(y, dtype=torch.long)


def load_model(path: Path, device: str = "cpu", use_lite: bool = True) -> nn.Module:
    if torch is None:
        raise RuntimeError("需要安装 torch 才能使用 --model。")
    if use_lite:
        model = ChartResNetLite(num_classes=2).to(device)
    else:
        model = ChartPatchNet().to(device)
    obj = torch.load(str(path), map_location=device)
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def cnn_score_v2(model: nn.Module, rgb: np.ndarray, box: Box, device: str = "cpu") -> Tuple[float, float]:
    """返回 (chart概率, 质量分)"""
    patch = rgb[box.y1:box.y2, box.x1:box.x2]
    if patch.size == 0:
        return 0.0, 0.0
    with torch.no_grad():
        x = preprocess_patch(patch).unsqueeze(0).to(device)
        if isinstance(model, ChartResNetLite):
            logits, quality = model(x)
            prob = F.softmax(logits, dim=1)[0, 1].item()
        else:
            logits = model(x)
            prob = F.softmax(logits, dim=1)[0, 1].item()
            quality = torch.tensor([0.5])
        return float(prob), float(quality.item())


def train_model_v2(label_dir: Path, model_out: Path, epochs: int = 30,
                   batch_size: int = 32, lr: float = 2e-4, seed: int = 7,
                   use_lite: bool = True):
    if torch is None:
        raise RuntimeError("需要安装 torch：pip install torch")
    random.seed(seed)
    torch.manual_seed(seed)
    ds = PatchFolder(label_dir)
    val_n = max(1, int(len(ds) * 0.15))
    train_n = len(ds) - val_n
    train_ds, val_ds = random_split(ds, [train_n, val_n],
                                    generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 2  # 默认二分类，如果文件夹有4类则自动扩展
    chart_dir = label_dir / "chart"
    if chart_dir.exists():
        num_classes = 2
    if (label_dir / "decorative").exists():
        num_classes = max(num_classes, 4)

    model = ChartResNetLite(num_classes=num_classes) if use_lite else ChartPatchNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_acc = 0.0
    model_out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}/{epochs} train"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            if isinstance(model, ChartResNetLite):
                logits, quality = model(x)
                cls_loss = F.cross_entropy(logits, y)
                # 质量回归辅助损失（真图表质量高，非图表质量低）
                quality_target = (y == 1).float().unsqueeze(1)
                quality_loss = F.mse_loss(quality, quality_target)
                loss = cls_loss + 0.1 * quality_loss
            else:
                loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= max(1, train_n)
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if isinstance(model, ChartResNetLite):
                    logits, _ = model(x)
                else:
                    logits = model(x)
                pred = logits.argmax(1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())
        acc = correct / max(1, total)
        print(f"epoch={epoch:03d} loss={train_loss:.4f} val_acc={acc:.4f}")
        if acc >= best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict(), "val_acc": best_acc,
                        "num_classes": num_classes, "arch": "resnet_lite" if use_lite else "small_cnn"},
                       str(model_out))
    print(f"已保存最佳模型：{model_out}，val_acc={best_acc:.4f}")


# ========================================================================
# L5: 置信度评分与三级分流
# ========================================================================

@dataclass
class ConfidenceScores:
    """五维度置信度评分"""
    geom_completeness: float = 0.0   # 几何完整度 (0-20)
    visual_contrast: float = 0.0     # 视觉对比度 (0-20)
    text_layout: float = 0.0         # 文本布局 (0-20)
    cnn_confidence: float = 0.0      # CNN 分类置信度 (0-20)
    engine_consistency: float = 0.0  # 引擎一致性 (0-20)
    total: float = 0.0
    tier: str = ""

    def to_dict(self) -> dict:
        return {
            "geom_completeness": round(self.geom_completeness, 1),
            "visual_contrast": round(self.visual_contrast, 1),
            "text_layout": round(self.text_layout, 1),
            "cnn_confidence": round(self.cnn_confidence, 1),
            "engine_consistency": round(self.engine_consistency, 1),
            "total": round(self.total, 1),
            "tier": self.tier,
        }


def compute_confidence(box: Box, page_rgb: np.ndarray, engine_votes: int,
                       cnn_prob: float = 0.0, has_text_nearby: bool = False) -> ConfidenceScores:
    """
    计算五维置信度评分。
    engine_votes: 有多少个独立引擎投票此框 (1-4)
    """
    cs = ConfidenceScores()
    feat = box.feat if box.feat else {}
    H, W = page_rgb.shape[:2]

    # 1. 几何完整度 (0-20)：基于坐标轴、线条密度、交点
    h_ratio = feat.get("max_h_ratio", 0)
    v_ratio = feat.get("max_v_ratio", 0)
    intersections = feat.get("intersections", 0)
    line_density = feat.get("line_density", 0)

    geom = 0.0
    geom += min(10, (h_ratio + v_ratio) * 8)  # 坐标轴信号
    geom += min(5, intersections * 0.8)        # 交点越多越像图
    geom += min(5, line_density * 80)          # 线条密度
    cs.geom_completeness = float(min(20, geom))

    # 2. 视觉对比度 (0-20)：基于区域内的灰度/颜色方差
    patch = page_rgb[box.y1:box.y2, box.x1:box.x2]
    if patch.size > 0:
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        contrast = gray_patch.std()
        # 边缘密度也反映对比度
        edges = cv2.Canny(gray_patch, 50, 150)
        edge_density = np.count_nonzero(edges) / max(1, box.area)
        cs.visual_contrast = float(min(20, contrast * 0.3 + edge_density * 200))
    else:
        cs.visual_contrast = 0.0

    # 3. 文本布局 (0-20)：图题/轴标签暗示
    cs.text_layout = 15.0 if has_text_nearby else 5.0
    if feat.get("curve_density", 0) > 0.01:
        cs.text_layout += 5.0  # 有曲线的图更可能有标签
    cs.text_layout = min(20, cs.text_layout)

    # 4. CNN 置信度 (0-20)
    cs.cnn_confidence = cnn_prob * 20.0

    # 5. 引擎一致性 (0-20)
    if engine_votes >= 4:
        cs.engine_consistency = 20.0
    elif engine_votes >= 3:
        cs.engine_consistency = 16.0
    elif engine_votes >= 2:
        cs.engine_consistency = 10.0
    else:
        cs.engine_consistency = 4.0

    cs.total = (cs.geom_completeness + cs.visual_contrast + cs.text_layout +
                cs.cnn_confidence + cs.engine_consistency)

    # 三级分流
    if cs.total >= 75:
        cs.tier = "trusted"
    elif cs.total >= 50:
        cs.tier = "uncertain"
    else:
        cs.tier = "unreliable"

    return cs


# ========================================================================
# 多引擎融合
# ========================================================================

def fuse_multi_engine(boxes_geom: List[Box], boxes_color: List[Box],
                      text_enhanced: List[Box]) -> List[Box]:
    """
    多引擎融合：基于 GIoU 和投票机制合并候选框。
    每个框记录有多少引擎投票（engine_votes）。
    """
    all_boxes = boxes_geom + boxes_color
    if not all_boxes:
        return text_enhanced

    # 按 IOU 聚类
    clusters: List[List[Box]] = []
    assigned = [False] * len(all_boxes)
    for i, b1 in enumerate(all_boxes):
        if assigned[i]:
            continue
        cluster = [b1]
        assigned[i] = True
        for j, b2 in enumerate(all_boxes):
            if assigned[j]:
                continue
            if iou(b1, b2) > 0.15:
                cluster.append(b2)
                assigned[j] = True
        clusters.append(cluster)

    fused: List[Box] = []
    for cluster in clusters:
        # 每个引擎源只计一票
        sources = set(b.source for b in cluster)
        engine_votes = len(sources)

        # 融合框：取并集包围盒的加权中心
        if len(cluster) == 1:
            base = cluster[0]
        else:
            scores = [b.score for b in cluster]
            total_score = sum(scores) + 1e-9
            x1 = int(sum(b.x1 * b.score / total_score for b in cluster))
            y1 = int(sum(b.y1 * b.score / total_score for b in cluster))
            x2 = int(sum(b.x2 * b.score / total_score for b in cluster))
            y2 = int(sum(b.y2 * b.score / total_score for b in cluster))
            base = Box(x1, y1, x2, y2,
                       score=sum(scores) / len(cluster) * min(1.0, engine_votes / 2.0),
                       source="+".join(sorted(sources)),
                       engine_votes=engine_votes)

        fused.append(base)

    # 合并 text_enhanced 的分数
    final_fused = []
    for fb in fused:
        matched = False
        for te in text_enhanced:
            if abs(fb.x1 - te.x1) < 10 and abs(fb.y1 - te.y1) < 10:
                fb.score = max(fb.score, te.score)
                fb.source = te.source
                fb.engine_votes = max(fb.engine_votes, te.engine_votes)
                matched = True
                break
        if not matched:
            # 也尝试按 IOU 匹配
            for te in text_enhanced:
                if iou(fb, te) > 0.5:
                    fb.score = max(fb.score, te.score)
                    fb.engine_votes += 1
                    break
        final_fused.append(fb)

    return nms(final_fused, iou_thr=0.25)


# ========================================================================
# YOLO 支持
# ========================================================================

def yolo_detect(rgb: np.ndarray, model_path: Path, conf: float = 0.25) -> List[Box]:
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("使用 --yolo-model 需要安装 ultralytics：pip install ultralytics") from e
    model = YOLO(str(model_path))
    results = model.predict(source=rgb, conf=conf, verbose=False)
    boxes: List[Box] = []
    if not results:
        return boxes
    H, W = rgb.shape[:2]
    r = results[0]
    if r.boxes is None:
        return boxes
    for xyxy, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        boxes.append(Box(x1, y1, x2, y2, score=float(c), source="yolo").clamp(W, H))
    return boxes


# ========================================================================
# 可视化
# ========================================================================

def draw_boxes_v2(rgb: np.ndarray, boxes: Sequence[Box], page_no: int) -> np.ndarray:
    vis = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    tier_colors = {
        "trusted": (0, 180, 0),      # 绿色
        "uncertain": (0, 200, 200),  # 黄色
        "unreliable": (0, 0, 255),   # 红色
        "": (0, 0, 255),
    }
    for idx, b in enumerate(boxes, 1):
        color = tier_colors.get(b.confidence_tier, (0, 0, 255))
        cv2.rectangle(vis, (b.x1, b.y1), (b.x2, b.y2), color,
                      max(2, min(7, rgb.shape[1] // 450)))
        tier_short = {"trusted": "T", "uncertain": "U", "unreliable": "L"}.get(b.confidence_tier, "?")
        label = f"p{page_no} #{idx} {b.score:.2f} [{tier_short}] v{b.engine_votes}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
        y_pos = max(0, b.y1 - th - 6)
        cv2.rectangle(vis, (b.x1, y_pos), (b.x1 + tw + 8, y_pos + th + 6), color, -1)
        cv2.putText(vis, label, (b.x1 + 4, y_pos + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


# ========================================================================
# 主流程
# ========================================================================

def extract_charts_v2(args):
    pdf_path = Path(args.pdf)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "annotated").mkdir(exist_ok=True)
    (out_dir / "crops").mkdir(exist_ok=True)
    meta_jsonl = out_dir / "metadata.jsonl"
    meta_csv = out_dir / "metadata.csv"
    conf_jsonl = out_dir / "confidence.jsonl"

    if meta_jsonl.exists() and not args.append:
        meta_jsonl.unlink()
    if meta_csv.exists() and not args.append:
        meta_csv.unlink()
    if conf_jsonl.exists() and not args.append:
        conf_jsonl.unlink()

    device = "cuda" if (torch is not None and torch.cuda.is_available() and not args.cpu) else "cpu"
    patch_model = load_model(Path(args.model), device=device) if args.model else None
    global_idx = 0
    total_pages = 0

    tier_counts = {"trusted": 0, "uncertain": 0, "unreliable": 0}

    for page_no, rgb in tqdm(render_pdf_pages(pdf_path, dpi=args.dpi,
                                               first=args.first, last=args.last),
                             desc="extract pages (V2)"):
        total_pages += 1
        H, W = rgb.shape[:2]

        # ---- 多引擎检测 ----
        # L1: 几何结构
        boxes_geom = geometric_engine(rgb, min_size_px=args.min_size,
                                      max_area_frac=args.max_area_frac)
        # L2: 颜色/图像区域（可选）
        boxes_color = []
        if not args.no_color_engine:
            boxes_color = color_region_engine(rgb, min_size_px=args.min_size,
                                              max_area_frac=args.max_area_frac)
        # L3: 文本布局增强
        text_enhanced = text_layout_engine(rgb, boxes_geom)

        # YOLO（可选）
        if args.yolo_model:
            yolo_boxes = yolo_detect(rgb, Path(args.yolo_model), conf=args.yolo_conf)
            for yb in yolo_boxes:
                boxes_geom.append(yb)

        # ---- 多引擎融合 ----
        candidates = fuse_multi_engine(boxes_geom, boxes_color, text_enhanced)
        candidates = nms(candidates, iou_thr=0.25)

        # ---- 间隙检测与分割 ----
        final_boxes_pre_split: List[Box] = []
        for b in candidates:
            # 几何精修
            b = refine_bbox_v2(rgb, b, crop_padding=args.crop_padding)
            if b.w < args.min_size or b.h < args.min_size:
                continue
            if b.area > args.max_area_frac * W * H:
                continue
            final_boxes_pre_split.append(b)

        # 内部间隙分割
        final_boxes: List[Box] = []
        for b in final_boxes_pre_split:
            gaps = detect_internal_gaps(rgb, b)
            sub_boxes = split_by_gaps(b, gaps)
            final_boxes.extend(sub_boxes)

        # ---- CNN 评分 ----
        for b in final_boxes:
            if patch_model is not None:
                p, quality = cnn_score_v2(patch_model, rgb, b, device=device)
                b.score = 0.70 * p + 0.30 * b.score
                b.source = f"cnn+{b.source}"
            if b.score < args.threshold:
                continue

        final_boxes = [b for b in final_boxes if b.score >= args.threshold]
        final_boxes = nms(final_boxes, iou_thr=0.25)
        final_boxes = sorted(final_boxes, key=lambda b: (b.y1, b.x1))

        # ---- 置信度评分 ----
        for b in final_boxes:
            cs = compute_confidence(b, rgb, b.engine_votes,
                                    cnn_prob=b.score if patch_model else 0.0,
                                    has_text_nearby=("text" in b.source))
            b.confidence = cs.total
            b.confidence_tier = cs.tier
            tier_counts[cs.tier] = tier_counts.get(cs.tier, 0) + 1

            # 写置信度日志
            conf_row = {
                "page": page_no,
                "box": b.to_xywh(),
                "score": b.score,
                "source": b.source,
                "engine_votes": b.engine_votes,
                **cs.to_dict(),
            }
            with conf_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(conf_row, ensure_ascii=False) + "\n")

        # ---- 可视化 ----
        annotated = draw_boxes_v2(rgb, final_boxes, page_no)
        ann_path = out_dir / "annotated" / f"page_{page_no:04d}.jpg"
        cv2.imwrite(str(ann_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if args.show:
            try:
                cv2.imshow("EconChartExtractor V2", annotated)
                cv2.waitKey(args.delay)
            except cv2.error:
                pass

        # ---- 保存裁切 ----
        for idx_page, b in enumerate(final_boxes, 1):
            global_idx += 1
            crop = rgb[b.y1:b.y2, b.x1:b.x2]
            fname = (f"chart_{global_idx:06d}_p{page_no:04d}_{idx_page:02d}"
                     f"_x{b.x1}_y{b.y1}_w{b.w}_h{b.h}.png")
            crop_path = out_dir / "crops" / fname
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(crop).save(crop_path, dpi=(args.dpi, args.dpi))

            row = {
                "id": global_idx, "pdf": str(pdf_path), "page": page_no,
                "page_width_px": W, "page_height_px": H, "dpi": args.dpi,
                "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                "width": b.w, "height": b.h,
                "score": round(float(b.score), 5), "source": b.source,
                "engine_votes": b.engine_votes,
                "confidence": round(b.confidence, 1),
                "confidence_tier": b.confidence_tier,
                "crop_file": str(crop_path), "annotated_page": str(ann_path),
            }
            with meta_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if not meta_csv.exists():
                with meta_csv.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writeheader()
            with meta_csv.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)

    if args.show:
        try:
            print("处理完毕。按任意键关闭窗口。")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    print(f"完成：共处理 {total_pages} 页，保存 {global_idx} 个图表。输出目录：{out_dir}")
    print(f"置信度分布: trusted={tier_counts['trusted']}, "
          f"uncertain={tier_counts['uncertain']}, unreliable={tier_counts['unreliable']}")


# ========================================================================
# CLI
# ========================================================================

def build_parser():
    p = argparse.ArgumentParser(description="EconChartExtractor V2 — 多引擎共识图表提取")

    def add_page_args(sp):
        sp.add_argument("pdf", help="输入 PDF 文件")
        sp.add_argument("--dpi", type=int, default=360)
        sp.add_argument("--first", type=int, default=1)
        sp.add_argument("--last", type=int, default=None)
        sp.add_argument("--min-size", type=int, default=42)
        sp.add_argument("--crop-padding", type=int, default=12)

    ex = sub = p.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="多引擎提取图表")
    add_page_args(ex)
    ex.add_argument("--out-dir", required=True)
    ex.add_argument("--model", default=None, help="CNN 模型路径 (ResNet-Lite 或原版)")
    ex.add_argument("--threshold", type=float, default=0.60,
                    help="保留阈值；多引擎时可用较低阈值，置信度分流会二次筛选")
    ex.add_argument("--max-area-frac", type=float, default=0.36)
    ex.add_argument("--show", action="store_true")
    ex.add_argument("--delay", type=int, default=1)
    ex.add_argument("--append", action="store_true")
    ex.add_argument("--cpu", action="store_true")
    ex.add_argument("--yolo-model", default=None)
    ex.add_argument("--yolo-conf", type=float, default=0.25)
    ex.add_argument("--no-color-engine", action="store_true",
                    help="禁用颜色/图像区域引擎 (L2)")
    ex.set_defaults(func=extract_charts_v2)

    tr = sub.add_parser("train", help="训练 ResNet-Lite 图表分类器")
    tr.add_argument("--label-dir", required=True)
    tr.add_argument("--model-out", required=True)
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--batch-size", type=int, default=32)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--use-small-cnn", action="store_true",
                    help="使用原版小 CNN 而非 ResNet-Lite")
    tr.set_defaults(func=lambda a: train_model_v2(
        Path(a.label_dir), Path(a.model_out), a.epochs, a.batch_size, a.lr,
        use_lite=not a.use_small_cnn))

    return p


def main(argv: Optional[Sequence[str]] = None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
