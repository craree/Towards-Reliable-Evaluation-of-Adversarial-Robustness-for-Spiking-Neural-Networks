import random

import math
import time

import matplotlib.pyplot as plt
import numpy as np
from my_node import MyBaseNode
import torch
import torch.nn as nn
from typing import Union, Sequence
from torchattacks.attack import Attack
import torch.nn.functional as F
from braincog.base.node.node import *
import seaborn as sns

def G_dist(alpha_all, membrane_dis):
    alpha_1 = 2 * np.ones_like(alpha_all)
    alpha_2 = 10 * np.ones_like(alpha_all)
    alpha_3 = 18 * np.ones_like(alpha_all)
    Ga = np.asarray((2 / np.pi) * np.arctan((np.pi / 2) * alpha_all * np.abs(membrane_dis)))
    G1 = np.asarray((2 / np.pi) * np.arctan((np.pi / 2) * alpha_1 * np.abs(membrane_dis)))
    G2 = np.asarray((2 / np.pi) * np.arctan((np.pi / 2) * alpha_2 * np.abs(membrane_dis)))
    G3 = np.asarray((2 / np.pi) * np.arctan((np.pi / 2) * alpha_3 * np.abs(membrane_dis)))
    Ga_f = Ga.ravel()
    G1_f = G1.ravel()
    G2_f = G2.ravel()
    G3_f = G3.ravel()

    datasets = [G1_f, G2_f, G3_f, Ga_f]
    titles = [
        r"$G(x)$ with Fixed $\alpha=2$",
        r"$G(x)$ with Fixed $\alpha=10$",
        r"$G(x)$ with Fixed $\alpha=18$",
        r"$G(x)$ with ASSG A=0.87",
    ]

    # ===== 统一横轴和 bin（关键）=====
    all_data = np.concatenate(datasets)
    xmin, xmax = all_data.min(), all_data.max()
    bins = np.linspace(xmin, xmax, 500)

    def compute_stats_and_mode(data, bins):
        hist, edges = np.histogram(data, bins=bins, density=True)
        p = hist + 1e-12
        p = p / p.sum()

        entropy = -np.sum(p * np.log(p))
        variance = np.var(data)
        mean = np.mean(data)

        # ===== 计算分布峰值（mode）=====
        max_idx = np.argmax(hist)
        mode = 0.5 * (edges[max_idx] + edges[max_idx + 1])

        return entropy, variance, mean, mode

    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
    })

    fig, axes = plt.subplots(len(datasets), 1, figsize=(6, 12), sharex=True)

    for ax, data, title in zip(axes, datasets, titles):
        ax.hist(data, bins=bins, density=True)
        ax.set_title(title)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)

        entropy, var, mean, mode = compute_stats_and_mode(data, bins)

        # ===== 左上角文字 =====
        textstr = (
            f"Mean = {mean:.4f}\n"
            f"Var = {var:.3e}\n"
            f"Entropy = {entropy:.4f}\n"
            f"Mode = {mode:.4f}"
        )

        ax.text(
            0.02, 0.95, textstr,
            transform=ax.transAxes,
            ha='left', va='top',
            bbox=dict(boxstyle='round', alpha=0.08)
        )

        # ===== 峰值竖线 =====
        ax.axvline(mode, linestyle='--', linewidth=1.5, color='r')

    axes[-1].set_xlabel(r"$G(x)$")
    axes[0].set_xlim(xmin, xmax)

    plt.tight_layout(h_pad=1.0)
    plt.show()

def show_alpha(alpha_r, membrane_r, alpha_r2):
    alpha75 = alpha_r[2]  # [l t b c w h]
    n_layer = len(alpha75)

    membrane_r = torch.stack(membrane_r, dim=0).detach().cpu()
    alpha_r2 = torch.stack(alpha_r2, dim=0).detach().cpu()

    layer = n_layer - 2
    ids = [8, 25, 4]
    names = ["input A", "input B", "input C"]

    # unified color palette
    colors = {
        "input A": "#1F77B4",
        "input B": "#FF7F0E",
        "input C": "#2CA02C",
    }

    def beautify_axes(ax):
        ax.grid(True, alpha=0.25, linewidth=0.8)
        ax.tick_params(axis="both", labelsize=14)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # ================= FIG 1: membrane =================
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for idx_id, name in zip(ids, names):
        ax.plot(
            membrane_r[:, layer, idx_id].numpy(),
            label=name,
            color=colors[name],
            linewidth=2.4,
            alpha=0.95,
        )

    ax.set_xlabel("iteration steps", fontsize=18)
    ax.set_ylabel(r"$\left|u_{i,t}^{l}\right|$", fontsize=18, rotation=0, labelpad=35)
    ax.legend(fontsize=14)
    beautify_axes(ax)

    plt.tight_layout()
    plt.show()

    # ================= FIG 2: alpha =================
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for idx_id, name in zip(ids, names):
        ax.plot(
            alpha_r2[:, layer, idx_id].numpy(),
            color=colors[name],
            linewidth=2.4,
            alpha=0.95,
        )

    ax.set_xlabel("iteration steps", fontsize=18)
    ax.set_ylabel(r"$\alpha_{i,t}^{l}$", fontsize=18, rotation=0, labelpad=35)
    # ax.legend(fontsize=14)
    beautify_axes(ax)

    plt.tight_layout()
    plt.show()

    alpha_A = alpha75[n_layer - 3][:, 0]  # [t c w h]
    alpha_B = alpha75[n_layer - 3][:, 1]  # [t c w h]
    alpha_C = alpha75[n_layer - 3][:, 2]  # [t c w h]
    A = alpha_A.detach().cpu().numpy().ravel()
    B = alpha_B.detach().cpu().numpy().ravel()
    C = alpha_C.detach().cpu().numpy().ravel()

    n = 128
    idx = np.random.choice(A.shape[0], size=min(n, A.shape[0]), replace=False)
    A_s, B_s, C_s = A[idx], B[idx], C[idx]

    rab = np.corrcoef(A_s, B_s)[0, 1]
    rac = np.corrcoef(A_s, C_s)[0, 1]
    rbc = np.corrcoef(B_s, C_s)[0, 1]

    plt.figure(figsize=(7, 4))

    x = np.arange(len(A_s))

    # color-blind friendly colors
    color_A = "#1F77B4"  # deep blue
    color_B = "#FF7F0E"  # muted red
    color_C = "#2CA02C"  # green
    plt.scatter(x, A_s, label="input A", alpha=0.85, s=35, color=color_A, edgecolors="k", linewidths=0.3)
    plt.scatter(x, B_s, label="input B", alpha=0.85, s=35, color=color_B, edgecolors="k", linewidths=0.3)
    plt.scatter(x, C_s, label="input C", alpha=0.85, s=35, color=color_C, edgecolors="k", linewidths=0.3)

    plt.xlabel(r"$\alpha_{i,t}^{l}$", fontsize=18)
    plt.ylabel("Value", fontsize=18)
    plt.legend(fontsize=14)

    # ---- Pearson text block ----
    textstr = (
        f"Pearson r\n"
        f"A-B: {rab:.4f}\n"
        f"A-C: {rac:.4f}\n"
        f"B-C: {rbc:.4f}"
    )

    plt.text(
        0.02, 0.95, textstr,
        transform=plt.gca().transAxes,
        fontsize=15,
        va="top",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85)
    )

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for j in range(4):
        L = len(alpha_r[j])
        alpha_all = []
        for i in range(L):
            alpha_all.append(torch.cat([a.flatten() for a in alpha_r[j][i][:, 0]]))
        alpha_all = torch.cat(alpha_all).cpu().numpy()
        p99 = np.percentile(alpha_all, 99)
        mean_val = np.mean(alpha_all[alpha_all <= p99])  # 去除极端值后的均值
        p90_val = np.percentile(alpha_all, 90)  # 计算P90分位

        # 绘制子图
        ax = axes[j]
        sns.histplot(alpha_all, bins=300, color='steelblue', alpha=0.7,
                     stat='density', ax=ax, edgecolor=None)
        sns.kdeplot(alpha_all, color='darkred', linewidth=2, ax=ax)

        # 绘制均值与P90线
        ax.axvline(mean_val, color='orange', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.2f}')
        ax.axvline(p90_val, color='purple', linestyle=':', linewidth=2, label=f'P90 = {p90_val:.2f}')

        # 设置样式
        ax.set_xlim(0, p99)
        # ax.set_xlabel(r'$\alpha$ value', fontsize=15)
        ax.set_ylabel('Density', fontsize=15)
        ax.set_title(f'Iteration {int((j + 1) * 25)}', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=15, loc='upper right')

    plt.tight_layout()
    plt.show()