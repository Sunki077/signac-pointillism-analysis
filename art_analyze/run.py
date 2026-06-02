# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
from skimage import feature
import os

# -----------------------------
# 输出文件夹
# -----------------------------
os.makedirs("figures", exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"

figure_id = 1
def save_fig(fig, name=None):
    global figure_id
    if name is None:
        name = f"Figure{figure_id}"
    fig.savefig(f"figures/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"figures/{name}.svg", bbox_inches="tight")
    figure_id += 1

# -----------------------------
# 读取图像
# -----------------------------
img = cv2.imread("signac_avignon.jpg")  # 替换为你的图片路径
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (800, 800))

# -----------------------------
# Figure 1 - 原始图像
# -----------------------------
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(img)
ax.axis("off")
ax.set_title("Original Artwork", fontsize=14)
save_fig(fig, "Original_Artwork")

# -----------------------------
# Figure 2 - LAB色彩聚类
# -----------------------------
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
pixels = lab.reshape((-1, 3))
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
kmeans.fit(pixels)
colors = kmeans.cluster_centers_
fig, ax = plt.subplots(figsize=(8,2))
ax.imshow([colors.astype(np.uint8)])
ax.axis("off")
ax.set_title("Dominant Color Clusters (LAB Space)", fontsize=14)
save_fig(fig, "Color_Clusters")

# -----------------------------
# Figure 3 - 点彩结构近似（边缘）
# -----------------------------
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
edges = feature.canny(gray, sigma=2)
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(edges, cmap="gray")
ax.axis("off")
ax.set_title("Edge Structure Approximation", fontsize=14)
save_fig(fig, "Edge_Structure")

# -----------------------------
# Figure 4 - 点密度热力图
# -----------------------------
density = gaussian_filter(edges.astype(float), sigma=15)
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(density, cmap="inferno")
ax.axis("off")
ax.set_title("Visual Density Distribution", fontsize=14)
save_fig(fig, "Density_Map")

# -----------------------------
# Figure 5 - 多尺度视觉混色（修正版，保留颜色）
# -----------------------------
blur_near = cv2.GaussianBlur(img, (0,0), sigmaX=2, sigmaY=2)
blur_mid  = cv2.GaussianBlur(img, (0,0), sigmaX=5, sigmaY=5)
blur_far  = cv2.GaussianBlur(img, (0,0), sigmaX=10, sigmaY=10)

fig, axes = plt.subplots(1,3, figsize=(12,4))
axes[0].imshow(blur_near)
axes[0].axis("off")
axes[0].set_title("Near View (Dots Visible)", fontsize=12)
axes[1].imshow(blur_mid)
axes[1].axis("off")
axes[1].set_title("Mid View (Partial Mixing)", fontsize=12)
axes[2].imshow(blur_far)
axes[2].axis("off")
axes[2].set_title("Far View (Optical Mixing)", fontsize=12)
plt.tight_layout()
save_fig(fig, "Optical_Mixing_Scale")

# -----------------------------
# Figure 6 - 潜在构图结构（Sobel梯度）
# -----------------------------
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
structure = np.sqrt(sobelx**2 + sobely**2)
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(structure, cmap="gray")
ax.axis("off")
ax.set_title("Latent Compositional Structure", fontsize=14)
save_fig(fig, "Composition_Structure")

