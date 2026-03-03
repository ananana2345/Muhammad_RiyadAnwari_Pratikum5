# ==========================================================
# Evaluasi Spatial Filtering - FINAL VERSION (COMPLETE)
# ==========================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from skimage.metrics import structural_similarity as ssim
import os

# ==========================================================
# CEK FILE
# ==========================================================
if not os.path.exists("image1.jpeg"):
    print("File image1.jpeg tidak ditemukan!")
    exit()

# ==========================================================
# LOAD IMAGE
# ==========================================================
original = cv2.imread("image1.jpeg")
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

# ==========================================================
# METRIK DASAR
# ==========================================================
def mse(img1, img2):
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

def psnr(img1, img2):
    m = mse(img1, img2)
    if m == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(m))

def compute_ssim(img1, img2):
    return ssim(img1, img2, channel_axis=2)

# ==========================================================
# METRIK VISUAL INSPECTION TAMBAHAN
# ==========================================================
def compute_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def compute_edge_preservation(original, filtered):
    edge_orig = cv2.Canny(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), 100, 200)
    edge_filt = cv2.Canny(cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY), 100, 200)
    return np.mean(np.abs(edge_orig - edge_filt))

# ==========================================================
# TAMBAH 3 JENIS NOISE
# ==========================================================
def add_noise(img):

    # Gaussian Noise
    gaussian = np.random.normal(0, 80, img.shape)
    gaussian_noise = np.clip(img + gaussian, 0, 255).astype(np.uint8)

    # Salt & Pepper Noise
    salt_pepper = img.copy()
    prob = 0.1
    num_pixels = int(prob * img.shape[0] * img.shape[1])

    for _ in range(num_pixels):
        y = np.random.randint(0, img.shape[0])
        x = np.random.randint(0, img.shape[1])
        salt_pepper[y, x] = [255, 255, 255]

    for _ in range(num_pixels):
        y = np.random.randint(0, img.shape[0])
        x = np.random.randint(0, img.shape[1])
        salt_pepper[y, x] = [0, 0, 0]

    # Speckle Noise
    noise = np.random.randn(*img.shape)
    speckle_noise = np.clip(img + img * noise * 0.7, 0, 255).astype(np.uint8)

    return gaussian_noise, salt_pepper, speckle_noise

# ==========================================================
# FILTER SPASIAL
# ==========================================================
def apply_filters(img):

    results = {}
    kernel = np.ones((5,5), np.uint8)

    filters = {

        # Linear Filters
        "Mean 3x3": lambda x: cv2.blur(x, (3,3)),
        "Mean 7x7": lambda x: cv2.blur(x, (7,7)),
        "Gaussian sigma1": lambda x: cv2.GaussianBlur(x, (7,7), 1),
        "Gaussian sigma3": lambda x: cv2.GaussianBlur(x, (9,9), 3),

        # Non-linear Filters
        "Median 3x3": lambda x: cv2.medianBlur(x, 3),
        "Median 7x7": lambda x: cv2.medianBlur(x, 7),

        # Min/Max Filter (pilih salah satu → di sini Max)
        "Max Filter": lambda x: cv2.dilate(x, kernel)
    }

    for name, func in filters.items():
        start = time.time()
        filtered = func(img)
        end = time.time()
        results[name] = (filtered, end - start)

    return results

# ==========================================================
# MAIN PROGRAM
# ==========================================================
def main():

    gaussian_noise, salt_pepper, speckle_noise = add_noise(original)

    noisy_images = {
        "Gaussian": gaussian_noise,
        "SaltPepper": salt_pepper,
        "Speckle": speckle_noise
    }

    records = []

    for noise_name, noisy_img in noisy_images.items():

        filtered_results = apply_filters(noisy_img)

        # ================= HITUNG METRIK =================
        for filter_name, (filtered_img, proc_time) in filtered_results.items():

            m = mse(original, filtered_img)
            p = psnr(original, filtered_img)
            s = compute_ssim(original, filtered_img)
            sharp = compute_sharpness(filtered_img)
            edge_diff = compute_edge_preservation(original, filtered_img)

            records.append([
                noise_name,
                filter_name,
                round(m, 2),
                round(p, 2),
                round(s, 4),
                round(sharp, 2),
                round(edge_diff, 2),
                round(proc_time, 6)
            ])

        # ================= VISUAL INSPECTION =================
        plt.figure(figsize=(16, 10))

        plt.subplot(3,4,1)
        plt.title("Original")
        plt.imshow(original)
        plt.axis('off')

        plt.subplot(3,4,2)
        plt.title(f"{noise_name} Noise")
        plt.imshow(noisy_img)
        plt.axis('off')

        i = 3
        for name, (img, _) in filtered_results.items():

            sharp = compute_sharpness(img)
            edge_diff = compute_edge_preservation(original, img)

            plt.subplot(3,4,i)
            plt.title(
                f"{name}\nSharp:{sharp:.1f} | EdgeDiff:{edge_diff:.1f}",
                fontsize=8
            )
            plt.imshow(img)
            plt.axis('off')

            i += 1

        plt.suptitle(f"Visual Inspection - {noise_name} Noise", fontsize=14)
        plt.tight_layout()
        plt.show()

    # ======================================================
    # TABEL PERBANDINGAN METRIK
    # ======================================================
    df = pd.DataFrame(records, columns=[
        "Noise",
        "Filter",
        "MSE",
        "PSNR",
        "SSIM",
        "Sharpness",
        "EdgeDiff",
        "Time (s)"
    ])

    df_sorted = df.sort_values(by=["Noise", "PSNR"], ascending=[True, False])

    print("\n===== TABEL PERBANDINGAN METRIK =====\n")
    print(df_sorted.to_string(index=False))

    # Visual Table
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')

    table = ax.table(
        cellText=df_sorted.values,
        colLabels=df_sorted.columns,
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(df_sorted.columns))))

    plt.title("Tabel Perbandingan Metrik Evaluasi Filter", fontsize=12)
    plt.tight_layout()
    plt.show()

    # Simpan CSV
    df_sorted.to_csv("hasil_perbandingan_metrik.csv", index=False)
    print("\nFile 'hasil_perbandingan_metrik.csv' berhasil disimpan.")

if __name__ == "__main__":
    main()