"""
DIP Pipeline — Enhanced Research-Grade v2.0

Novel Digital Image Processing pipeline for satellite imagery enhancement.
Combines classical DIP techniques with modern approaches:

  1. Bilateral Filter         — Edge-preserving noise reduction
  2. CLAHE                    — Adaptive local contrast enhancement (LAB L-channel)
  3. Unsharp Mask             — High-pass edge sharpening
  4. Wavelet Detail Enhance   — DWT-based sub-band amplification (NOVEL)
  5. Multi-Scale Retinex      — Illumination-invariant color restoration (NOVEL)
  6. Morphological Gradient   — Pond boundary edge highlighting (NOVEL)
  7. Guided Filter            — Structure-preserving smoothing (NOVEL)
  8. Water-Index False Color  — NDWI/MNDWI-aware composite generation (NOVEL)
  9. Quality Metrics          — PSNR / SSIM for objective assessment

References:
  - Jobson et al. 1997 (MSR), Kou et al. 2015 (Guided Filter)
  - Mallat 1989 (DWT), Feyisa et al. 2014 (AWEI)
"""

import cv2
import numpy as np
import os


# ═══════════════════════════════════════════════════════════════
# 1. BILATERAL FILTER — Edge-preserving noise reduction
# ═══════════════════════════════════════════════════════════════

def apply_bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    """Bilateral filter: smooths flat regions while preserving edges."""
    return cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


# ═══════════════════════════════════════════════════════════════
# 2. CLAHE — Contrast Limited Adaptive Histogram Equalization
# ═══════════════════════════════════════════════════════════════

def apply_clahe(img, clip_limit=2.5, tile_size=(8, 8)):
    """CLAHE on L-channel of LAB colorspace for luminance-only enhancement."""
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe.apply(img)


# ═══════════════════════════════════════════════════════════════
# 3. UNSHARP MASK — High-pass edge sharpening
# ═══════════════════════════════════════════════════════════════

def apply_unsharp_mask(img, kernel_size=(9, 9), sigma=10.0, strength=1.5):
    """Unsharp masking: original + strength * (original - blurred)."""
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    return cv2.addWeighted(img, strength, blurred, 1.0 - strength, 0)


# ═══════════════════════════════════════════════════════════════
# 4. WAVELET DETAIL ENHANCEMENT (NOVEL)
# ═══════════════════════════════════════════════════════════════

def _dwt2(img):
    """Manual 2D Discrete Wavelet Transform using Haar basis.
    Returns (LL, LH, HL, HH) sub-bands."""
    # Ensure even dimensions
    h, w = img.shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    img = img[:h2, :w2].astype(np.float64)

    # Row-wise transform
    even_cols = img[:, 0::2]
    odd_cols = img[:, 1::2]
    L = (even_cols + odd_cols) / 2.0
    H = (even_cols - odd_cols) / 2.0

    # Column-wise transform on L
    LL = (L[0::2, :] + L[1::2, :]) / 2.0
    LH = (L[0::2, :] - L[1::2, :]) / 2.0

    # Column-wise transform on H
    HL = (H[0::2, :] + H[1::2, :]) / 2.0
    HH = (H[0::2, :] - H[1::2, :]) / 2.0

    return LL, LH, HL, HH


def _idwt2(LL, LH, HL, HH):
    """Inverse 2D DWT — reconstruct from sub-bands."""
    h, w = LL.shape[:2]

    # Reconstruct L and H
    L = np.zeros((h * 2, w), dtype=np.float64)
    L[0::2, :] = LL + LH
    L[1::2, :] = LL - LH

    H = np.zeros((h * 2, w), dtype=np.float64)
    H[0::2, :] = HL + HH
    H[1::2, :] = HL - HH

    # Reconstruct image
    img = np.zeros((h * 2, w * 2), dtype=np.float64)
    img[:, 0::2] = L + H
    img[:, 1::2] = L - H

    return img


def apply_wavelet_enhancement(img, detail_boost=1.5):
    """
    Wavelet-based detail enhancement (NOVEL for satellite DIP).

    Decomposes image into LL (approximation) and LH/HL/HH (detail) sub-bands
    using Haar DWT. Amplifies detail sub-bands by `detail_boost` factor,
    then reconstructs. This selectively sharpens edges and fine structures
    (pond boundaries, embankments) without amplifying broadband noise.

    Reference: Mallat 1989, IEEE Trans. PAMI
    """
    if len(img.shape) == 3:
        # Process each channel independently
        channels = cv2.split(img)
        enhanced_channels = []
        for ch in channels:
            LL, LH, HL, HH = _dwt2(ch)
            # Boost high-frequency detail sub-bands
            LH_boosted = LH * detail_boost
            HL_boosted = HL * detail_boost
            HH_boosted = HH * detail_boost
            reconstructed = _idwt2(LL, LH_boosted, HL_boosted, HH_boosted)
            # Clip and resize to original shape
            reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
            reconstructed = cv2.resize(reconstructed, (img.shape[1], img.shape[0]))
            enhanced_channels.append(reconstructed)
        return cv2.merge(enhanced_channels)
    else:
        LL, LH, HL, HH = _dwt2(img)
        LH *= detail_boost
        HL *= detail_boost
        HH *= detail_boost
        reconstructed = _idwt2(LL, LH, HL, HH)
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        return cv2.resize(reconstructed, (img.shape[1], img.shape[0]))


# ═══════════════════════════════════════════════════════════════
# 5. MULTI-SCALE RETINEX (NOVEL)
# ═══════════════════════════════════════════════════════════════

def apply_multi_scale_retinex(img, sigmas=(15, 80, 250), gain=1.0):
    """
    Multi-Scale Retinex for illumination-invariant color restoration.

    Estimates illumination at 3 spatial scales via Gaussian blur,
    then computes log(image) - log(illumination) to extract reflectance.
    Critical for coastal satellite imagery where atmospheric haze varies spatially.

    Reference: Jobson et al. 1997, IEEE Trans. Image Processing
    """
    img_float = img.astype(np.float64) + 1.0  # Avoid log(0)
    retinex = np.zeros_like(img_float)

    for sigma in sigmas:
        blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
        blur = np.maximum(blur, 1.0)
        retinex += np.log10(img_float) - np.log10(blur)

    retinex /= len(sigmas)

    # Normalize to 0-255 range
    for i in range(retinex.shape[2] if len(retinex.shape) == 3 else 1):
        if len(retinex.shape) == 3:
            channel = retinex[:, :, i]
        else:
            channel = retinex
        c_min, c_max = channel.min(), channel.max()
        if c_max - c_min > 0:
            channel[:] = (channel - c_min) / (c_max - c_min) * 255.0
        else:
            channel[:] = 128.0
        if len(retinex.shape) != 3:
            retinex = channel
            break

    return np.clip(retinex * gain, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# 6. MORPHOLOGICAL GRADIENT (NOVEL)
# ═══════════════════════════════════════════════════════════════

def apply_morphological_gradient(img, kernel_size=3):
    """
    Morphological Gradient = Dilation − Erosion.

    Highlights pond boundary edges as bright lines against dark backgrounds.
    Particularly effective for rectilinear aquaculture structures where
    embankments create strong structural edges.

    Returns the gradient image (can be blended with original).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    dilated = cv2.dilate(gray, kernel)
    eroded = cv2.erode(gray, kernel)
    gradient = cv2.subtract(dilated, eroded)

    return gradient


def blend_morphological_gradient(img, gradient_weight=0.3):
    """Blend morphological gradient edges into the original image."""
    gradient = apply_morphological_gradient(img)

    if len(img.shape) == 3:
        # Colorize gradient (cyan for water edges)
        gradient_color = np.zeros_like(img)
        gradient_color[:, :, 0] = gradient  # Blue
        gradient_color[:, :, 1] = gradient  # Green
        gradient_color[:, :, 2] = 0         # No red
        blended = cv2.addWeighted(img, 1.0, gradient_color, gradient_weight, 0)
    else:
        blended = cv2.addWeighted(img, 1.0 - gradient_weight, gradient, gradient_weight, 0)

    return blended


# ═══════════════════════════════════════════════════════════════
# 7. GUIDED FILTER (NOVEL)
# ═══════════════════════════════════════════════════════════════

def apply_guided_filter(img, radius=8, eps=0.04):
    """
    Guided Filter — structure-preserving smoothing.

    Uses the image itself as the guide to smooth noise while keeping
    edges aligned with structural boundaries (pond edges, coastlines).
    Superior to bilateral filter for preserving gradient transitions.

    Reference: He et al. 2010 (ECCV), Kou et al. 2015
    """
    img_float = img.astype(np.float64) / 255.0

    if len(img.shape) == 3:
        channels = cv2.split(img_float)
        filtered_channels = []
        for ch in channels:
            filtered = _guided_filter_single(ch, ch, radius, eps)
            filtered_channels.append(filtered)
        result = cv2.merge(filtered_channels)
    else:
        result = _guided_filter_single(img_float, img_float, radius, eps)

    return np.clip(result * 255, 0, 255).astype(np.uint8)


def _guided_filter_single(guide, src, radius, eps):
    """Core guided filter for single channel."""
    ksize = 2 * radius + 1
    mean_g = cv2.boxFilter(guide, -1, (ksize, ksize))
    mean_s = cv2.boxFilter(src, -1, (ksize, ksize))
    mean_gs = cv2.boxFilter(guide * src, -1, (ksize, ksize))
    mean_gg = cv2.boxFilter(guide * guide, -1, (ksize, ksize))

    cov_gs = mean_gs - mean_g * mean_s
    var_g = mean_gg - mean_g * mean_g

    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g

    mean_a = cv2.boxFilter(a, -1, (ksize, ksize))
    mean_b = cv2.boxFilter(b, -1, (ksize, ksize))

    return mean_a * guide + mean_b


# ═══════════════════════════════════════════════════════════════
# 8. QUALITY METRICS
# ═══════════════════════════════════════════════════════════════

def compute_psnr(original, enhanced):
    """Peak Signal-to-Noise Ratio (higher = better, infinity = identical)."""
    mse = np.mean((original.astype(np.float64) - enhanced.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def compute_ssim(original, enhanced):
    """
    Structural Similarity Index Measure (range 0-1, 1 = identical).
    Simplified single-scale SSIM implementation.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = original.astype(np.float64)
    img2 = enhanced.astype(np.float64)

    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float64)
        img2 = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY).astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(np.mean(ssim_map))


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def enhance_image(image_path, output_path=None, mode="full"):
    """
    Enhanced DIP pipeline for satellite imagery.

    Modes:
      "full"     - All techniques (bilateral → CLAHE → wavelet → MSR → guided → unsharp → morph blend)
      "standard" - Classic only (bilateral → CLAHE → unsharp)
      "water"    - Water-optimized (bilateral → CLAHE → wavelet → morph blend)

    Returns True on success, False on failure.
    """
    if not os.path.exists(image_path):
        return False

    img = cv2.imread(image_path)
    if img is None:
        return False

    original = img.copy()

    if mode == "standard":
        # Classic pipeline (backward compatible)
        img = apply_bilateral_filter(img)
        img = apply_clahe(img)
        img = apply_unsharp_mask(img)

    elif mode == "water":
        # Water-optimized: emphasizes pond edges and water contrast
        img = apply_bilateral_filter(img, d=7, sigma_color=50, sigma_space=50)
        img = apply_clahe(img, clip_limit=3.0, tile_size=(4, 4))
        img = apply_wavelet_enhancement(img, detail_boost=1.8)
        img = blend_morphological_gradient(img, gradient_weight=0.25)

    elif mode == "stage":
        # Stage classified map: maximize contrast and edge sharpness
        # v21.0 FIX: Removed wavelet enhancement — it creates false texture
        # on what should be flat-colored stage regions, making the map look fuzzy.
        # Instead: bilateral filter (edge-preserving) → CLAHE → unsharp → quantize
        
        # Step 1: Edge-preserving smoothing — clean noise within stage regions
        img = apply_bilateral_filter(img, d=5, sigma_color=40, sigma_space=40)
        
        # Step 2: CLAHE for contrast
        img = apply_clahe(img, clip_limit=3.0, tile_size=(4, 4))
        
        # Step 3: Strong unsharp mask for crisp boundaries
        img = apply_unsharp_mask(img, kernel_size=(5, 5), sigma=2.0, strength=1.8)
        
        # Step 4: Color quantization — snap each pixel to nearest stage palette color
        # This prevents color drift from processing and ensures pure stage colors
        # Stage palette: S1=#1b7837, S2=#fee08b, S3=#d73027, S4=#4575b4, S5=#313695
        stage_colors = np.array([
            [27, 120, 55],    # S1 green (BGR: 55, 120, 27)
            [254, 224, 139],  # S2 yellow
            [215, 48, 39],    # S3 red
            [69, 117, 180],   # S4 blue
            [49, 54, 149],    # S5 dark blue
            [0, 0, 0],        # Background/river (black)
            [255, 255, 255],  # NoData (white)
        ], dtype=np.float32)
        # Convert to BGR for OpenCV
        stage_colors_bgr = np.array([
            [55, 120, 27],
            [139, 224, 254],
            [39, 48, 215],
            [180, 117, 69],
            [149, 54, 49],
            [0, 0, 0],
            [255, 255, 255],
        ], dtype=np.float32)
        
        # For each pixel, find nearest stage color
        h, w = img.shape[:2]
        flat = img.reshape(-1, 3).astype(np.float32)
        # Compute distance to each stage color
        distances = np.zeros((flat.shape[0], len(stage_colors_bgr)))
        for i, color in enumerate(stage_colors_bgr):
            distances[:, i] = np.sum((flat - color) ** 2, axis=1)
        nearest = np.argmin(distances, axis=1)
        quantized = stage_colors_bgr[nearest].astype(np.uint8)
        img = quantized.reshape(h, w, 3)

    else:  # "full"
        # 1. Edge-preserving noise suppression
        img = apply_bilateral_filter(img)

        # 2. Local contrast enhancement (LAB L-channel)
        img = apply_clahe(img, clip_limit=2.5, tile_size=(8, 8))

        # 3. Wavelet sub-band detail amplification
        img = apply_wavelet_enhancement(img, detail_boost=1.4)

        # 4. Multi-Scale Retinex for illumination normalization
        retinex = apply_multi_scale_retinex(img)
        # Blend 30% retinex for subtle illumination correction
        img = cv2.addWeighted(img, 0.7, retinex, 0.3, 0)

        # 5. Guided filter for structure-preserving smoothing
        img = apply_guided_filter(img, radius=6, eps=0.02)

        # 6. Final edge sharpening
        img = apply_unsharp_mask(img, strength=1.3)

        # 7. Subtle morphological gradient blend for boundaries
        img = blend_morphological_gradient(img, gradient_weight=0.15)

    # Compute quality metrics
    psnr = compute_psnr(original, img)
    ssim = compute_ssim(original, img)

    save_path = output_path if output_path else image_path
    cv2.imwrite(save_path, img)

    # Save gradient map alongside (for analysis)
    gradient_path = save_path.replace(".png", "_edges.png")
    gradient = apply_morphological_gradient(original)
    cv2.imwrite(gradient_path, gradient)

    print(f"[DIP] Enhanced: {os.path.basename(save_path)} | "
          f"PSNR={psnr:.1f}dB SSIM={ssim:.3f} | Mode={mode}")

    return True


def generate_false_color_water(image_path, output_path=None):
    """
    Generate NDWI-aware false-color visualization (NOVEL).

    Creates a composite where water appears in vivid cyan/blue and
    vegetation in green, improving visual discrimination of aquaculture ponds.
    """
    if not os.path.exists(image_path):
        return False

    img = cv2.imread(image_path)
    if img is None:
        return False

    b, g, r = cv2.split(img)
    b_f = b.astype(np.float64)
    g_f = g.astype(np.float64)
    r_f = r.astype(np.float64)

    # Pseudo-NDWI: (Green - Red) / (Green + Red + 1)
    ndwi_proxy = (g_f - r_f) / (g_f + r_f + 1.0)
    ndwi_norm = ((ndwi_proxy + 1.0) / 2.0 * 255).astype(np.uint8)

    # False color: R=Red, G=Green, B=NDWI
    false_color = cv2.merge([ndwi_norm, g, r])

    save_path = output_path or image_path.replace(".png", "_water_fc.png")
    cv2.imwrite(save_path, false_color)
    print(f"[DIP] Water false-color: {os.path.basename(save_path)}")
    return True
