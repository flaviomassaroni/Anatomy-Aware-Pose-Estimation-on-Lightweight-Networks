"""Utils: generazione/decodifica heatmap, conversione coordinate, conteggio parametri.
"""
import numpy as np


def generate_heatmap(center_x, center_y, heatmap_h, heatmap_w, sigma):
    """Gaussiana 2D (heatmap_h x heatmap_w) centrata su (center_x, center_y)."""
    xs = np.arange(0, heatmap_w, 1, dtype=np.float32)
    ys = np.arange(0, heatmap_h, 1, dtype=np.float32)[:, np.newaxis]
    heatmap = np.exp(-((xs - center_x) ** 2 + (ys - center_y) ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)


def decode_heatmaps(heatmaps):
    """Da heatmap [B,K,H,W] a coordinate [B,K,2] in spazio heatmap + scores [B,K].
    argmax sul picco + raffinamento sub-pixel di 0.25 px nella direzione del gradiente.
    """
    B, K, H, W = heatmaps.shape
    hm = heatmaps.detach().cpu().numpy()
    hm_flat = hm.reshape(B, K, -1)
    idx = np.argmax(hm_flat, axis=2)
    scores = np.take_along_axis(hm_flat, idx[..., None], axis=2).squeeze(-1)

    coords = np.zeros((B, K, 2), dtype=np.float32)
    coords[..., 0] = idx % W
    coords[..., 1] = idx // W

    for b in range(B):
        for k in range(K):
            x, y = int(coords[b, k, 0]), int(coords[b, k, 1])
            if 1 < x < W - 1 and 1 < y < H - 1:
                dx = hm[b, k, y, x + 1] - hm[b, k, y, x - 1]
                dy = hm[b, k, y + 1, x] - hm[b, k, y - 1, x]
                coords[b, k, 0] += 0.25 * np.sign(dx)
                coords[b, k, 1] += 0.25 * np.sign(dy)
    return coords, scores


def heatmap_to_original(coords_hm, bbox, input_size, heatmap_size):
    """Riporta le coordinate da spazio heatmap a spazio immagine originale,
    invertendo la catena crop -> scala uniforme -> padding centrato.
    """
    input_h, input_w = input_size
    hm_h, hm_w = heatmap_size
    x, y, w, h = bbox

    scale = min(input_w / w, input_h / h)
    new_w, new_h = w * scale, h * scale
    pad_left = (input_w - new_w) / 2
    pad_top = (input_h - new_h) / 2

    coords_canvas = coords_hm.copy()
    coords_canvas[:, 0] *= input_w / hm_w
    coords_canvas[:, 1] *= input_h / hm_h

    coords_img = coords_canvas.copy()
    coords_img[:, 0] = (coords_canvas[:, 0] - pad_left) / scale + x
    coords_img[:, 1] = (coords_canvas[:, 1] - pad_top) / scale + y
    return coords_img


def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable
