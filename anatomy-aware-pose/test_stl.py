"""Test della STL: verifica differenziabilita' e sanity check.

Lancia con:  python test_stl.py
Deve stampare tutti OK. Se un gradcheck fallisce, il termine non e'
correttamente differenziabile e non puo' essere usato in training.
"""

import torch
from torch.autograd import gradcheck
from stl import (soft_argmax, bone_ratio_loss, joint_angle_loss,
                 geometric_ordering_loss, SkeletalTopologyLoss)
from train import WeightedMSELoss


def test_soft_argmax():
    """Verifica che soft-argmax produca coordinate sensate e sia differenziabile."""
    print("=== Test soft-argmax ===")

    # Heatmap sintetica: un picco gaussiano in (20, 30) su griglia 64x48
    B, K, H, W = 2, 17, 64, 48
    hm = torch.zeros(B, K, H, W, dtype=torch.float64, requires_grad=True)

    # Piazza picchi in posizioni note per il primo sample
    # keypoint 0 in (col=20, row=30), keypoint 1 in (col=10, row=40)
    with torch.no_grad():
        for b in range(B):
            for k in range(K):
                cx, cy = 20.0 + k, 30.0 - k * 0.5
                cx = min(max(cx, 1), W - 2)
                cy = min(max(cy, 1), H - 2)
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        y, x = int(cy) + dy, int(cx) + dx
                        if 0 <= y < H and 0 <= x < W:
                            hm.data[b, k, y, x] = torch.exp(
                                torch.tensor(-((x - cx)**2 + (y - cy)**2) / 8.0)
                            )

    coords = soft_argmax(hm, beta=10.0)
    print(f"  Forma output: {coords.shape} (atteso: [2, 17, 2])")

    # Verifica che le coordinate siano nel range della heatmap
    assert coords.shape == (B, K, 2)
    assert (coords[:, :, 0] >= 0).all() and (coords[:, :, 0] < W).all(), "x fuori range"
    assert (coords[:, :, 1] >= 0).all() and (coords[:, :, 1] < H).all(), "y fuori range"
    print(f"  Coordinate keypoint 0, sample 0: x={coords[0,0,0]:.2f}, y={coords[0,0,1]:.2f}")
    print(f"  (atteso circa x=20.0, y=30.0)")

    # Gradcheck: verifica numerica che il gradiente analitico sia corretto
    # Usiamo un input piccolo per velocita'
    hm_small = torch.randn(1, 3, 8, 6, dtype=torch.float64, requires_grad=True)
    func = lambda h: soft_argmax(h, beta=5.0).sum()
    ok = gradcheck(func, (hm_small,), eps=1e-4, atol=1e-3)
    print(f"  Gradcheck soft-argmax: {'OK' if ok else 'FALLITO'}")
    return ok


def test_bone_ratio():
    """Verifica bone_ratio_loss: zero su pose perfette, positiva su pose rotte."""
    print("\n=== Test bone ratio loss ===")

    B, K = 2, 17
    # Posa "perfetta": proporzioni da manuale (Winter 2009)
    coords = torch.zeros(B, K, 2, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        # Costruisci uno scheletro con proporzioni corrette
        # Spalle a y=10, gomiti a y=10+18.6, polsi a y=10+18.6+14.6
        # Anche a y=30, ginocchia a y=30+24.5, caviglie a y=30+24.5+24.6
        for b in range(B):
            coords.data[b, 5]  = torch.tensor([20.0, 10.0])   # left_shoulder
            coords.data[b, 6]  = torch.tensor([30.0, 10.0])   # right_shoulder
            coords.data[b, 7]  = torch.tensor([18.0, 28.6])   # left_elbow
            coords.data[b, 8]  = torch.tensor([32.0, 28.6])   # right_elbow
            coords.data[b, 9]  = torch.tensor([16.0, 43.2])   # left_wrist
            coords.data[b, 10] = torch.tensor([34.0, 43.2])   # right_wrist
            coords.data[b, 11] = torch.tensor([22.0, 30.0])   # left_hip
            coords.data[b, 12] = torch.tensor([28.0, 30.0])   # right_hip
            coords.data[b, 13] = torch.tensor([22.0, 54.5])   # left_knee
            coords.data[b, 14] = torch.tensor([28.0, 54.5])   # right_knee
            coords.data[b, 15] = torch.tensor([22.0, 79.1])   # left_ankle
            coords.data[b, 16] = torch.tensor([28.0, 79.1])   # right_ankle

    loss_good = bone_ratio_loss(coords)
    print(f"  Loss su posa corretta: {loss_good.item():.6f} (atteso: ~0)")

    # Posa rotta: braccio sinistro 3x piu' lungo del destro
    coords_bad = coords.detach().clone().requires_grad_(True)
    with torch.no_grad():
        coords_bad.data[:, 9] = torch.tensor([10.0, 80.0])  # polso sx lontanissimo

    loss_bad = bone_ratio_loss(coords_bad)
    print(f"  Loss su posa rotta:   {loss_bad.item():.6f} (atteso: >> 0)")
    assert loss_bad > loss_good, "La posa rotta dovrebbe avere loss piu' alta!"

    # Gradcheck
    coords_gc = torch.randn(1, 17, 2, dtype=torch.float64, requires_grad=True) * 10
    ok = gradcheck(bone_ratio_loss, (coords_gc,), eps=1e-4, atol=1e-3)
    print(f"  Gradcheck bone_ratio_loss: {'OK' if ok else 'FALLITO'}")
    return ok


def test_joint_angle():
    """Verifica joint_angle_loss."""
    print("\n=== Test joint angle loss ===")

    B, K = 1, 17
    coords = torch.zeros(B, K, 2, dtype=torch.float64, requires_grad=True)

    # Gomito a ~90 gradi (dentro il range [10, 180])
    with torch.no_grad():
        coords.data[0, 5]  = torch.tensor([0.0, 0.0])   # shoulder
        coords.data[0, 7]  = torch.tensor([10.0, 0.0])   # elbow
        coords.data[0, 9]  = torch.tensor([10.0, 10.0])  # wrist -> angolo 90°
        # Riempi anche il lato destro e le gambe per avere tutte le regole
        coords.data[0, 6]  = torch.tensor([0.0, 0.0])
        coords.data[0, 8]  = torch.tensor([-10.0, 0.0])
        coords.data[0, 10] = torch.tensor([-10.0, 10.0])
        coords.data[0, 11] = torch.tensor([5.0, 15.0])
        coords.data[0, 12] = torch.tensor([-5.0, 15.0])
        coords.data[0, 13] = torch.tensor([5.0, 30.0])
        coords.data[0, 14] = torch.tensor([-5.0, 30.0])
        coords.data[0, 15] = torch.tensor([5.0, 45.0])
        coords.data[0, 16] = torch.tensor([-5.0, 45.0])

    loss_normal = joint_angle_loss(coords)
    print(f"  Loss su angoli normali: {loss_normal.item():.6f} (atteso: ~0)")

    # Gomito collassato: 3 punti quasi sovrapposti -> angolo ~0
    coords_bad = coords.detach().clone().requires_grad_(True)
    with torch.no_grad():
        coords_bad.data[0, 5] = torch.tensor([10.0, 0.0])   # shoulder = elbow!
        coords_bad.data[0, 7] = torch.tensor([10.0, 0.0])   # elbow
        coords_bad.data[0, 9] = torch.tensor([10.1, 0.0])   # wrist appena spostato

    loss_bad = joint_angle_loss(coords_bad)
    print(f"  Loss su gomito collassato: {loss_bad.item():.6f} (atteso: >> 0)")
    assert loss_bad > loss_normal

    # Gradcheck
    coords_gc = torch.randn(1, 17, 2, dtype=torch.float64, requires_grad=True) * 10 + 5
    ok = gradcheck(joint_angle_loss, (coords_gc,), eps=1e-4, atol=1e-3)
    print(f"  Gradcheck joint_angle_loss: {'OK' if ok else 'FALLITO'}")
    return ok


def test_geometric_ordering():
    """Verifica geometric_ordering_loss."""
    print("\n=== Test geometric ordering loss ===")

    B, K = 1, 17
    coords = torch.zeros(B, K, 2, dtype=torch.float64, requires_grad=True)

    # Catena corretta: spalla(5) -> gomito(7) -> polso(9) in ordine
    with torch.no_grad():
        coords.data[0, 5]  = torch.tensor([0.0, 0.0])
        coords.data[0, 7]  = torch.tensor([5.0, 5.0])    # a meta'
        coords.data[0, 9]  = torch.tensor([10.0, 10.0])
        coords.data[0, 6]  = torch.tensor([20.0, 0.0])
        coords.data[0, 8]  = torch.tensor([25.0, 5.0])
        coords.data[0, 10] = torch.tensor([30.0, 10.0])
        coords.data[0, 11] = torch.tensor([5.0, 20.0])
        coords.data[0, 13] = torch.tensor([5.0, 30.0])
        coords.data[0, 15] = torch.tensor([5.0, 40.0])
        coords.data[0, 12] = torch.tensor([15.0, 20.0])
        coords.data[0, 14] = torch.tensor([15.0, 30.0])
        coords.data[0, 16] = torch.tensor([15.0, 40.0])

    loss_ok = geometric_ordering_loss(coords)
    print(f"  Loss catene ordinate: {loss_ok.item():.6f} (atteso: ~0)")

    # Ginocchio DOPO la caviglia -> violazione
    coords_bad = coords.detach().clone().requires_grad_(True)
    with torch.no_grad():
        coords_bad.data[0, 13] = torch.tensor([5.0, 50.0])  # ginocchio sotto caviglia!

    loss_bad = geometric_ordering_loss(coords_bad)
    print(f"  Loss catena rotta:    {loss_bad.item():.6f} (atteso: >> 0)")
    assert loss_bad > loss_ok

    # Gradcheck
    coords_gc = torch.randn(1, 17, 2, dtype=torch.float64, requires_grad=True) * 10
    ok = gradcheck(geometric_ordering_loss, (coords_gc,), eps=1e-4, atol=1e-3)
    print(f"  Gradcheck geometric_ordering_loss: {'OK' if ok else 'FALLITO'}")
    return ok


def test_combined():
    """Test end-to-end: heatmap -> soft-argmax -> STL."""
    print("\n=== Test loss combinata (end-to-end) ===")

    B, K, H, W = 2, 17, 64, 48
    pred = torch.randn(B, K, H, W, requires_grad=True)
    target = torch.randn(B, K, H, W)
    weight = torch.ones(B, K, 1)

    criterion = SkeletalTopologyLoss(
        heatmap_criterion=WeightedMSELoss(),
        lambda_bone=0.5, lambda_angle=0.5, lambda_order=0.5, beta=10.0,
    )

    loss, terms = criterion(pred, target, weight)
    print(f"  Loss totale: {terms['total']:.4f}")
    print(f"    heatmap: {terms['heatmap']:.4f}")
    print(f"    bone:    {terms['bone']:.4f}")
    print(f"    angle:   {terms['angle']:.4f}")
    print(f"    order:   {terms['order']:.4f}")

    # Verifica che il backward funzioni senza errori
    loss.backward()
    grad_norm = pred.grad.norm().item()
    print(f"  Backward OK, norma gradiente: {grad_norm:.4f}")
    assert grad_norm > 0, "I gradienti sono zero — qualcosa non torna"
    print("  End-to-end: OK")
    return True


if __name__ == '__main__':
    results = []
    results.append(('soft_argmax',         test_soft_argmax()))
    results.append(('bone_ratio_loss',     test_bone_ratio()))
    results.append(('joint_angle_loss',    test_joint_angle()))
    results.append(('geometric_ordering',  test_geometric_ordering()))
    results.append(('combined_e2e',        test_combined()))

    print("\n" + "=" * 50)
    print("RIEPILOGO:")
    all_ok = True
    for name, ok in results:
        status = "OK" if ok else "FALLITO"
        print(f"  {name:30s} {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nTutti i test passati. La STL e' differenziabile e pronta per il training.")
    else:
        print("\nATTENZIONE: alcuni test falliti! Controlla prima di usare in training.")
