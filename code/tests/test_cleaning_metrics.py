"""Independent small-array oracles for polarity, pseudo recall and DRD borders."""
import numpy as np
import pytest

from cleaning.metrics import BinaryReference


def brute_drd(ground, pred):
    gt, prediction = ground == 255, pred == 255
    weights = {(dy, dx): 1 / np.hypot(dx, dy) for dy in range(-2, 3)
               for dx in range(-2, 3) if dx or dy}
    normalizer = sum(weights.values())
    blocks = 0
    for y in range(0, gt.shape[0], 8):
        for x in range(0, gt.shape[1], 8):
            values = gt[y:y+8, x:x+8]
            blocks += bool(values.any() and not values.all())
    distortion = 0
    for y, x in np.argwhere(gt != prediction):
        for (dy, dx), weight in weights.items():
            yy, xx = y+dy, x+dx
            if 0 <= yy < gt.shape[0] and 0 <= xx < gt.shape[1]:
                distortion += float(gt[yy, xx] != prediction[y, x]) * weight / normalizer
    return distortion / blocks


def test_known_confusion_counts_and_binary_psnr():
    gt = np.zeros((8, 8), np.uint8)
    gt[2:4, 2:4] = 255
    pred = gt.copy()
    pred[2, 2] = 0
    pred[6, 6] = 255
    score = BinaryReference(gt).score(pred)
    assert (score['tp'], score['fp'], score['fn']) == (3, 1, 1)
    assert score['precision'] == score['recall'] == score['f1'] == .75
    assert score['psnr_db'] == pytest.approx(-10*np.log10(2/64))
    assert score['drd'] == pytest.approx(brute_drd(gt, pred))


def test_skeleton_pseudo_recall_is_not_ordinary_recall():
    gt = np.zeros((15, 17), np.uint8)
    gt[3:12, 4:13] = 255
    reference = BinaryReference(gt)
    pred = reference.skeleton.astype(np.uint8) * 255
    score = reference.score(pred)
    assert score['pseudo_recall'] == score['precision'] == score['pfm_skeleton'] == 1
    assert score['recall'] < 1 and score['f1'] < 1


@pytest.mark.parametrize('shape', [(8, 8), (9, 13), (16, 17)])
def test_vectorized_drd_matches_explicit_neighbors_including_corners(shape):
    gt = np.random.default_rng(9).integers(0, 2, shape, dtype=np.uint8) * 255
    pred = gt.copy()
    pred[0, 0] = 255-pred[0, 0]
    pred[-1, -1] = 255-pred[-1, -1]
    pred[3, 4] = 255-pred[3, 4]
    assert BinaryReference(gt).score(pred)['drd'] == pytest.approx(brute_drd(gt, pred), abs=1e-12)


def test_perfect_and_empty_predictions_have_explicit_edge_conventions():
    gt = np.zeros((9, 11), np.uint8)
    score = BinaryReference(gt).score(gt)
    assert score['f1'] == score['pfm_skeleton'] == 1
    assert score['psnr_db'] == float('inf') and score['drd'] == 0
    pred = gt.copy(); pred[0, 0] = 255
    score = BinaryReference(gt).score(pred)
    assert score['f1'] == score['pfm_skeleton'] == 0 and np.isnan(score['drd'])
    gt[3:6, 3:6] = 255
    score = BinaryReference(gt).score(np.zeros_like(gt))
    assert score['f1'] == score['pfm_skeleton'] == score['recall'] == 0


@pytest.mark.parametrize('bad', [np.zeros((5, 5), float), np.ones((5, 5), np.uint8),
                                 np.zeros((5, 5, 3), np.uint8), np.zeros((0, 5), np.uint8)])
def test_reject_nonbinary_or_nonuint8_inputs(bad):
    with pytest.raises(ValueError, match='binary image'):
        BinaryReference(bad)


def test_reference_does_not_mutate_and_rejects_shape_mismatch():
    gt = np.zeros((9, 11), np.uint8); gt[3:6, 3:6] = 255
    original = gt.copy(); reference = BinaryReference(gt)
    reference.score(gt)
    assert np.array_equal(gt, original)
    with pytest.raises(ValueError, match='same shape'):
        reference.score(np.zeros((9, 10), np.uint8))
