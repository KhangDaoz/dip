"""Test vaipe.dataset — đọc nhãn VAIPE (plan §9). Cần data/1_vaipe/."""
import pytest

from vaipe.dataset import list_images, filter_by_label, stats, read_label_json
from common.paths import vaipe_labels, vaipe_images


@pytest.fixture(scope="module")
def train_images():
    return list_images("train")


def test_train_count(train_images):
    # plan chốt 820 ảnh train
    assert len(train_images) == 820


def test_valid_count():
    assert len(list_images("valid")) == 176


def test_test_count():
    assert len(list_images("test")) == 176


def test_first_image_has_annotations(train_images):
    img = train_images[0]
    assert len(img.annotations) > 0


def test_boxes_are_xyxy_pixel(train_images):
    a = train_images[0].annotations[0]
    assert len(a.box) == 4
    assert a.box[2] >= a.box[0] and a.box[3] >= a.box[1]


def test_labels_in_six_classes(train_images):
    from common.contracts import BOX_LABELS
    for img in train_images[:10]:
        for a in img.annotations:
            assert a.label in BOX_LABELS


def test_drugname_count_matches_plan(train_images):
    # plan ghi ~1.995 drugname trong train
    dn = filter_by_label(train_images, "drugname")
    assert 1.5e3 < len(dn) < 2.5e3


def test_mapping_only_on_drugname(train_images):
    for img in train_images[:50]:
        for a in img.annotations:
            if a.mapping is not None:
                assert a.label == "drugname"


def test_stats_keys(train_images):
    s = stats(train_images[:20])
    assert "n_images" in s and "by_label" in s
    assert set(s["by_label"]) == {"drugname", "quantity", "usage", "diagnose", "date", "other"}
