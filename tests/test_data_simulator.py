import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "outputs"))

from data_simulator import Config, simulate  # noqa: E402


def test_simulator_is_deterministic_and_writes_valid_yolo(tmp_path: Path) -> None:
    config = Config(output_dir=str(tmp_path / "first"), num_samples=6, image_width=128, image_height=128, seed=42)
    first = simulate(config)
    labels = sorted((first / "labels").glob("*.txt"))
    assert len(labels) == config.num_samples
    csv_hash = hashlib.sha256((first / "plc_timeseries.csv").read_bytes()).hexdigest()

    second = simulate(Config(**{**config.__dict__, "output_dir": str(tmp_path / "second")}))
    assert csv_hash == hashlib.sha256((second / "plc_timeseries.csv").read_bytes()).hexdigest()
    for label in labels:
        for row in label.read_text(encoding="utf-8").splitlines():
            class_id, x_center, y_center, width, height = map(float, row.split())
            assert class_id in (0, 1)
            assert all(0 <= item <= 1 for item in (x_center, y_center, width, height))
            assert width > 0 and height > 0
