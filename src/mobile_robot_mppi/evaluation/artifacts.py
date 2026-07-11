import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from mobile_robot_mppi.core.config import config_hash, git_sha


class ArtifactWriter:
    def __init__(self, output_dir, config, project_root=None):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = dict(config)
        self.provenance = {
            "git_sha": git_sha(project_root),
            "config_hash": config_hash(config),
        }
        with (self.output_dir / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.config, handle, sort_keys=True)
        with (self.output_dir / "provenance.json").open("w", encoding="utf-8") as handle:
            json.dump(self.provenance, handle, indent=2, sort_keys=True)

    def write(self, records: Sequence[Mapping[str, object]], summary: Mapping[str, object], metadata=None):
        if records:
            with (self.output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)
        payload = dict(summary)
        payload["provenance"] = self.provenance
        if metadata:
            payload["metadata"] = dict(metadata)
        with (self.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
