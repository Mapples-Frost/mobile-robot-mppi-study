"""Causal multi-target LaserScan association for probabilistic forecasting."""

from dataclasses import replace
from typing import Mapping

import numpy as np

from mobile_robot_mppi.obstacles.online_tracking import (
    OnlineTrackingUpdate,
    SingleObstacleChangeAwareTracker,
)


class MultiObstacleChangeAwareTracker:
    """Track a bounded number of scan clusters with independent frozen IMMs.

    Association uses only the current LaserScan and each track's causal
    one-step prediction.  No simulator identity, future trajectory or motion
    seed is accepted by the public API.
    """

    def __init__(self, trackers):
        trackers = tuple(trackers)
        if len(trackers) < 2:
            raise ValueError("multi-obstacle tracker requires at least 2 tracks")
        if not all(
            isinstance(item, SingleObstacleChangeAwareTracker)
            for item in trackers
        ):
            raise TypeError("multi-obstacle tracks must use the frozen IMM")
        self.trackers = trackers
        self.config = trackers[0].config
        if any(item.config != self.config for item in trackers[1:]):
            raise ValueError("multi-obstacle tracker configs must match")
        self.maximum_tracks = len(trackers)
        self.reset()

    @classmethod
    def from_mapping(cls, project_root, values: Mapping[str, object]):
        maximum_tracks = int(values.get("maximum_tracks", 1))
        if maximum_tracks < 2:
            raise ValueError("maximum_tracks must be at least 2")
        return cls(
            SingleObstacleChangeAwareTracker.from_mapping(
                project_root, values
            )
            for _ in range(maximum_tracks)
        )

    def reset(self):
        for tracker in self.trackers:
            tracker.reset()
        self.last_timestamp = None
        self.update_count = 0
        self.retirement_count = 0

    def scan_clusters(self, observation):
        return self.trackers[0].scan_clusters(observation)

    def _assign(self, clusters, timestamp):
        assignments = {}
        available_tracks = set(range(self.maximum_tracks))
        available_clusters = set(range(len(clusters)))
        pairs = []
        for track_index, tracker in enumerate(self.trackers):
            reference = tracker._association_reference(timestamp)
            if reference is None:
                continue
            for cluster_index, cluster in enumerate(clusters):
                distance = float(
                    np.linalg.norm(cluster.center - reference)
                )
                if distance <= self.config.association_gate_m:
                    pairs.append((
                        distance,
                        track_index,
                        cluster_index,
                    ))
        for distance, track_index, cluster_index in sorted(pairs):
            if (
                track_index not in available_tracks
                or cluster_index not in available_clusters
            ):
                continue
            assignments[track_index] = (cluster_index, distance)
            available_tracks.remove(track_index)
            available_clusters.remove(cluster_index)

        uninitialized = [
            index for index in sorted(available_tracks)
            if self.trackers[index].predictor.state is None
        ]
        remaining_clusters = sorted(
            available_clusters,
            key=lambda index: (
                -clusters[index].support_beams,
                clusters[index].minimum_range_m,
                index,
            ),
        )
        for track_index, cluster_index in zip(
            uninitialized, remaining_clusters
        ):
            assignments[track_index] = (cluster_index, None)
        return assignments

    @staticmethod
    def _cluster_observation(observation, cluster):
        if observation.scan is None:
            return observation
        scan = observation.scan
        ranges = np.full_like(
            np.asarray(scan.ranges, dtype=np.float64),
            float(scan.range_max),
        )
        if cluster is not None:
            indices = np.asarray(cluster.beam_indices, dtype=np.int64)
            ranges[indices] = np.asarray(
                scan.ranges, dtype=np.float64
            )[indices]
        obstacle_ranges = np.where(
            ranges < float(scan.range_max) - 1.0e-12,
            ranges,
            np.inf,
        )
        return replace(
            observation,
            scan=replace(
                scan,
                ranges=ranges,
                obstacle_ranges=obstacle_ranges,
            ),
        )

    def update(self, observation):
        timestamp = float(observation.timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("tracker timestamp must be finite")
        if (
            self.last_timestamp is not None
            and timestamp <= float(self.last_timestamp)
        ):
            raise ValueError("tracker timestamps must increase strictly")
        retired_track_indices = []
        retirement_duration = float(
            self.config.track_retirement_duration_s
        )
        if retirement_duration > 0.0:
            for track_index, tracker in enumerate(self.trackers):
                if tracker.last_associated_timestamp is None:
                    continue
                age = (
                    timestamp
                    - float(tracker.last_associated_timestamp)
                )
                if age > retirement_duration + 1.0e-12:
                    tracker.reset()
                    retired_track_indices.append(track_index)
            self.retirement_count += len(retired_track_indices)
        clusters = self.scan_clusters(observation)
        assignments = self._assign(clusters, timestamp)
        updates = []
        for track_index, tracker in enumerate(self.trackers):
            assignment = assignments.get(track_index)
            cluster = (
                None if assignment is None else clusters[assignment[0]]
            )
            filtered = self._cluster_observation(observation, cluster)
            updates.append(tracker.update(filtered))
        self.last_timestamp = timestamp
        self.update_count += 1

        associated = [
            (index, update)
            for index, update in enumerate(updates)
            if update.measurement is not None
        ]
        origin = self.trackers[0]._sensor_origin(observation)
        nearest = (
            min(
                associated,
                key=lambda item: float(
                    np.linalg.norm(item[1].measurement - origin)
                ),
            )
            if associated
            else None
        )
        forecast_pairs = tuple(
            (index, update.forecast)
            for index, update in enumerate(updates)
            if update.forecast is not None
        )
        forecasts = tuple(value for _, value in forecast_pairs)
        forecast_track_indices = tuple(
            index for index, _ in forecast_pairs
        )
        nearest_track_index = (
            None if nearest is None else int(nearest[0])
        )
        nearest_forecast_index = (
            None
            if nearest_track_index not in forecast_track_indices
            else int(forecast_track_indices.index(nearest_track_index))
        )
        nearest_diagnostics = (
            {} if nearest is None else dict(nearest[1].diagnostics)
        )
        track_diagnostics = []
        for index, update in enumerate(updates):
            values = dict(update.diagnostics)
            values["track_index"] = index
            values["assigned_cluster_index"] = (
                None
                if index not in assignments
                else int(assignments[index][0])
            )
            track_diagnostics.append(values)
        diagnostics = dict(nearest_diagnostics)
        diagnostics.update({
            "enabled": True,
            "multi_obstacle": True,
            "maximum_tracks": self.maximum_tracks,
            "track_count": sum(
                tracker.predictor.state is not None
                for tracker in self.trackers
            ),
            "cluster_count": len(clusters),
            "associated": nearest is not None,
            "associated_track_count": len(associated),
            "nearest_track_index": nearest_track_index,
            "nearest_forecast_index": nearest_forecast_index,
            "forecast_track_indices": forecast_track_indices,
            "valid_forecast_count": len(forecasts),
            "forecast_valid": bool(forecasts),
            "update_count": self.update_count,
            "forecast_count": sum(
                tracker.forecast_count for tracker in self.trackers
            ),
            "retired_track_indices": tuple(retired_track_indices),
            "retirement_count": self.retirement_count,
            "forecast_availability": float(
                sum(tracker.forecast_count for tracker in self.trackers)
                / (self.maximum_tracks * self.update_count)
            ),
            "tracks": track_diagnostics,
        })
        return OnlineTrackingUpdate(
            measurement=(
                None if nearest is None else nearest[1].measurement
            ),
            forecast=forecasts if forecasts else None,
            diagnostics=diagnostics,
        )


__all__ = ["MultiObstacleChangeAwareTracker"]
