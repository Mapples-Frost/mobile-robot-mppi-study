"""Deterministic array replay buffer with compact resumable checkpoints."""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, observation_dim, action_dim, seed=0):
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        if min(self.capacity, self.observation_dim, self.action_dim) <= 0:
            raise ValueError("replay buffer dimensions and capacity must be positive")
        self.observations = np.empty((self.capacity, self.observation_dim), dtype=np.float32)
        self.actions = np.empty((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.next_observations = np.empty((self.capacity, self.observation_dim), dtype=np.float32)
        self.dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.groups = np.empty((self.capacity,), dtype=np.int32)
        # -1: current/incomplete episode, 0: completed failure, 1: completed
        # success.  Labels are assigned only when the episode terminates.
        self.outcomes = np.full((self.capacity,), -1, dtype=np.int8)
        self.transition_ids = np.full((self.capacity,), -1, dtype=np.int64)
        self.next_transition_id = 0
        self.position = 0
        self.size = 0
        self.rng = np.random.RandomState(int(seed))

    def add(self, observation, action, reward, next_observation, done, group=0):
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        next_observation = np.asarray(next_observation, dtype=np.float32).reshape(-1)
        if observation.shape != (self.observation_dim,) or next_observation.shape != observation.shape:
            raise ValueError("replay observation dimension mismatch")
        if action.shape != (self.action_dim,):
            raise ValueError("replay action dimension mismatch")
        if not (
            np.isfinite(observation).all()
            and np.isfinite(action).all()
            and np.isfinite(next_observation).all()
            and np.isfinite(float(reward))
        ):
            raise ValueError("replay transition must be finite")
        group = int(group)
        if group < 0:
            raise ValueError("replay group must be a non-negative integer")
        index = self.position
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index, 0] = float(reward)
        self.next_observations[index] = next_observation
        self.dones[index, 0] = float(bool(done))
        self.groups[index] = group
        self.outcomes[index] = -1
        transition_id = int(self.next_transition_id)
        self.transition_ids[index] = transition_id
        self.next_transition_id += 1
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return index, transition_id

    def mark_episode_outcome(self, handles, success):
        """Label transitions without corrupting entries overwritten by wraparound."""

        outcome = np.int8(1 if bool(success) else 0)
        marked = 0
        for handle in handles:
            index, transition_id = (int(value) for value in handle)
            if not 0 <= index < self.capacity:
                continue
            if int(self.transition_ids[index]) != transition_id:
                continue
            self.outcomes[index] = outcome
            marked += 1
        return marked

    def _uniform_indices(self, batch_size):
        return self.rng.randint(0, self.size, size=int(batch_size))

    def _balanced_indices(self, batch_size):
        available_groups = np.unique(self.groups[: self.size])
        if available_groups.size == 0:
            raise ValueError("scene-balanced replay has no populated groups")
        base = int(batch_size) // int(available_groups.size)
        remainder = int(batch_size) % int(available_groups.size)
        counts = np.full(available_groups.size, base, dtype=np.int64)
        if remainder:
            order = self.rng.permutation(available_groups.size)
            counts[order[:remainder]] += 1
        selected = []
        for group, count in zip(available_groups, counts):
            if count <= 0:
                continue
            candidates = np.flatnonzero(self.groups[: self.size] == group)
            selected.append(
                self.rng.choice(candidates, size=int(count), replace=True)
            )
        indices = np.concatenate(selected).astype(np.int64, copy=False)
        return indices[self.rng.permutation(indices.size)]

    def _outcome_balanced_indices(self, batch_size, success_fraction):
        success_fraction = float(success_fraction)
        if not np.isfinite(success_fraction) or not 0.0 <= success_fraction <= 1.0:
            raise ValueError("replay success_fraction must be in [0, 1]")
        outcomes = self.outcomes[: self.size]
        success_candidates = np.flatnonzero(outcomes == 1)
        other_candidates = np.flatnonzero(outcomes != 1)
        # Before the first successful episode, preserve ordinary replay rather
        # than blocking SAC updates or fabricating a positive transition.
        if success_candidates.size == 0 or success_fraction <= 0.0:
            return self._uniform_indices(batch_size)
        if other_candidates.size == 0 or success_fraction >= 1.0:
            return self.rng.choice(
                success_candidates, size=int(batch_size), replace=True
            )
        success_count = int(round(int(batch_size) * success_fraction))
        success_count = min(max(success_count, 1), int(batch_size) - 1)
        selected_success = self.rng.choice(
            success_candidates, size=success_count, replace=True
        )
        selected_other = self.rng.choice(
            other_candidates,
            size=int(batch_size) - success_count,
            replace=True,
        )
        indices = np.concatenate((selected_success, selected_other)).astype(
            np.int64, copy=False
        )
        return indices[self.rng.permutation(indices.size)]

    def sample(self, batch_size, strategy="uniform", success_fraction=0.25):
        batch_size = int(batch_size)
        if batch_size <= 0 or self.size < batch_size:
            raise ValueError("replay buffer does not contain the requested batch")
        strategy = str(strategy)
        if strategy == "uniform":
            indices = self._uniform_indices(batch_size)
        elif strategy == "scene_balanced":
            indices = self._balanced_indices(batch_size)
        elif strategy == "outcome_balanced":
            indices = self._outcome_balanced_indices(
                batch_size, success_fraction
            )
        else:
            raise ValueError(
                "replay sampling strategy must be uniform, scene_balanced, "
                "or outcome_balanced"
            )
        return {
            "observations": self.observations[indices].copy(),
            "actions": self.actions[indices].copy(),
            "rewards": self.rewards[indices].copy(),
            "next_observations": self.next_observations[indices].copy(),
            "dones": self.dones[indices].copy(),
            "groups": self.groups[indices].copy(),
            "outcomes": self.outcomes[indices].copy(),
        }

    def group_counts(self):
        if self.size == 0:
            return {}
        groups, counts = np.unique(self.groups[: self.size], return_counts=True)
        return {
            int(group): int(count) for group, count in zip(groups, counts)
        }

    def outcome_counts(self):
        if self.size == 0:
            return {}
        outcomes, counts = np.unique(
            self.outcomes[: self.size], return_counts=True
        )
        return {
            int(outcome): int(count) for outcome, count in zip(outcomes, counts)
        }

    def state_dict(self, include_data=True):
        state = {
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "position": self.position,
            "size": self.size,
            "rng_state": self.rng.get_state(),
            "next_transition_id": self.next_transition_id,
        }
        if include_data:
            state.update({
                "observations": self.observations[: self.size].copy(),
                "actions": self.actions[: self.size].copy(),
                "rewards": self.rewards[: self.size].copy(),
                "next_observations": self.next_observations[: self.size].copy(),
                "dones": self.dones[: self.size].copy(),
                "groups": self.groups[: self.size].copy(),
                "outcomes": self.outcomes[: self.size].copy(),
                "transition_ids": self.transition_ids[: self.size].copy(),
            })
        return state

    @classmethod
    def from_state_dict(cls, state, seed=0):
        result = cls(state["capacity"], state["observation_dim"], state["action_dim"], seed)
        size = int(state.get("size", 0))
        if "observations" in state:
            result.observations[:size] = np.asarray(state["observations"], dtype=np.float32)
            result.actions[:size] = np.asarray(state["actions"], dtype=np.float32)
            result.rewards[:size] = np.asarray(state["rewards"], dtype=np.float32)
            result.next_observations[:size] = np.asarray(state["next_observations"], dtype=np.float32)
            result.dones[:size] = np.asarray(state["dones"], dtype=np.float32)
            if "groups" in state:
                result.groups[:size] = np.asarray(
                    state["groups"], dtype=np.int32
                )
            else:
                # v1 checkpoints predate scene labels and remain resumable as
                # one replay group rather than being silently discarded.
                result.groups[:size] = 0
            if "outcomes" in state:
                result.outcomes[:size] = np.asarray(
                    state["outcomes"], dtype=np.int8
                )
            else:
                # Older checkpoints contain no episode outcomes; treating
                # them as completed non-success samples keeps resume behavior
                # deterministic while never inventing successful experience.
                result.outcomes[:size] = 0
            if "transition_ids" in state:
                result.transition_ids[:size] = np.asarray(
                    state["transition_ids"], dtype=np.int64
                )
            else:
                result.transition_ids[:size] = np.arange(size, dtype=np.int64)
            result.size = size
            result.position = int(state.get("position", size % result.capacity))
            result.next_transition_id = int(state.get(
                "next_transition_id",
                int(result.transition_ids[:size].max()) + 1 if size else 0,
            ))
        if "rng_state" in state:
            result.rng.set_state(state["rng_state"])
        return result
