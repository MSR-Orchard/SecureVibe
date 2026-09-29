"""Deterministic weighted prompt-group source for joint SecureVibe training."""

from __future__ import annotations

import copy
import hashlib
import logging
import math
import os
import random
from pathlib import Path
from typing import Any

import torch

from slime.rollout.data_source import RolloutDataSourceWithBuffer, pop_first
from slime.utils.data import Dataset
from slime.utils.misc import load_function
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_SOURCES = ("patcheval", "autobax")
_ROUTING = {
    "patcheval": ("patcheval", "repo_patch"),
    "autobax": ("autobax", "app_builder"),
}
_STATE_VERSION = 1


def _configured_value(args: Any, argument_name: str, environment_name: str) -> Any:
    value = getattr(args, argument_name, None)
    if value is not None:
        return value
    return os.environ.get(environment_name)


def _required_path(args: Any, argument_name: str, environment_name: str) -> Path:
    value = _configured_value(args, argument_name, environment_name)
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise ValueError(
            f"Set {environment_name} or args.{argument_name} to a routed JSONL file"
        )
    path = Path(value).expanduser().absolute()
    if not path.is_file():
        raise FileNotFoundError(
            f"Joint dataset does not exist or is not a file: {path}"
        )
    return path


def _weight(args: Any, source: str) -> float:
    environment_name = f"JOINT_{source.upper()}_WEIGHT"
    value = _configured_value(args, f"joint_{source}_weight", environment_name)
    if value is None:
        value = "0.5"
    try:
        weight = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{environment_name} must be numeric, got {value!r}"
        ) from error
    if not math.isfinite(weight) or weight < 0:
        raise ValueError(
            f"{environment_name} must be finite and nonnegative, got {value!r}"
        )
    return weight


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class JointRolloutDataSource(RolloutDataSourceWithBuffer):
    """Sample whole prompt groups from independently shuffled task sources."""

    def __init__(self, args: Any):
        if not getattr(args, "rollout_global_dataset", True):
            raise ValueError(
                "JointRolloutDataSource requires the global rollout dataset"
            )
        self.args = args
        self.paths = {
            "patcheval": _required_path(
                args, "joint_patcheval_data", "JOINT_PATCHEVAL_DATA"
            ),
            "autobax": _required_path(args, "joint_autobax_data", "JOINT_AUTOBAX_DATA"),
        }
        self.weights = {source: _weight(args, source) for source in _SOURCES}
        weight_sum = math.fsum(self.weights.values())
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "Joint task weights must sum to one; got "
                f"patcheval={self.weights['patcheval']} and "
                f"autobax={self.weights['autobax']} (sum={weight_sum})"
            )

        self.fingerprints = {
            source: _fingerprint(path) for source, path in self.paths.items()
        }
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        dump_details = getattr(args, "dump_details", None)
        if dump_details is not None:
            tokenizer.save_pretrained(Path(dump_details) / "tokenizer")
            if processor:
                processor.save_pretrained(Path(dump_details) / "processor")

        self.datasets = {
            source: self._load_dataset(path, tokenizer, processor)
            for source, path in self.paths.items()
        }
        for source in _SOURCES:
            self._validate_dataset(source)
            if self.weights[source] > 0 and not self.datasets[source].origin_samples:
                raise ValueError(
                    f"Joint source {source!r} is empty but has positive weight"
                )

        seed = int(getattr(args, "rollout_seed", 42))
        self._selector_rng = random.Random(seed ^ 0x4A4F494E54)
        self._source_rngs = {
            "patcheval": random.Random(seed ^ 0x5041544348),
            "autobax": random.Random(seed ^ 0x4155544F42),
        }
        self.source_orders = {
            source: list(range(len(self.datasets[source].origin_samples)))
            for source in _SOURCES
        }
        self.source_offsets = dict.fromkeys(_SOURCES, 0)
        self.source_epochs = dict.fromkeys(_SOURCES, 0)
        if getattr(args, "rollout_shuffle", False):
            for source in _SOURCES:
                self._source_rngs[source].shuffle(self.source_orders[source])
        self._apply_source_orders()

        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        self.epoch_id = 0
        self.source_selection_counts = dict.fromkeys(_SOURCES, 0)
        self.source_emission_counts = dict.fromkeys(_SOURCES, 0)
        self.metadata: dict[str, Any] = {}
        self.buffer: list[list[Sample]] = []
        buffer_filter_path = getattr(args, "buffer_filter_path", None)
        self.buffer_filter = (
            pop_first
            if buffer_filter_path is None
            else load_function(buffer_filter_path)
        )
        self._update_metadata()

    def _load_dataset(self, path: Path, tokenizer: Any, processor: Any) -> Dataset:
        return Dataset(
            str(path),
            tokenizer=tokenizer,
            processor=processor,
            max_length=getattr(self.args, "rollout_max_prompt_len", None),
            prompt_key=getattr(self.args, "input_key", "prompt"),
            multimodal_keys=getattr(self.args, "multimodal_keys", None),
            label_key=getattr(self.args, "label_key", "label"),
            metadata_key=getattr(self.args, "metadata_key", "metadata"),
            tool_key=getattr(self.args, "tool_key", "tools"),
            apply_chat_template=getattr(self.args, "apply_chat_template", False),
            apply_chat_template_kwargs=getattr(
                self.args, "apply_chat_template_kwargs", None
            ),
            seed=int(getattr(self.args, "rollout_seed", 42)),
        )

    def _validate_dataset(self, source: str) -> None:
        reward_type, agent_profile = _ROUTING[source]
        for position, sample in enumerate(self.datasets[source].origin_samples):
            metadata = sample.metadata
            location = f"{self.paths[source]} record {position + 1}"
            if not isinstance(metadata, dict):
                raise ValueError(f"{location} has non-object metadata")
            expected = {
                "task_type": source,
                "reward_type": reward_type,
                "agent_profile": agent_profile,
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise ValueError(
                        f"{location} has metadata.{key}={metadata.get(key)!r}; "
                        f"expected {value!r}"
                    )
            config_path = metadata.get("swe_config_path")
            if not isinstance(config_path, str) or not config_path:
                raise ValueError(f"{location} has an invalid metadata.swe_config_path")
            if not Path(config_path).is_file():
                raise FileNotFoundError(
                    f"{location} references missing Mini-SWE config {config_path!r}"
                )

    def _apply_source_orders(self) -> None:
        for source in _SOURCES:
            origin = self.datasets[source].origin_samples
            self.datasets[source].samples = [
                origin[index] for index in self.source_orders[source]
            ]

    def _start_next_source_epoch(self, source: str) -> None:
        self.source_epochs[source] += 1
        self.source_offsets[source] = 0
        self.source_orders[source] = list(
            range(len(self.datasets[source].origin_samples))
        )
        if getattr(self.args, "rollout_shuffle", False):
            self._source_rngs[source].shuffle(self.source_orders[source])
        origin = self.datasets[source].origin_samples
        self.datasets[source].samples = [
            origin[index] for index in self.source_orders[source]
        ]
        self.epoch_id = max(self.source_epochs.values())

    def _select_source(self) -> str:
        next_total = sum(self.source_selection_counts.values()) + 1
        deficits = {
            source: self.weights[source] * next_total
            - self.source_selection_counts[source]
            for source in _SOURCES
            if self.weights[source] > 0
        }
        largest = max(deficits.values())
        candidates = [
            source
            for source, deficit in deficits.items()
            if math.isclose(deficit, largest, rel_tol=0.0, abs_tol=1e-12)
        ]
        source = self._selector_rng.choice(candidates)
        self.source_selection_counts[source] += 1
        return source

    def _next_prompt(self, source: str) -> Sample:
        dataset = self.datasets[source]
        if self.source_offsets[source] >= len(dataset):
            self._start_next_source_epoch(source)
        offset = self.source_offsets[source]
        self.source_offsets[source] += 1
        self.sample_offset = sum(self.source_offsets.values())
        return dataset.samples[offset]

    def _make_group(self, source: str, prompt_sample: Sample) -> list[Sample]:
        group: list[Sample] = []
        group_base_index = self.sample_index
        selection_index = sum(self.source_selection_counts.values()) - 1
        for _ in range(self.args.n_samples_per_prompt):
            sample = copy.deepcopy(prompt_sample)
            sample.group_index = self.sample_group_index
            sample.index = self.sample_index
            sample.metadata["joint_source"] = source
            sample.metadata["joint_source_selection_index"] = selection_index
            self.sample_index += 1
            group.append(sample)

        maximum = getattr(self.args, "n_samples_per_prompt_max", None)
        stride = max(self.args.n_samples_per_prompt, maximum or 0)
        self.sample_index = group_base_index + stride
        self.sample_group_index += 1
        return group

    @staticmethod
    def _group_source(group: list[Sample]) -> str:
        task_types = {
            sample.metadata.get("task_type")
            for sample in group
            if isinstance(sample.metadata, dict)
        }
        if len(task_types) != 1 or next(iter(task_types), None) not in _SOURCES:
            raise ValueError(
                f"Buffered prompt group has mixed or invalid task types: {task_types!r}"
            )
        return next(iter(task_types))

    def _record_emission(self, group: list[Sample], source: str) -> None:
        self.source_emission_counts[source] += 1
        total = sum(self.source_emission_counts.values())
        ratios = {name: self.source_emission_counts[name] / total for name in _SOURCES}
        for sample in group:
            sample.metadata["joint_source"] = source
            sample.metadata["joint_realized_source_ratio"] = dict(ratios)
        self._update_metadata()

    def _update_metadata(self) -> None:
        emitted = sum(self.source_emission_counts.values())
        ratios = {
            source: (self.source_emission_counts[source] / emitted if emitted else 0.0)
            for source in _SOURCES
        }
        self.metadata.update(
            {
                "joint_source_selection_counts": dict(self.source_selection_counts),
                "joint_source_emission_counts": dict(self.source_emission_counts),
                "joint_source_realized_ratio": ratios,
                "joint_source_offsets": dict(self.source_offsets),
                "joint_source_epochs": dict(self.source_epochs),
            }
        )

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        if not isinstance(num_samples, int) or num_samples < 0:
            raise ValueError(
                f"num_samples must be a nonnegative integer, got {num_samples!r}"
            )
        groups = self._get_samples_from_buffer(num_samples)
        for group in groups:
            self._record_emission(group, self._group_source(group))

        for _ in range(num_samples - len(groups)):
            source = self._select_source()
            group = self._make_group(source, self._next_prompt(source))
            self._record_emission(group, source)
            groups.append(group)
        return groups

    def _state_dict(self) -> dict[str, Any]:
        return {
            "joint_data_source_version": _STATE_VERSION,
            "paths": {source: str(self.paths[source]) for source in _SOURCES},
            "fingerprints": dict(self.fingerprints),
            "weights": dict(self.weights),
            "source_orders": copy.deepcopy(self.source_orders),
            "source_offsets": dict(self.source_offsets),
            "source_epochs": dict(self.source_epochs),
            "source_rng_states": {
                source: self._source_rngs[source].getstate() for source in _SOURCES
            },
            "selector_rng_state": self._selector_rng.getstate(),
            "source_selection_counts": dict(self.source_selection_counts),
            "source_emission_counts": dict(self.source_emission_counts),
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
            "buffer": self.buffer,
        }

    @staticmethod
    def _checkpoint_path(root: str | os.PathLike[str], rollout_id: Any) -> Path:
        return Path(root) / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"

    def save(self, rollout_id: Any) -> None:
        save_root = getattr(self.args, "save", None)
        if save_root is None:
            raise ValueError(
                "args.save is required to checkpoint JointRolloutDataSource"
            )
        path = self._checkpoint_path(save_root, rollout_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(self._state_dict(), temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, rollout_id: Any = None) -> None:
        load_root = getattr(self.args, "load", None)
        if load_root is None:
            return
        path = self._checkpoint_path(load_root, rollout_id)
        if not path.exists():
            logger.info("Checkpoint %s does not exist.", path)
            return
        logger.info("Loading joint data-source state from %s", path)
        try:
            state = torch.load(path, weights_only=False)
        except TypeError:
            state = torch.load(path)
        if state.get("joint_data_source_version") != _STATE_VERSION:
            raise ValueError(
                f"Checkpoint {path} is not a supported joint data-source state"
            )
        expected_paths = {source: str(self.paths[source]) for source in _SOURCES}
        if state.get("paths") != expected_paths:
            raise ValueError(
                f"Joint source paths changed since checkpoint: {state.get('paths')!r} "
                f"!= {expected_paths!r}"
            )
        if state.get("fingerprints") != self.fingerprints:
            raise ValueError(
                "Joint source contents changed since the checkpoint was saved"
            )
        if state.get("weights") != self.weights:
            raise ValueError(
                "Joint source weights changed since the checkpoint was saved"
            )

        source_orders = state["source_orders"]
        for source in _SOURCES:
            order = list(source_orders[source])
            expected = list(range(len(self.datasets[source].origin_samples)))
            if sorted(order) != expected:
                raise ValueError(f"Checkpoint contains an invalid {source} permutation")
            self.source_orders[source] = order
            self.source_offsets[source] = int(state["source_offsets"][source])
            if not 0 <= self.source_offsets[source] <= len(order):
                raise ValueError(f"Checkpoint contains an invalid {source} offset")
            self.source_epochs[source] = int(state["source_epochs"][source])
            self._source_rngs[source].setstate(state["source_rng_states"][source])
        self._apply_source_orders()
        self._selector_rng.setstate(state["selector_rng_state"])
        self.source_selection_counts = {
            source: int(state["source_selection_counts"][source]) for source in _SOURCES
        }
        self.source_emission_counts = {
            source: int(state["source_emission_counts"][source]) for source in _SOURCES
        }
        self.sample_group_index = int(state["sample_group_index"])
        self.sample_index = int(state["sample_index"])
        self.sample_offset = sum(self.source_offsets.values())
        self.epoch_id = max(self.source_epochs.values())
        self.metadata = dict(state.get("metadata", {}))
        self.buffer = list(state.get("buffer", []))
        self._update_metadata()

    def __len__(self) -> int:
        return sum(len(self.datasets[source]) for source in _SOURCES)
