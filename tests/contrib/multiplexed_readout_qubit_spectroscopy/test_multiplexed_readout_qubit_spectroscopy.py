"""Tests for multiplexed-readout qubit spectroscopy contrib helper."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qubex import contrib
from qubex.backend.quel1.managers.frequency_configuration_manager import (
    Quel1FrequencyConfigurationManager,
)
from qubex.backend.quel1.sweep_configuration import Quel1SweepConfiguration
from qubex.contrib.experiment.multiplexed_readout_qubit_spectroscopy import (
    multiplexed_readout_qubit_spectroscopy,
)
from qubex.experiment.models.result import Result
from qubex.system.quel1.quel1_system_synchronizer import Quel1SystemSynchronizer
from qubex.system.system_manager import SystemManager

QUBITS = ["Q00", "Q01", "Q02", "Q03", "Q04"]
READOUT_FREQUENCIES = {
    "RQ00": 7.30,
    "RQ01": 7.10,
    "RQ02": 7.40,
    "RQ03": 7.20,
}


class _Capture:
    def __init__(self, value: complex) -> None:
        self.kerneled = value


class _MeasurementResult:
    def __init__(self, data: dict[str, complex]) -> None:
        self.data = {key: [_Capture(value)] for key, value in data.items()}


class _SystemManager:
    """Plan drive and readout configurations and record context entry and exit."""

    def __init__(
        self,
        segment_indices: list[tuple[int, ...]] | None,
        unrealizable_readouts: set[str],
    ) -> None:
        self.segment_indices = segment_indices
        self.unrealizable_readouts = unrealizable_readouts
        self.plan_calls: list[dict[str, Any]] = []
        self.events: list[tuple[str, str]] = []
        self.stack: list[str] = []

    @property
    def entered(self) -> list[str]:
        return [name for event, name in self.events if event == "enter"]

    @property
    def active(self) -> int:
        return len(self.stack)

    def plan_frequency_sweep(
        self,
        target: str,
        frequencies: Any,
        *,
        subrange_width: float | None = None,
    ) -> tuple[SimpleNamespace, ...]:
        values = np.asarray(frequencies, dtype=float)
        self.plan_calls.append(
            {"target": target, "frequencies": values, "subrange_width": subrange_width}
        )
        if target.startswith("R"):
            groups = (
                [(i,) for i in range(len(values))]
                if target in self.unrealizable_readouts
                else [tuple(range(len(values)))]
            )
            prefix = f"readout-{target}"
        else:
            groups = self.segment_indices or [tuple(range(len(values)))]
            prefix = "segment"
        return tuple(
            SimpleNamespace(
                indices=indices,
                values=tuple(float(values[i]) for i in indices),
                configuration=prefix if prefix != "segment" else f"segment-{n}",
            )
            for n, indices in enumerate(groups)
        )

    @contextmanager
    def modified_sweep_configuration(self, configuration: str):
        self.events.append(("enter", configuration))
        self.stack.append(configuration)
        try:
            yield
        finally:
            assert self.stack.pop() == configuration
            self.events.append(("exit", configuration))


class _Context:
    def __init__(
        self,
        segment_indices: list[tuple[int, ...]] | None,
        unrealizable_readouts: set[str],
    ) -> None:
        self.qubit_labels = list(QUBITS)
        self.params = SimpleNamespace(
            readout_amplitude={q: 0.01 * (i + 1) for i, q in enumerate(QUBITS)},
        )
        self.reset_calls: list[set[str]] = []
        self.system_manager = _SystemManager(segment_indices, unrealizable_readouts)
        self.experiment_system = SimpleNamespace(
            get_control_box_for_qubit=lambda _qubit: SimpleNamespace(
                traits=SimpleNamespace(
                    default_control_frequency_range=(5.00, 5.025, 0.01),
                )
            ),
            get_mux_by_qubit=lambda qubit: SimpleNamespace(index=int(qubit[1:]) // 4),
        )

    def resolve_qubit_label(self, label: str) -> str:
        if label in self.qubit_labels:
            return label
        if label.startswith("R") and label[1:] in self.qubit_labels:
            return label[1:]
        raise ValueError(f"Qubit label could not be resolved from `{label}`.")

    def resolve_read_label(self, label: str) -> str:
        return f"R{self.resolve_qubit_label(label)}"

    def reset_awg_and_capunits(self, *, qubits: set[str]) -> None:
        self.reset_calls.append(set(qubits))


class _MeasurementService:
    """Encode control point and readout identity in each slot's signal."""

    def __init__(self, ctx: _Context, fail_on_call: int | None) -> None:
        self.ctx = ctx
        self.fail_on_call = fail_on_call
        self.calls: list[dict[str, Any]] = []
        self.stacks: list[list[str]] = []
        self.on_execute: Any = lambda _kwargs: None

    def execute(self, **kwargs: Any) -> _MeasurementResult:
        self.calls.append(kwargs)
        self.stacks.append(list(self.ctx.system_manager.stack))
        self.on_execute(kwargs)
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("hardware failure")
        schedule = kwargs["schedule"]
        frequencies = kwargs["frequencies"]
        drives = _drive_labels(schedule)
        assert len(drives) == 1
        drive = drives[0]
        drive_amplitude = float(np.max(np.abs(schedule.get_sequence(drive).values)))
        data: dict[str, complex] = {}
        for label in schedule.labels:
            if label.startswith("R"):
                # Real part: control point; imaginary part: this slot's frequency.
                data[label[1:]] = complex(
                    frequencies[drive] + 100 * drive_amplitude,
                    frequencies[label],
                )
        return _MeasurementResult(data)


class _Util:
    """Fail if the helper falls back to qubit-index subgroups."""

    def create_qubit_subgroups(self, qubits: Any) -> list[list[str]]:
        raise AssertionError("readout candidates must not be grouped by qubit index")


class _Experiment:
    def __init__(
        self,
        *,
        fail_on_call: int | None = None,
        segment_indices: list[tuple[int, ...]] | None = None,
        unrealizable_readouts: set[str] | None = None,
    ) -> None:
        self.ctx = _Context(segment_indices, unrealizable_readouts or set())
        self.util = _Util()
        self.measurement_service = _MeasurementService(self.ctx, fail_on_call)


def _drive_labels(schedule: Any) -> list[str]:
    return [
        label
        for label in schedule.labels
        if not label.startswith("R")
        and np.any(np.abs(schedule.get_sequence(label).values) > 0)
    ]


def _readout_labels(schedule: Any) -> list[str]:
    return [label for label in schedule.labels if label.startswith("R")]


def _first_active_index(schedule: Any, label: str) -> int:
    values = np.abs(schedule.get_sequence(label).values)
    return int(np.flatnonzero(values > 1e-6)[0])


def _run(exp: _Experiment, **kwargs: Any) -> Result:
    options: dict[str, Any] = {
        "target": "Q04",
        "readout_frequencies": dict(READOUT_FREQUENCIES),
        "frequency_range": [5.00, 5.01, 5.02],
        "power_range": [-20.0, -10.0],
        "plot": False,
        "save_image": False,
    }
    options.update(kwargs)
    target = options.pop("target")
    return multiplexed_readout_qubit_spectroscopy(cast(Any, exp), target, **options)


def test_drives_only_the_requested_target_with_all_readouts() -> None:
    """Every schedule drives exactly one control target and reads every candidate slot."""
    exp = _Experiment()

    _run(exp, readout_grouping="multiplexed")

    calls = exp.measurement_service.calls
    assert len(calls) == 6
    for call in calls:
        schedule = call["schedule"]
        assert _drive_labels(schedule) == ["Q04"]
        assert _readout_labels(schedule) == ["RQ00", "RQ01", "RQ02", "RQ03"]
        assert call["reset_awg_and_capunits"] is False


def test_drive_target_is_independent_of_readout_slots() -> None:
    """The driven qubit's own resonator is neither assumed nor added to the readouts."""
    exp = _Experiment()

    result = _run(exp)

    schedule = exp.measurement_service.calls[0]["schedule"]
    assert "RQ04" not in schedule.labels
    assert result.data["target"] == "Q04"
    assert list(result.data["candidates"]) == ["RQ00", "RQ01", "RQ02", "RQ03"]


def test_applies_candidate_frequencies_per_point() -> None:
    """Each point overrides the drive frequency and every candidate readout frequency."""
    exp = _Experiment()

    _run(exp, readout_grouping="multiplexed")

    frequencies = [call["frequencies"] for call in exp.measurement_service.calls]
    assert frequencies[0] == {"Q04": 5.00, **READOUT_FREQUENCIES}
    assert [f["Q04"] for f in frequencies] == [5.00, 5.01, 5.02] * 2


def test_permuted_candidates_keep_independent_iq_identity() -> None:
    """Each candidate keeps its own complex IQ even when frequencies are permuted."""
    exp = _Experiment()

    result = _run(exp)

    for label, frequency in READOUT_FREQUENCIES.items():
        signals = result.data["candidates"][label]["signals"]
        assert signals.dtype == np.complex128
        assert_allclose(signals.imag, np.full((2, 3), frequency), atol=1e-12)


def test_payload_shapes_axes_and_metadata() -> None:
    """Payload exposes shared axes, candidate metadata, and spectroscopy-compatible phases."""
    exp = _Experiment()

    result = _run(exp, readout_amplitudes={"RQ02": 0.25}, simultaneous_drive=False)

    data = result.data
    assert data["target"] == "Q04"
    assert data["readout_labels"] == ["RQ00", "RQ01", "RQ02", "RQ03"]
    assert data["simultaneous_drive"] is False
    assert_allclose(data["frequency_range"], [5.00, 5.01, 5.02])
    assert_allclose(data["power_range"], [-20.0, -10.0])
    entry = data["candidates"]["RQ02"]
    assert entry["readout_label"] == "RQ02"
    assert entry["candidate_index"] == 2
    assert entry["readout_frequency"] == pytest.approx(7.40)
    assert entry["readout_amplitude"] == pytest.approx(0.25)
    assert data["candidates"]["RQ00"]["readout_amplitude"] == pytest.approx(0.01)
    assert entry["signals"].shape == (2, 3)
    assert entry["phases"].shape == (2, 3)
    assert entry["data"] is entry["phases"]
    assert np.all(np.abs(entry["phases"]) <= np.pi)


def test_signals_follow_frequency_and_power_axes() -> None:
    """Signals are placed by power row and sweep index even for non-contiguous segments."""
    exp = _Experiment(segment_indices=[(0, 2), (1,)])

    result = _run(exp)

    signals = result.data["candidates"]["RQ01"]["signals"]
    amplitudes = np.sqrt(10 ** (np.array([-20.0, -10.0]) / 10))
    expected = np.array([5.00, 5.01, 5.02])[None, :] + 100 * amplitudes[:, None]
    assert_allclose(signals.real, expected, rtol=1e-4)


def test_retunes_only_the_drive_target_once_per_segment() -> None:
    """Sweep planning targets the drive line and each segment is applied once."""
    exp = _Experiment(segment_indices=[(0, 1), (2,)])

    _run(exp, readout_grouping="multiplexed", subrange_width=0.2)

    manager = exp.ctx.system_manager
    drive_plans = [c for c in manager.plan_calls if c["target"] == "Q04"]
    assert len(drive_plans) == 1
    assert drive_plans[0]["subrange_width"] == pytest.approx(0.2)
    assert [c for c in manager.entered if c.startswith("segment")] == [
        "segment-0",
        "segment-1",
    ]


def test_resets_drive_and_all_readout_resources_per_segment() -> None:
    """AWG/capture reset covers the drive qubit and every candidate slot."""
    exp = _Experiment(segment_indices=[(0, 1), (2,)])

    _run(exp, readout_grouping="multiplexed")

    assert exp.ctx.reset_calls == [{"Q00", "Q01", "Q02", "Q03", "Q04"}] * 2


def test_readout_amplitudes_are_applied_per_candidate() -> None:
    """Explicit and default readout amplitudes drive each candidate slot's pulse."""
    exp = _Experiment()

    _run(
        exp,
        readout_grouping="multiplexed",
        readout_amplitudes={"RQ01": 0.3, "Q03": 0.4},
    )

    schedule = exp.measurement_service.calls[0]["schedule"]
    peaks = {
        label: float(np.max(np.abs(schedule.get_sequence(label).values)))
        for label in _readout_labels(schedule)
    }
    assert peaks == pytest.approx(
        {"RQ00": 0.01, "RQ01": 0.3, "RQ02": 0.03, "RQ03": 0.4}
    )


@pytest.mark.parametrize(
    ("simultaneous_drive", "starts_after_drive"),
    [(None, False), (True, False), (False, True)],
)
def test_scheduling_timing_modes(
    simultaneous_drive: bool | None, starts_after_drive: bool
) -> None:
    """Readout starts with the drive by default and after it when not simultaneous."""
    exp = _Experiment()

    _run(exp, simultaneous_drive=simultaneous_drive)

    schedule = exp.measurement_service.calls[0]["schedule"]
    drive_samples = len(schedule.get_sequence("Q04").values)
    for label in _readout_labels(schedule):
        start = _first_active_index(schedule, label)
        if starts_after_drive:
            assert start >= drive_samples // 2
        else:
            assert start < 16


def test_figures_are_keyed_by_candidate() -> None:
    """One phase heatmap per candidate is returned with the first as primary figure."""
    exp = _Experiment()

    result = _run(exp)

    assert result.figures is not None
    assert list(result.figures) == ["RQ00", "RQ01", "RQ02", "RQ03"]
    assert result.figure is result.figures["RQ00"]
    heatmap: Any = result.figures["RQ03"].data[0]
    assert heatmap.type == "heatmap"
    assert_allclose(heatmap.z, result.data["candidates"]["RQ03"]["phases"])
    assert "Q04" in result.figures["RQ03"].layout.title.text
    assert "RQ03" in result.figures["RQ03"].layout.title.text


def test_supports_qubit_label_keys_and_other_candidate_counts() -> None:
    """Candidates may be keyed by qubit or readout labels and are not fixed to four."""
    exp = _Experiment()

    result = _run(
        exp,
        readout_grouping="multiplexed",
        target="Q00",
        readout_frequencies={"Q02": 7.2, "RQ00": 7.0, "Q04": 7.4},
    )

    assert result.data["target"] == "Q00"
    assert result.data["readout_labels"] == ["RQ02", "RQ00", "RQ04"]
    schedule = exp.measurement_service.calls[0]["schedule"]
    assert _readout_labels(schedule) == ["RQ02", "RQ00", "RQ04"]


def test_uses_default_drive_frequency_range() -> None:
    """Omitted frequency range uses the drive target's control-box default range."""
    exp = _Experiment()

    result = _run(exp, frequency_range=None)

    assert_allclose(result.data["frequency_range"], [5.00, 5.01, 5.02])


def test_restores_sweep_configuration_after_failure() -> None:
    """Sweep configuration contexts are exited when a measurement fails."""
    exp = _Experiment(fail_on_call=2, segment_indices=[(0, 1), (2,)])

    with pytest.raises(RuntimeError, match="hardware failure"):
        _run(exp)

    assert exp.ctx.system_manager.active == 0


def test_does_not_mutate_persistent_parameters() -> None:
    """Running the helper leaves configured readout amplitudes untouched."""
    exp = _Experiment()
    before = dict(exp.ctx.params.readout_amplitude)

    _run(exp, readout_amplitudes={"RQ00": 0.9})

    assert exp.ctx.params.readout_amplitude == before


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"target": ["Q00", "Q01"]}, TypeError, "target"),
        ({"target": ""}, ValueError, "target"),
        ({"readout_frequencies": [7.0]}, TypeError, "readout_frequencies"),
        ({"readout_frequencies": {}}, ValueError, "readout_frequencies"),
        (
            {"readout_frequencies": {"Q00": 7.0, "RQ00": 7.1}},
            ValueError,
            "duplicate readout",
        ),
        ({"readout_frequencies": {"RQ00": float("nan")}}, ValueError, "finite"),
        ({"readout_frequencies": {"RQ00": -1.0}}, ValueError, "positive"),
        ({"readout_amplitudes": 0.5}, TypeError, "readout_amplitudes"),
        ({"readout_amplitudes": {"RQ04": 0.5}}, ValueError, "not a readout candidate"),
        ({"readout_amplitudes": {"RQ00": -0.1}}, ValueError, "readout_amplitudes"),
        ({"readout_amplitudes": {"RQ00": float("inf")}}, ValueError, "finite"),
        ({"frequency_range": {"Q04": [5.0]}}, TypeError, "frequency_range"),
        ({"frequency_range": []}, ValueError, "frequency_range"),
        ({"frequency_range": [5.0, float("nan")]}, ValueError, "frequency_range"),
        ({"power_range": [[-20.0]]}, ValueError, "power_range"),
        ({"power_range": []}, ValueError, "power_range"),
        ({"shots": 0}, ValueError, "shots"),
        ({"readout_grouping": "diagonal"}, ValueError, "readout_grouping"),
        ({"readout_grouping": 1}, ValueError, "readout_grouping"),
    ],
)
def test_rejects_invalid_inputs(
    kwargs: dict[str, Any], error: type[Exception], match: str
) -> None:
    """Invalid targets, readout candidates, or sweep axes are rejected before measuring."""
    exp = _Experiment()

    with pytest.raises(error, match=match):
        _run(exp, **kwargs)

    assert exp.measurement_service.calls == []
    assert exp.ctx.system_manager.entered == []


def test_rejects_invalid_grouping_before_planning() -> None:
    """An unknown readout grouping is rejected before any sweep is planned."""
    exp = _Experiment()

    with pytest.raises(ValueError, match="'multiplexed' or 'frequency_pairs'"):
        _run(exp, readout_grouping="subgroups")

    assert exp.ctx.system_manager.plan_calls == []


def test_rejects_candidate_without_default_amplitude() -> None:
    """A candidate without explicit or configured readout amplitude is rejected."""
    exp = _Experiment()
    del exp.ctx.params.readout_amplitude["Q01"]

    with pytest.raises(ValueError, match="RQ01"):
        _run(exp)

    assert exp.measurement_service.calls == []


def test_is_exported_from_contrib() -> None:
    """The helper is exported through the contrib namespace."""
    assert contrib.multiplexed_readout_qubit_spectroscopy is (
        multiplexed_readout_qubit_spectroscopy
    )


def test_default_grouping_reads_frequency_adjacent_pairs() -> None:
    """By default the two lowest and the two highest candidates are read together."""
    exp = _Experiment()

    result = _run(exp)

    assert result.data["readout_grouping"] == "frequency_pairs"
    assert result.data["readout_groups"] == [["RQ01", "RQ03"], ["RQ00", "RQ02"]]


def test_explicit_multiplexed_grouping_reads_every_candidate_in_one_group() -> None:
    """Explicit multiplexed grouping reads every candidate in one group."""
    exp = _Experiment()

    result = _run(exp, readout_grouping="multiplexed")

    assert result.data["readout_grouping"] == "multiplexed"
    assert result.data["readout_groups"] == [["RQ00", "RQ01", "RQ02", "RQ03"]]
    for label in READOUT_FREQUENCIES:
        assert result.data["candidates"][label]["readout_group_index"] == 0


def test_configures_every_readout_before_first_measurement() -> None:
    """Each candidate's readout is configured from all same-mux group frequencies first."""
    exp = _Experiment()

    _run(exp, readout_grouping="multiplexed")

    manager = exp.ctx.system_manager
    readout_plans = [c for c in manager.plan_calls if c["target"].startswith("R")]
    assert [c["target"] for c in readout_plans] == ["RQ00", "RQ01", "RQ02", "RQ03"]
    for plan in readout_plans:
        assert_allclose(plan["frequencies"], list(READOUT_FREQUENCIES.values()))
        assert plan["subrange_width"] is None
    readouts = [f"readout-{label}" for label in READOUT_FREQUENCIES]
    assert manager.entered[:5] == [*readouts, "segment-0"]
    assert exp.measurement_service.stacks[0] == [*readouts, "segment-0"]


def test_readout_plans_use_only_same_mux_candidates() -> None:
    """Candidates on different muxes are configured from their own mux's frequencies."""
    exp = _Experiment()

    _run(
        exp,
        readout_grouping="multiplexed",
        readout_frequencies={"RQ00": 7.0, "RQ04": 7.5, "RQ01": 7.1},
    )

    plans = {
        c["target"]: list(c["frequencies"])
        for c in exp.ctx.system_manager.plan_calls
        if c["target"].startswith("R")
    }
    assert plans == {"RQ00": [7.0, 7.1], "RQ04": [7.5], "RQ01": [7.0, 7.1]}


@pytest.mark.parametrize(
    ("grouping", "expected_groups"),
    [
        (None, [["RQ01", "RQ03"], ["RQ00", "RQ02"]]),
        ("multiplexed", [["RQ00", "RQ01", "RQ02", "RQ03"]]),
        ("frequency_pairs", [["RQ01", "RQ03"], ["RQ00", "RQ02"]]),
    ],
)
def test_grouping_modes_select_measured_readouts(
    grouping: str | None, expected_groups: list[list[str]]
) -> None:
    """Each mode measures its groups in turn with the same drive sweep per group."""
    exp = _Experiment()

    result = _run(exp, readout_grouping=grouping)

    calls = exp.measurement_service.calls
    assert len(calls) == 6 * len(expected_groups)
    measured = [_readout_labels(call["schedule"]) for call in calls]
    assert measured == [group for group in expected_groups for _ in range(6)]
    for call, labels in zip(calls, measured, strict=True):
        assert set(call["frequencies"]) == {"Q04", *labels}
    drive_points = [call["frequencies"]["Q04"] for call in calls]
    assert drive_points == [5.00, 5.01, 5.02] * 2 * len(expected_groups)
    assert result.data["readout_groups"] == expected_groups


@pytest.mark.parametrize(
    ("frequencies", "expected_groups"),
    [
        ({"RQ02": 7.1}, [["RQ02"]]),
        ({"RQ03": 7.4, "RQ00": 7.1}, [["RQ03", "RQ00"]]),
        ({"RQ02": 7.3, "RQ00": 7.1, "RQ01": 7.2}, [["RQ00", "RQ01"], ["RQ02"]]),
        (
            {"RQ00": 7.2, "RQ01": 7.2, "RQ02": 7.1, "RQ03": 7.2, "RQ04": 7.0},
            [["RQ02", "RQ04"], ["RQ00", "RQ01"], ["RQ03"]],
        ),
    ],
)
def test_frequency_pairs_cover_odd_counts_and_ties(
    frequencies: dict[str, float], expected_groups: list[list[str]]
) -> None:
    """Pairs ascend by frequency, ties keep candidate order, and none is omitted."""
    exp = _Experiment()

    result = _run(exp, readout_frequencies=frequencies)

    assert result.data["readout_groups"] == expected_groups
    assert result.data["readout_labels"] == list(frequencies)
    for index, label in enumerate(frequencies):
        entry = result.data["candidates"][label]
        assert entry["candidate_index"] == index
        groups = result.data["readout_groups"]
        assert label in groups[entry["readout_group_index"]]


def test_frequency_pairs_ignore_qubit_index() -> None:
    """Pairs follow candidate frequencies, not slot or qubit labels."""
    exp = _Experiment()

    result = _run(
        exp,
        readout_frequencies={"RQ00": 7.00, "RQ01": 7.40, "RQ02": 7.02, "RQ03": 7.42},
    )

    assert result.data["readout_groups"] == [["RQ00", "RQ02"], ["RQ01", "RQ03"]]


def test_frequency_pairs_configure_readout_per_group() -> None:
    """Each pair configures only its readouts from its own frequencies."""
    exp = _Experiment()

    _run(exp, readout_grouping="frequency_pairs")

    manager = exp.ctx.system_manager
    plans = {
        c["target"]: list(c["frequencies"])
        for c in manager.plan_calls
        if c["target"].startswith("R")
    }
    assert plans == {
        "RQ01": [7.10, 7.20],
        "RQ03": [7.10, 7.20],
        "RQ00": [7.30, 7.40],
        "RQ02": [7.30, 7.40],
    }
    stacks = exp.measurement_service.stacks
    assert stacks[0] == ["readout-RQ01", "readout-RQ03", "segment-0"]
    assert stacks[-1] == ["readout-RQ00", "readout-RQ02", "segment-0"]


def test_frequency_pairs_keep_global_candidate_identity_and_order() -> None:
    """Candidates keep input order, global index, amplitude, and their own signals."""
    exp = _Experiment()

    result = _run(
        exp,
        readout_grouping="frequency_pairs",
        readout_amplitudes={"RQ02": 0.25},
    )

    data = result.data
    assert data["readout_labels"] == ["RQ00", "RQ01", "RQ02", "RQ03"]
    assert list(data["candidates"]) == ["RQ00", "RQ01", "RQ02", "RQ03"]
    assert result.figures is not None
    assert list(result.figures) == ["RQ00", "RQ01", "RQ02", "RQ03"]
    for index, (label, frequency) in enumerate(READOUT_FREQUENCIES.items()):
        entry = data["candidates"][label]
        assert entry["candidate_index"] == index
        assert entry["readout_frequency"] == pytest.approx(frequency)
        assert_allclose(entry["signals"].imag, np.full((2, 3), frequency), atol=1e-12)
    assert data["candidates"]["RQ02"]["readout_amplitude"] == pytest.approx(0.25)
    peaks = {
        label: float(np.max(np.abs(call["schedule"].get_sequence(label).values)))
        for call in exp.measurement_service.calls[:1]
        + exp.measurement_service.calls[6:7]
        for label in _readout_labels(call["schedule"])
    }
    assert peaks == pytest.approx(
        {"RQ00": 0.01, "RQ01": 0.02, "RQ02": 0.25, "RQ03": 0.04}
    )


def test_rejects_unrealizable_multiplexed_group_before_hardware() -> None:
    """An unrealizable multiplexed group is rejected first with a pairing hint."""
    exp = _Experiment(unrealizable_readouts={"RQ01"})

    with pytest.raises(ValueError, match=r"RQ01.*readout_grouping='frequency_pairs'"):
        _run(exp, readout_grouping="multiplexed")

    assert exp.ctx.system_manager.entered == []
    assert exp.measurement_service.calls == []


def test_rejects_unrealizable_pair_with_frequency_diagnostics() -> None:
    """An unrealizable pair is rejected first with its tones, span, and windows."""
    exp = _Experiment(unrealizable_readouts={"RQ01"})

    with pytest.raises(ValueError, match="RQ01") as excinfo:
        _run(exp)

    message = str(excinfo.value)
    assert "RQ01=7.100000, RQ03=7.200000 GHz" in message
    assert "span 100.0 MHz" in message
    assert "2 windows [RQ01]; [RQ03]" in message
    assert "readout_grouping=" not in message
    assert "fewer candidates per call" in message
    assert exp.ctx.system_manager.entered == []
    assert exp.measurement_service.calls == []


def test_restores_every_configuration_after_success() -> None:
    """Readout and drive configurations are exited in reverse order after success."""
    exp = _Experiment(segment_indices=[(0, 1), (2,)])

    _run(exp, readout_grouping="frequency_pairs")

    manager = exp.ctx.system_manager
    assert manager.active == 0
    assert manager.events[:8] == [
        ("enter", "readout-RQ01"),
        ("enter", "readout-RQ03"),
        ("enter", "segment-0"),
        ("exit", "segment-0"),
        ("enter", "segment-1"),
        ("exit", "segment-1"),
        ("exit", "readout-RQ03"),
        ("exit", "readout-RQ01"),
    ]
    assert len(manager.events) == 16


def test_restores_every_configuration_after_failure_in_later_group() -> None:
    """A failure in a later group exits all active configurations in reverse order."""
    exp = _Experiment(fail_on_call=7)

    with pytest.raises(RuntimeError, match="hardware failure"):
        _run(exp, readout_grouping="frequency_pairs")

    manager = exp.ctx.system_manager
    assert manager.active == 0
    assert manager.events[-3:] == [
        ("exit", "segment-0"),
        ("exit", "readout-RQ02"),
        ("exit", "readout-RQ00"),
    ]


class _Quel1ReadoutBackend:
    """Hold one QuEL-1 readout mux: a generator channel and four capture runits."""

    def __init__(self) -> None:
        self.cache: dict[str, Any] = {
            "box": {
                "ports": {
                    1: {
                        "lo_freq": 8_000_000_000,
                        "cnco_freq": 1_500_000_000,
                        "channels": {0: {"fnco_freq": 23_437_500}},
                    },
                    2: {
                        "lo_freq": 8_000_000_000,
                        "cnco_freq": 1_500_000_000,
                        "runits": {r: {"fnco_freq": 23_437_500} for r in range(4)},
                    },
                }
            }
        }
        self.hardware = deepcopy(self.cache)

    def get_box_config_cache(self) -> dict[str, Any]:
        return deepcopy(self.cache)

    def update_box_config_cache(self, values: dict[str, Any]) -> None:
        self.cache.update(deepcopy(values))

    def replace_box_config_cache(self, values: dict[str, Any]) -> None:
        self.cache = deepcopy(values)

    def config_port(self, box: str, *, port: int, lo_freq_hz: Any, cnco_freq_hz: Any):
        self.hardware[box]["ports"][port].update(
            lo_freq=lo_freq_hz, cnco_freq=cnco_freq_hz
        )

    def config_channel(self, box: str, *, port: int, channel: int, fnco_freq_hz: int):
        self.hardware[box]["ports"][port]["channels"][channel]["fnco_freq"] = (
            fnco_freq_hz
        )

    def config_runit(self, box: str, *, port: int, runit: int, fnco_freq_hz: int):
        self.hardware[box]["ports"][port]["runits"][runit]["fnco_freq"] = fnco_freq_hz

    def initialize_awg_and_capunits(self, boxes: Any) -> None:
        pass


class _Quel1SystemManager(_SystemManager):
    """Plan and apply readout settings through the real QuEL-1 synchronizer path."""

    def __init__(self, backend: _Quel1ReadoutBackend) -> None:
        super().__init__(None, set())
        manager = Quel1FrequencyConfigurationManager(cast(Any, backend))
        ports: dict[int, SimpleNamespace] = {}
        for number, values in backend.cache["box"]["ports"].items():
            channels = values.get("channels", values.get("runits"))
            port = SimpleNamespace(
                number=number,
                box_id="box",
                lo_freq=values["lo_freq"],
                cnco_freq=values["cnco_freq"],
            )
            port.channels = [
                SimpleNamespace(number=c, fnco_freq=v["fnco_freq"], port=port)
                for c, v in channels.items()
            ]
            ports[number] = port
        system = SimpleNamespace(
            control_system=SimpleNamespace(
                get_port=lambda _box, number: ports[number],
                get_box=lambda _box: SimpleNamespace(
                    traits=SimpleNamespace(readout_cnco_center=1_500_000_000)
                ),
            ),
            get_target=lambda _label: SimpleNamespace(
                channel=ports[1].channels[0], is_read=True, sideband="U"
            ),
            get_cap_target=lambda label: SimpleNamespace(
                channel=ports[2].channels[int(label[1:].lstrip("Q")) % 4]
            ),
        )
        controller = SimpleNamespace(
            plan_frequency_sweep=manager.plan,
            modified_sweep_configuration=manager.modified,
        )
        synchronizer = Quel1SystemSynchronizer(backend_controller=cast(Any, controller))
        self.system = SimpleNamespace(
            experiment_system=system,
            _resolve_system_synchronizer=lambda: synchronizer,
        )

    def plan_frequency_sweep(
        self,
        target: str,
        frequencies: Any,
        *,
        subrange_width: float | None = None,
    ) -> tuple[Any, ...]:
        if not target.startswith("R"):
            return super().plan_frequency_sweep(
                target, frequencies, subrange_width=subrange_width
            )
        self.plan_calls.append({"target": target, "frequencies": frequencies})
        return SystemManager.plan_frequency_sweep(
            cast(Any, self.system), target, frequencies
        )

    @contextmanager
    def modified_sweep_configuration(self, configuration: Any):
        if not isinstance(configuration, Quel1SweepConfiguration):
            with super().modified_sweep_configuration(configuration):
                yield
            return
        with SystemManager.modified_sweep_configuration(
            cast(Any, self.system), configuration
        ):
            yield


def _quel1_experiment() -> tuple[_Experiment, _Quel1ReadoutBackend]:
    backend = _Quel1ReadoutBackend()
    exp = _Experiment()
    exp.ctx.system_manager = _Quel1SystemManager(backend)
    return exp, backend


# Candidates as QDash assigns them: sorted by frequency onto RQ00..RQ03. The two
# extremes span 440 MHz, beyond one QuEL-1 safe AWG window of +/-200 MHz.
QUEL1_CANDIDATES = {"RQ02": 10.40, "RQ00": 10.00, "RQ03": 10.44, "RQ01": 10.05}


def test_quel1_planner_reads_frequency_pairs_under_shared_oscillators() -> None:
    """Each QuEL-1 pair shares LO/CNCO and tunes only its own runits within the window."""
    exp, backend = _quel1_experiment()
    original = deepcopy(backend.hardware)
    applied: dict[tuple[str, ...], dict[int, Any]] = {}

    def snapshot(kwargs: dict[str, Any]) -> None:
        labels = tuple(_readout_labels(kwargs["schedule"]))
        applied.setdefault(labels, deepcopy(backend.hardware["box"]["ports"]))

    exp.measurement_service.on_execute = snapshot

    result = _run(exp, readout_frequencies=dict(QUEL1_CANDIDATES))

    assert result.data["readout_groups"] == [["RQ00", "RQ01"], ["RQ02", "RQ03"]]
    assert set(applied) == {("RQ00", "RQ01"), ("RQ02", "RQ03")}
    for labels, ports in applied.items():
        generator, capture = ports[1], ports[2]
        assert (capture["lo_freq"], capture["cnco_freq"]) == (
            generator["lo_freq"],
            generator["cnco_freq"],
        )
        assert generator["channels"][0]["fnco_freq"] == 0
        carrier = (generator["lo_freq"] + generator["cnco_freq"]) / 1e9
        for label in labels:
            assert abs(QUEL1_CANDIDATES[label] - carrier) <= 0.2
        runits = {int(label[-1]) for label in labels}
        assert {
            r: v["fnco_freq"] for r, v in capture["runits"].items() if r in runits
        } == dict.fromkeys(runits, 0)
        assert all(
            v["fnco_freq"] == 23_437_500
            for r, v in capture["runits"].items()
            if r not in runits
        )
    assert backend.hardware == backend.cache == original


def test_quel1_planner_rejects_extreme_pair_before_hardware() -> None:
    """A QuEL-1 pair spanning both extremes is rejected with its split windows."""
    exp, backend = _quel1_experiment()
    original = deepcopy(backend.hardware)

    with pytest.raises(ValueError, match="RQ00") as excinfo:
        _run(
            exp,
            readout_grouping="multiplexed",
            readout_frequencies={"RQ00": 10.00, "RQ03": 10.44},
        )

    message = str(excinfo.value)
    assert "RQ00=10.000000, RQ03=10.440000 GHz span 440.0 MHz" in message
    assert "2 windows [RQ00]; [RQ03]" in message
    assert exp.measurement_service.calls == []
    assert backend.hardware == backend.cache == original
