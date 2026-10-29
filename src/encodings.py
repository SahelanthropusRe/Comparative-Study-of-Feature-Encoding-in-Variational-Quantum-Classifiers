"""
encodings.py

Feature-encoding strategies for the multi-dataset Variational Quantum
Classifier (VQC).

Input convention
----------------
Each method receives two finite numerical features. The standard experiment
preprocesses these into [0, 1] using a MinMaxScaler fitted on the training set.

Supported methods
-----------------
ANGLE:
    Standard angle encoding: theta_i = pi * x_i, using RX rotations.

ZZ:
    A data-dependent feature map with RX(pi*x_i) rotations and an RZZ
    interaction whose angle is phi = pi*(2*x_1 - 1)*(2*x_2 - 1).

E1:
    Preserved original method. Maps each feature from [0, 1] to [-pi, pi].

E2:
    Preserved original method. Maps each feature from [0, 1] to
    [-pi/2, pi/2].

E3:
    Preserved original method. Uses the E1 mapping but swaps feature order.

Important integration note
--------------------------
The legacy get_encoding() interface returns two angles and is retained for
ANGLE, E1, E2, and E3. ZZ is a circuit-level feature map, not just a pair of
rotation angles; use apply_encoding() inside a PennyLane QNode to apply it.

This module defines the encoding behavior. It does not itself create a
quantum device or execute a quantum circuit.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np


VALID_ENCODINGS = ("ANGLE", "ZZ", "E1", "E2", "E3")


def _validate_features(features) -> np.ndarray:
    """Return exactly two finite feature values as a NumPy array."""
    values = np.asarray(features, dtype=float)

    if values.shape != (2,):
        raise ValueError(
            "Each encoding expects exactly two features; "
            f"received shape {values.shape}."
        )
    if not np.isfinite(values).all():
        raise ValueError("Feature values must be finite; NaN/inf is not allowed.")

    # The project preprocessing contract is [0, 1]. Test-set values can
    # legitimately fall outside this interval after train-only MinMaxScaler
    # transformation, so this function does not reject or clip them.
    return values


def scale_to_range(x, new_min, new_max):
    """
    Linearly map x from the assumed source range [0, 1] to a target range.

    The function intentionally does not clip x. This keeps encoding behavior
    consistent with train-only scaling when test values fall outside [0, 1].
    """
    return new_min + x * (new_max - new_min)


def encoding_angle(features) -> np.ndarray:
    """
    Standard angle encoding.

    theta_i = pi*x_i; feature 1 -> qubit 0, feature 2 -> qubit 1.
    The quantum circuit applies RX(theta_i) to each qubit.
    """
    x1, x2 = _validate_features(features)
    return np.array([np.pi * x1, np.pi * x2], dtype=float)


def encoding_zz(features) -> np.ndarray:
    """
    Return the three data-dependent angles for the ZZ feature map.

    Returns [theta_1, theta_2, phi], where:
        theta_1 = pi*x_1
        theta_2 = pi*x_2
        phi = pi*(2*x_1 - 1)*(2*x_2 - 1)

    This function only computes angles. In a PennyLane QNode, apply the first
    two as RX rotations and phi as an RZZ interaction.
    """
    x1, x2 = _validate_features(features)
    theta_1 = np.pi * x1
    theta_2 = np.pi * x2
    phi = np.pi * (2.0 * x1 - 1.0) * (2.0 * x2 - 1.0)
    return np.array([theta_1, theta_2, phi], dtype=float)


def encoding_e1(features) -> np.ndarray:
    """
    E1: full-range angle encoding, preserved from the original project.

    Maps [0, 1] to [-pi, pi], keeping the original feature order.
    """
    x1, x2 = _validate_features(features)
    angle_1 = scale_to_range(x1, -np.pi, np.pi)
    angle_2 = scale_to_range(x2, -np.pi, np.pi)
    return np.array([angle_1, angle_2], dtype=float)


def encoding_e2(features) -> np.ndarray:
    """
    E2: compressed angle encoding, preserved from the original project.

    Maps [0, 1] to [-pi/2, pi/2], keeping the original feature order.
    """
    x1, x2 = _validate_features(features)
    angle_1 = scale_to_range(x1, -np.pi / 2.0, np.pi / 2.0)
    angle_2 = scale_to_range(x2, -np.pi / 2.0, np.pi / 2.0)
    return np.array([angle_1, angle_2], dtype=float)


def encoding_e3(features) -> np.ndarray:
    """
    E3: full-range angle encoding with reversed feature assignment.

    Maps [0, 1] to [-pi, pi], then returns feature 2's angle first and
    feature 1's angle second, preserving the original project definition.
    """
    x1, x2 = _validate_features(features)
    angle_for_qubit_0 = scale_to_range(x2, -np.pi, np.pi)
    angle_for_qubit_1 = scale_to_range(x1, -np.pi, np.pi)
    return np.array([angle_for_qubit_0, angle_for_qubit_1], dtype=float)


# Existing E1/E2/E3 names remain available for current project imports.
ENCODING_FUNCTIONS: Dict[str, Callable] = {
    "ANGLE": encoding_angle,
    "E1": encoding_e1,
    "E2": encoding_e2,
    "E3": encoding_e3,
}


def get_encoding(name: str) -> Callable:
    """
    Return an angle-mapping function.

    For ANGLE, E1, E2, and E3, the returned function gives two angles.
    For ZZ, use get_zz_angles() and apply the interaction in the QNode.
    """
    normalized_name = str(name).strip().upper()

    if normalized_name == "ZZ":
        raise ValueError(
            "ZZ is a circuit-level feature map and returns three angles. "
            "Use get_zz_angles() and apply an RZZ interaction in the QNode."
        )

    try:
        return ENCODING_FUNCTIONS[normalized_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown encoding '{name}'. Choose from "
            f"{', '.join(VALID_ENCODINGS)}."
        ) from exc


def get_zz_angles(features) -> np.ndarray:
    """Explicitly return [theta_1, theta_2, phi] for the ZZ feature map."""
    return encoding_zz(features)


def apply_encoding(features, encoding_name: str, wires=(0, 1)):
    """
    Apply an encoding directly inside a PennyLane QNode.

    ANGLE, E1, E2, and E3 apply RX rotations on the two specified wires.
    ZZ applies the same RX rotations followed by RZZ(phi).

    PennyLane is imported locally so the angle functions can be tested even
    in environments where PennyLane is not installed.
    """
    try:
        import pennylane as qml
    except ImportError as exc:
        raise ImportError(
            "PennyLane is required for apply_encoding(); install project "
            "dependencies before building quantum circuits."
        ) from exc

    if len(wires) != 2 or wires[0] == wires[1]:
        raise ValueError("wires must contain two distinct wire identifiers.")

    normalized_name = str(encoding_name).strip().upper()
    values = _validate_features(features)

    if normalized_name == "ZZ":
        theta_1, theta_2, phi = encoding_zz(values)
        qml.RX(theta_1, wires=wires[0])
        qml.RX(theta_2, wires=wires[1])
        qml.IsingZZ(phi, wires=[wires[0], wires[1]])
        return

    angles = get_encoding(normalized_name)(values)
    qml.RX(angles[0], wires=wires[0])
    qml.RX(angles[1], wires=wires[1])


def validate_encodings() -> bool:
    """Run deterministic sanity checks for all five encoding definitions."""
    sample = np.array([0.25, 0.75], dtype=float)
    tolerance = 1e-10

    expected = {
        "ANGLE": np.array([np.pi / 4.0, 3.0 * np.pi / 4.0]),
        "E1": np.array([-np.pi / 2.0, np.pi / 2.0]),
        "E2": np.array([-np.pi / 4.0, np.pi / 4.0]),
        "E3": np.array([np.pi / 2.0, -np.pi / 2.0]),
    }

    print("=" * 72)
    print("FEATURE ENCODING VALIDATION")
    print("=" * 72)
    print(f"Input features: {sample.tolist()}")

    for name, expected_angles in expected.items():
        actual = get_encoding(name)(sample)
        if actual.shape != (2,) or not np.allclose(
            actual, expected_angles, atol=tolerance
        ):
            raise AssertionError(
                f"{name} failed: expected {expected_angles}, received {actual}."
            )
        print(f"{name:>6}: {np.array2string(actual, precision=6)}  PASS")

    expected_zz = np.array(
        [
            np.pi / 4.0,
            3.0 * np.pi / 4.0,
            -np.pi / 4.0,
        ]
    )
    actual_zz = get_zz_angles(sample)
    if actual_zz.shape != (3,) or not np.allclose(
        actual_zz, expected_zz, atol=tolerance
    ):
        raise AssertionError(
            f"ZZ failed: expected {expected_zz}, received {actual_zz}."
        )
    print(f"{'ZZ':>6}: {np.array2string(actual_zz, precision=6)}  PASS")

    # Validate input checking.
    for invalid in ([0.2], [0.2, 0.3, 0.4], [0.2, np.nan]):
        try:
            encoding_e1(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid input was not rejected: {invalid}")

    print("Input validation: PASS")
    print("\nAll five encoding definitions passed their numerical checks.")
    print(
        "Note: circuit execution and ZZ gate integration will be tested "
        "when the quantum model is updated."
    )
    return True


if __name__ == "__main__":
    validate_encodings()
