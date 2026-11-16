# Quantum_Dirichlet_Laplacian V2



#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Quantum Dirichlet Laplacian V2.

Promene prema V1:
    - uci se tezina poslednjeg kola;
    - uci se brzina slabljenja istorije;
    - istorijska memorija ima korekciju pocetne mase;
    - analiticki gradijenti ukljucuju oba parametra memorije;
    - optimizacija koristi centriranu funkciju cilja;
    - pri zastoju BFGS-a dozvoljen je nastavak CG postupkom;
    - prag izvornog gradijenta nije olabavljen.

Zadrzano:
    - uceni graf svih 39 brojeva;
    - Dirichletova energija;
    - lokalna kvantna propagacija exp(-i*t*L);
    - kolektivna amplituda cele sedmorke;
    - tacna analiticka normalizacija;
    - sva istorija za zavrsno ucenje;
    - iscrpna NEXT pretraga.

Nema zabrane ponavljanja poslednjeg kola.
Nema prisilnog razlikovanja rezultata razlicitih CSV-ova.
"""


import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import csv
import hashlib
import math
import random
import re
import time
from itertools import combinations, islice
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit.quantum_info import Statevector


random.seed(39)
np.random.seed(39)



FILES = (
    Path("/data/loto7_4720_k91.csv"),
    Path("/data/loto7_4720_k91_loto_2982.csv"),
    Path("/data/loto7_4720_k91_loto_plus_1738.csv"),
)

N = 39
K = 7
QUBITS = 6
DIM = 1 << QUBITS

EDGE_I, EDGE_J = np.triu_indices(N, 1)
EDGE_COUNT = len(EDGE_I)
GRAPH_PARAMETERS = EDGE_COUNT - 1
PARAMETERS = GRAPH_PARAMETERS + 2

TOTAL_EDGE_WEIGHT = N / 2.0

WALK_TIMES = (1.0, 3.0)
REGULARIZATIONS = (0.1, 0.01)

TRAIN_FRACTION = 0.70
VALIDATION_END = 0.85

MAX_RUN = 4
SEARCH_BATCH = 32768
GRADIENT_TOLERANCE = 1e-7
DIAGNOSTICS = False

MEMORY_EPSILON = 1e-6

PARAMETER_SCALE = np.ones(PARAMETERS, dtype=np.float64)
PARAMETER_SCALE[:GRAPH_PARAMETERS] = math.sqrt(EDGE_COUNT)

C1 = math.comb(N - 1, K - 1)
C2 = math.comb(N - 2, K - 2)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_csv(path):
    require(path.is_file(), f"CSV ne postoji: {path}")

    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    require(bool(lines), "CSV je prazan.")

    try:
        dialect = csv.Sniffer().sniff(
            "\n".join(lines[:20]), delimiters=",;\t"
        )
        parsed = csv.reader(lines, dialect)
    except csv.Error:
        parsed = (re.split(r"\s+", line) for line in lines)

    rows = []

    for number, fields in enumerate(parsed, start=1):
        require(
            len(fields) == K,
            f"Red {number}: potrebno je sedam kolona bez hedera.",
        )

        try:
            row = [int(field.strip()) for field in fields]
        except ValueError as exc:
            raise ValueError(
                f"Red {number}: ocekuju se celi brojevi."
            ) from exc

        require(
            all(1 <= value <= N for value in row),
            f"Red {number}: broj izvan opsega 1–39.",
        )
        require(
            len(set(row)) == K,
            f"Red {number}: ponovljen broj unutar kombinacije.",
        )
        rows.append(sorted(row))

    draws = np.asarray(rows, dtype=np.int64)
    require(len(draws) >= 80, "Potrebno je najmanje 80 kola.")
    return draws


def indicators(draws, n=N):
    result = np.zeros((len(draws), n), dtype=np.float64)
    result[np.arange(len(draws))[:, None], draws - 1] = 1.0
    return result


def make_dataset(draws, end=None):
    """
    end je broj istorijskih prelaza.
    Ulazi su redovi 0..end-1, ciljevi su redovi 1..end.
    """
    if end is None:
        end = len(draws) - 1

    require(
        0 < end < len(draws),
        "Neispravna granica skupa podataka.",
    )

    return {
        "history": indicators(draws[:end]),
        "targets": indicators(draws[1:end + 1]),
        "draws": draws[1:end + 1].copy(),
    }


def memory_parameters(parameters):
    direct_fraction = float(expit(parameters[-2]))
    decay_fraction = float(expit(parameters[-1]))

    direct_weight = 2.0 * direct_fraction
    direct_derivative = (
        2.0 * direct_fraction * (1.0 - direct_fraction)
    )

    decay = (
        MEMORY_EPSILON
        + (1.0 - 2.0 * MEMORY_EPSILON) * decay_fraction
    )
    decay_derivative = (
        (1.0 - 2.0 * MEMORY_EPSILON)
        * decay_fraction * (1.0 - decay_fraction)
    )

    return direct_weight, decay, direct_derivative, decay_derivative


def history_states(history, parameters, derivatives=False):
    """
    x_t = (a * poslednje_kolo + i * ucena_memorija) / norma

    Parametri a i decay uce se iz ciljnog NLL-a.
    Memorija koristi samo redove do t, ukljucujuci t.
    """
    a, decay, da, db = memory_parameters(parameters)

    count = len(history)
    states = np.empty((count, N), dtype=np.complex128)

    direct_gradient = (
        np.empty_like(states) if derivatives else None
    )
    decay_gradient = (
        np.empty_like(states) if derivatives else None
    )

    accumulator = np.zeros(N, dtype=np.float64)
    accumulated_mass = 0.0

    accumulator_derivative = np.zeros(N, dtype=np.float64)
    mass_derivative = 0.0

    for t, current in enumerate(history):
        if derivatives:
            accumulator_derivative = (
                (1.0 - decay) * accumulator_derivative
                + current - accumulator
            )
            mass_derivative = (
                (1.0 - decay) * mass_derivative
                + 1.0 - accumulated_mass
            )

        accumulator += decay * (current - accumulator)
        accumulated_mass += decay * (1.0 - accumulated_mass)

        require(accumulated_mass > 0.0, "Prazna istorijska memorija.")

        memory = accumulator / accumulated_mass

        vector = a * current + 1j * memory
        norm_squared = a * a * K + float(memory @ memory)
        norm = math.sqrt(norm_squared)

        require(norm > 0.0, "Prazno ulazno stanje.")
        states[t] = vector / norm

        if derivatives:
            memory_derivative = (
                accumulator_derivative
                - memory * mass_derivative
            ) / accumulated_mass

            direct_gradient[t] = da * (
                current / norm
                - vector * (a * K / (norm ** 3))
            )

            decay_gradient[t] = db * (
                1j * memory_derivative / norm
                - vector * (
                    float(memory @ memory_derivative) / (norm ** 3)
                )
            )

    return states, direct_gradient, decay_gradient


def normalization(states, k=K):
    states = np.atleast_2d(states)
    n = states.shape[1]

    require(1 <= k < n, "Neispravna velicina kombinacije.")

    c1 = math.comb(n - 1, k - 1)
    c2 = math.comb(n - 2, k - 2) if k >= 2 else 0

    result = (
        (c1 - c2) * np.sum(np.abs(states) ** 2, axis=1)
        + c2 * np.abs(np.sum(states, axis=1)) ** 2
    )

    require(
        np.all(np.isfinite(result)) and np.all(result > 0.0),
        "Neispravna normalizacija.",
    )
    return result


def graph_from_parameters(parameters):
    logits = np.empty(EDGE_COUNT, dtype=np.float64)
    logits[:-1] = parameters[:GRAPH_PARAMETERS]
    logits[-1] = 0.0
    logits -= np.max(logits)

    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()

    weights = TOTAL_EDGE_WEIGHT * probabilities

    degrees = np.bincount(
        np.concatenate((EDGE_I, EDGE_J)),
        weights=np.concatenate((weights, weights)),
        minlength=N,
    )

    require(
        np.all(np.isfinite(weights)) and np.all(degrees > 0.0),
        "Graf sadrzi numericki izolovan cvor.",
    )

    laplacian = np.diag(degrees)
    laplacian[EDGE_I, EDGE_J] = -weights
    laplacian[EDGE_J, EDGE_I] = -weights

    return weights, degrees, laplacian


def propagator(laplacian, walk_time):
    values, vectors = np.linalg.eigh(laplacian)
    phases = np.exp(-1j * walk_time * values)
    unitary = (vectors * phases[None, :]) @ vectors.T
    return unitary, values, vectors


def initial_parameters(dataset):
    parameters = np.zeros(PARAMETERS, dtype=np.float64)

    # Pocetak odgovara tezini poslednjeg kola a=1.
    parameters[-2] = 0.0

    requested_decay = 2.0 / 9.0
    fraction = (
        (requested_decay - MEMORY_EPSILON)
        / (1.0 - 2.0 * MEMORY_EPSILON)
    )
    parameters[-1] = math.log(fraction / (1.0 - fraction))

    states = history_states(
        dataset["history"], parameters
    )[0]

    covariance = (states.conj().T @ states).real / len(states)
    diagonal = np.diag(covariance)

    distances = (
        diagonal[EDGE_I] + diagonal[EDGE_J]
        - 2.0 * covariance[EDGE_I, EDGE_J]
    )
    distances = np.maximum(distances, 0.0)

    scale = max(float(np.mean(distances)), 1e-12)
    logits = -distances / scale
    logits -= logits[-1]

    parameters[:GRAPH_PARAMETERS] = logits[:-1]
    return parameters


def positive_log_ratio(current, reference):
    """
    Stabilno racunanje log(current/reference), posebno blizu 1.
    """
    require(
        np.all(current > 0.0) and np.all(reference > 0.0),
        "Neispravan argument logaritamskog odnosa.",
    )

    relative = (current - reference) / reference
    close = np.abs(relative) < 0.5

    result = np.empty_like(current)
    result[close] = np.log1p(relative[close])
    result[~close] = (
        np.log(current[~close]) - np.log(reference[~close])
    )
    return result


def graph_objective(
    parameters, dataset, walk_time, regularization, anchor=None
):
    weights, degrees, laplacian = graph_from_parameters(parameters)
    unitary, eigenvalues, eigenvectors = propagator(
        laplacian, walk_time
    )

    inputs, direct_derivative, decay_derivative = history_states(
        dataset["history"], parameters, derivatives=True
    )

    targets = dataset["targets"]
    count = len(inputs)

    evolved = inputs @ unitary.T
    amplitudes = np.sum(evolved * targets, axis=1)
    observed_mass = np.abs(amplitudes) ** 2
    denominator = normalization(inputs)

    require(
        np.all(np.isfinite(observed_mass))
        and np.all(observed_mass > 0.0),
        "Nulta ili neispravna verovatnoca trening kombinacije.",
    )

    # Kompleksni izvod gubitka po propagiranom stanju.
    output_adjoint = (
        -targets / amplitudes.conj()[:, None]
    ) / count

    unitary_adjoint = output_adjoint.T @ inputs.conj()
    transformed_adjoint = (
        eigenvectors.T @ unitary_adjoint @ eigenvectors
    )

    difference = (
        eigenvalues[:, None] - eigenvalues[None, :]
    )
    midpoint = 0.5 * (
        eigenvalues[:, None] + eigenvalues[None, :]
    )

    divided_difference = (
        -1j * walk_time
        * np.exp(-1j * walk_time * midpoint)
        * np.sinc(walk_time * difference / (2.0 * np.pi))
    )

    laplacian_gradient = 2.0 * np.real(
        eigenvectors
        @ (divided_difference.conj() * transformed_adjoint)
        @ eigenvectors.T
    )
    laplacian_gradient = 0.5 * (
        laplacian_gradient + laplacian_gradient.T
    )

    diagonal_gradient = np.diag(laplacian_gradient)
    weight_gradient = (
        diagonal_gradient[EDGE_I]
        + diagonal_gradient[EDGE_J]
        - 2.0 * laplacian_gradient[EDGE_I, EDGE_J]
    )

    covariance = (inputs.conj().T @ inputs).real / count
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.diag(covariance)

    edge_distance = (
        diagonal[EDGE_I] + diagonal[EDGE_J]
        - 2.0 * covariance[EDGE_I, EDGE_J]
    )

    energy = float(weights @ edge_distance)
    degree_barrier = -float(np.mean(np.log(degrees)))

    concentration_scale = EDGE_COUNT / (TOTAL_EDGE_WEIGHT ** 2)
    concentration = (
        0.5 * concentration_scale * float(weights @ weights)
    )
    penalty = energy + degree_barrier + concentration

    weight_gradient += regularization * (
        edge_distance
        - (1.0 / N) * (
            1.0 / degrees[EDGE_I] + 1.0 / degrees[EDGE_J]
        )
        + concentration_scale * weights
    )

    probabilities = weights / TOTAL_EDGE_WEIGHT
    average_gradient = float(probabilities @ weight_gradient)

    logit_gradient = weights * (
        weight_gradient - average_gradient
    )

    # Normalizacija zavisi od naucenih parametara memorije.
    # Zato njen izvod mora biti ukljucen.
    sums = np.sum(inputs, axis=1)

    normalization_adjoint = (
        (C1 - C2) * inputs + C2 * sums[:, None]
    ) / denominator[:, None] / count

    input_adjoint = (
        output_adjoint @ unitary.conj()
        + normalization_adjoint
        + regularization * (inputs @ laplacian) / count
    )

    gradient = np.empty(PARAMETERS, dtype=np.float64)
    gradient[:GRAPH_PARAMETERS] = logit_gradient[:-1]

    gradient[-2] = 2.0 * float(np.real(np.sum(
        input_adjoint.conj() * direct_derivative
    )))
    gradient[-1] = 2.0 * float(np.real(np.sum(
        input_adjoint.conj() * decay_derivative
    )))

    if anchor is None:
        value = float(np.mean(
            np.log(denominator) - np.log(observed_mass)
        )) + regularization * penalty
    else:
        value = float(np.mean(
            positive_log_ratio(denominator, anchor["denominator"])
            - positive_log_ratio(observed_mass, anchor["mass"])
        )) + regularization * (penalty - anchor["penalty"])

    require(
        np.isfinite(value) and np.all(np.isfinite(gradient)),
        "Nefinitan cilj ili gradijent.",
    )

    reference = {
        "mass": observed_mass.copy(),
        "denominator": denominator.copy(),
        "penalty": penalty,
    }

    return value, gradient, reference


class StationaryPointFound(Exception):
    pass


def fit_graph(dataset, walk_time, regularization, starting=None):
    theta0 = (
        initial_parameters(dataset)
        if starting is None
        else np.asarray(starting, dtype=np.float64).copy()
    )

    anchor = graph_objective(
        theta0, dataset, walk_time, regularization
    )[2]

    cache = {}
    best = {}
    accepted = {}

    iterations = 0
    last_report = time.monotonic()

    def objective(eta):
        if "eta" in cache and np.array_equal(eta, cache["eta"]):
            return cache["value"], cache["gradient"]

        theta = PARAMETER_SCALE * np.asarray(eta)

        value, raw_gradient, _ = graph_objective(
            theta, dataset, walk_time, regularization, anchor
        )
        norm = float(np.max(np.abs(raw_gradient)))

        cache.clear()
        cache.update(
            eta=np.asarray(eta).copy(),
            theta=theta.copy(),
            value=float(value),
            gradient=PARAMETER_SCALE * raw_gradient,
            norm=norm,
        )

        if (
            not best
            or value < best["value"]
            or (
                value <= best["value"] + 1e-12
                and norm < best["norm"]
            )
        ):
            best.clear()
            best.update(
                eta=np.asarray(eta).copy(),
                theta=theta.copy(),
                value=float(value),
                norm=norm,
            )

        return cache["value"], cache["gradient"]

    def progress(eta):
        nonlocal iterations, last_report
        iterations += 1
        objective(eta)

        if cache["norm"] <= GRADIENT_TOLERANCE:
            accepted["theta"] = cache["theta"].copy()
            raise StationaryPointFound()

        now = time.monotonic()

        if now - last_report >= 60.0:
            print(
                f"Ucenje: iteracija {iterations}"
                f" | gradijent: {cache['norm']:.6g}",
                flush=True,
            )
            last_report = now

    current = theta0 / PARAMETER_SCALE
    objective(current)
    message = ""

    if cache["norm"] <= GRADIENT_TOLERANCE:
        return cache["theta"].copy()

    stages = (
        ("BFGS", 1600),
        ("CG", 600),
    )

    for method, limit in stages:
        if accepted:
            break

        try:
            result = minimize(
                objective,
                current,
                method=method,
                jac=True,
                callback=progress,
                options={
                    "maxiter": limit,
                    "gtol": 0.5 * GRADIENT_TOLERANCE,
                    "disp": False,
                },
            )

            objective(result.x)
            message = str(result.message)

            if cache["norm"] <= GRADIENT_TOLERANCE:
                accepted["theta"] = cache["theta"].copy()

        except StationaryPointFound:
            message = "Dostignut prag izvornog gradijenta."

        current = best["eta"].copy()

    parameters = (
        accepted["theta"] if accepted else best["theta"]
    )

    _, gradient, _ = graph_objective(
        parameters, dataset, walk_time, regularization, anchor
    )
    norm = float(np.max(np.abs(gradient)))

    require(
        norm <= GRADIENT_TOLERANCE,
        f"Optimizacija nije konvergirala; gradijent={norm:.6g};"
        f" iteracije={iterations}; {message}",
    )

    return parameters.copy()


def evaluate(parameters, dataset, start, end, walk_time):
    states = history_states(
        dataset["history"][:end], parameters
    )[0][start:end]

    targets = dataset["targets"][start:end]

    _, _, laplacian = graph_from_parameters(parameters)
    unitary = propagator(laplacian, walk_time)[0]

    evolved = states @ unitary.T
    amplitudes = np.sum(evolved * targets, axis=1)
    mass = np.abs(amplitudes) ** 2

    require(
        np.all(np.isfinite(mass)) and np.all(mass > 0.0),
        "Neispravna evaluaciona verovatnoca.",
    )

    return float(np.mean(
        np.log(normalization(states)) - np.log(mass)
    ))


def allowed_next(combos, maximum_run=MAX_RUN):
    differences = np.diff(combos, axis=1)
    allowed = np.ones(len(combos), dtype=bool)

    for start in range(max(0, combos.shape[1] - maximum_run)):
        allowed &= ~np.all(
            differences[:, start:start + maximum_run] == 1,
            axis=1,
        )

    return allowed


def exhaustive_next(signal, k=K, maximum_run=MAX_RUN, announce=True):
    n = len(signal)
    total = math.comb(n, k)
    denominator = float(normalization(signal, k)[0])

    if announce:
        print(f"Pretraga NEXT: svih {total} kombinacija.", flush=True)

    iterator = combinations(range(1, n + 1), k)

    best = None
    best_mass = -1.0
    total_mass = 0.0
    allowed_mass = 0.0
    seen = 0

    while True:
        block = list(islice(iterator, SEARCH_BATCH))
        if not block:
            break

        combos = np.asarray(block, dtype=np.int64)
        amplitudes = np.sum(signal[combos - 1], axis=1)
        masses = np.abs(amplitudes) ** 2

        require(
            np.all(np.isfinite(masses)),
            "Neispravna masa u NEXT pretrazi.",
        )

        total_mass += float(np.sum(masses))
        permitted = allowed_next(combos, maximum_run)

        if np.any(permitted):
            allowed_mass += float(np.sum(masses[permitted]))
            masked = np.where(permitted, masses, -1.0)
            index = int(np.argmax(masked))

            if masked[index] > best_mass:
                best_mass = float(masked[index])
                best = combos[index].copy()

        seen += len(combos)

    require(seen == total, "Pretraga nije kompletna.")
    require(best is not None, "Nema dozvoljene kombinacije.")
    require(
        math.isclose(
            total_mass / denominator, 1.0,
            rel_tol=0.0, abs_tol=5e-9,
        ),
        "Zbir zajednickih verovatnoca nije jedan.",
    )
    require(
        0.0 < allowed_mass <= total_mass * (1.0 + 1e-10),
        "Neispravna masa dozvoljenog domena.",
    )

    checked = abs(sum(
        complex(signal[int(value) - 1]) for value in best
    )) ** 2

    require(
        math.isclose(
            checked, best_mass, rel_tol=1e-11, abs_tol=1e-12
        ),
        "Skalarna provera maksimuma nije prosla.",
    )
    require(
        len(set(best.tolist())) == k
        and np.all((best >= 1) & (best <= n))
        and allowed_next(best[None, :], maximum_run)[0],
        "NEXT nije legalna kombinacija.",
    )

    return best, best_mass / denominator, allowed_mass / denominator


def qiskit_state(input_state, unitary):
    padded = np.zeros(DIM, dtype=np.complex128)
    padded[:N] = input_state

    embedded = np.eye(DIM, dtype=np.complex128)
    embedded[:N, :N] = unitary

    circuit = QuantumCircuit(QUBITS)
    circuit.append(StatePreparation(padded), range(QUBITS))
    circuit.append(UnitaryGate(embedded), range(QUBITS))

    return (
        np.asarray(Statevector.from_instruction(circuit).data),
        circuit,
    )


def internal_checks():
    history = np.asarray(
        [
            [1, 3, 8, 12, 19, 27, 38],
            [2, 6, 9, 15, 21, 30, 39],
            [4, 7, 11, 16, 23, 29, 35],
            [1, 5, 13, 18, 25, 32, 37],
            [3, 8, 14, 20, 24, 31, 36],
        ],
        dtype=np.int64,
    )

    dataset = make_dataset(history)
    parameters = initial_parameters(dataset)

    states, dx_a, dx_b = history_states(
        dataset["history"], parameters, derivatives=True
    )

    require(
        np.allclose(np.sum(np.abs(states) ** 2, axis=1), 1.0, atol=1e-12),
        "Istorijska stanja nisu normalizovana.",
    )

    epsilon = 1e-6

    # Posebno proveriti izvode naucene memorije.
    for parameter_index, expected in (
        (-2, dx_a),
        (-1, dx_b),
    ):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[parameter_index] += epsilon
        minus[parameter_index] -= epsilon

        numerical = (
            history_states(dataset["history"], plus)[0]
            - history_states(dataset["history"], minus)[0]
        ) / (2.0 * epsilon)

        require(
            np.allclose(numerical, expected, rtol=3e-5, atol=3e-7),
            "Izvod naucene memorije nije tacan.",
        )

    weights, degrees, laplacian = graph_from_parameters(parameters)
    unitary, _, _ = propagator(laplacian, 1.7)
    evolved = states @ unitary.T

    require(
        np.all(weights >= 0.0)
        and np.all(degrees > 0.0)
        and np.allclose(laplacian, laplacian.T, atol=1e-12)
        and np.allclose(laplacian.sum(axis=1), 0.0, atol=1e-12)
        and np.isclose(np.trace(laplacian), N, atol=1e-12),
        "Neispravan Laplacijan.",
    )
    require(
        np.min(np.linalg.eigvalsh(laplacian)) >= -1e-11,
        "Laplacijan nije pozitivno semidefinitan.",
    )
    require(
        np.allclose(
            unitary.conj().T @ unitary, np.eye(N), atol=2e-11
        )
        and np.allclose(
            unitary @ np.ones(N), np.ones(N), atol=2e-11
        ),
        "Neispravan unitarni propagator.",
    )
    require(
        np.allclose(
            normalization(states), normalization(evolved),
            rtol=1e-11, atol=1e-8,
        ),
        "Propagacija menja normalizaciju.",
    )

    covariance = (states.conj().T @ states).real / len(states)
    diagonal = np.diag(covariance)
    edge_energy = float(weights @ (
        diagonal[EDGE_I] + diagonal[EDGE_J]
        - 2.0 * covariance[EDGE_I, EDGE_J]
    ))
    direct_energy = float(np.mean(np.real(np.einsum(
        "bi,ij,bj->b",
        states.conj(), laplacian, states,
        optimize=True,
    ))))

    require(
        np.isclose(edge_energy, direct_energy, atol=1e-12),
        "Dirichletova energija nije saglasna sa ivicnim izrazom.",
    )

    measured, circuit = qiskit_state(states[1], unitary)

    require(
        np.allclose(measured[:N], evolved[1], atol=2e-11)
        and np.allclose(measured[N:], 0.0, atol=2e-11),
        "Qiskit i lokalna simulacija nisu saglasni.",
    )

    target = history[2]
    target_state = np.zeros(DIM, dtype=np.complex128)
    target_state[target - 1] = 1.0 / math.sqrt(K)

    circuit.append(
        StatePreparation(target_state).inverse(),
        range(QUBITS),
    )
    overlap = Statevector.from_instruction(circuit).data[0]

    require(
        np.allclose(
            overlap,
            np.sum(evolved[1, target - 1]) / math.sqrt(K),
            atol=2e-11,
        ),
        "Qiskit provera kolektivne amplitude nije prosla.",
    )

    # Gradijent grafa, memorije i normalizacije.
    for regularization in (0.0, 0.1):
        raw_value, raw_gradient, anchor = graph_objective(
            parameters, dataset, 1.7, regularization
        )

        centered_value, centered_gradient, _ = graph_objective(
            parameters, dataset, 1.7, regularization, anchor
        )

        require(
            abs(centered_value) < 1e-12
            and np.allclose(
                raw_gradient, centered_gradient, atol=1e-12
            ),
            "Centriranje menja gradijent.",
        )

        directions = []

        for shift in (0.2, 1.1):
            direction = np.sin(
                np.arange(PARAMETERS) * 0.71 + shift
            )
            direction /= np.linalg.norm(direction)
            directions.append(direction)

        for index in (-2, -1):
            direction = np.zeros(PARAMETERS)
            direction[index] = 1.0
            directions.append(direction)

        for direction in directions:
            numerical = (
                graph_objective(
                    parameters + epsilon * direction,
                    dataset, 1.7, regularization, anchor,
                )[0]
                - graph_objective(
                    parameters - epsilon * direction,
                    dataset, 1.7, regularization, anchor,
                )[0]
            ) / (2.0 * epsilon)

            require(
                math.isclose(
                    numerical,
                    float(raw_gradient @ direction),
                    rel_tol=5e-5,
                    abs_tol=5e-6,
                ),
                "Analiticki gradijent nije prosao proveru.",
            )

        shifted = parameters.copy()
        shifted[-2] += 0.13

        raw_shifted = graph_objective(
            shifted, dataset, 1.7, regularization
        )[0]
        centered_shifted = graph_objective(
            shifted, dataset, 1.7, regularization, anchor
        )[0]

        require(
            np.isclose(
                raw_shifted - raw_value,
                centered_shifted,
                atol=1e-11,
            ),
            "Centriranje menja ciljnu funkciju.",
        )

    # Ponovljene svojstvene vrednosti.
    equal = parameters.copy()
    equal[:GRAPH_PARAMETERS] = 0.0
    _, gradient, anchor = graph_objective(
        equal, dataset, 1.7, 0.1
    )

    direction = np.cos(np.arange(PARAMETERS) * 0.47)
    direction /= np.linalg.norm(direction)

    numerical = (
        graph_objective(
            equal + epsilon * direction, dataset, 1.7, 0.1, anchor
        )[0]
        - graph_objective(
            equal - epsilon * direction, dataset, 1.7, 0.1, anchor
        )[0]
    ) / (2.0 * epsilon)

    require(
        math.isclose(
            numerical, float(gradient @ direction),
            rel_tol=5e-5, abs_tol=5e-6,
        ),
        "Nestabilan gradijent kod ponovljenih svojstvenih vrednosti.",
    )

    # Normalizacija i maksimum na malom domenu.
    n, k = 9, 4
    index = np.arange(n, dtype=np.float64)
    signal = (
        np.cos(0.37 * index + 0.2)
        + 1j * np.sin(0.63 * index + 0.4)
    )
    signal /= np.linalg.norm(signal)

    combos = np.asarray(
        list(combinations(range(1, n + 1), k)), dtype=np.int64
    )
    masses = np.abs(np.sum(signal[combos - 1], axis=1)) ** 2
    denominator = float(normalization(signal, k)[0])

    require(
        np.isclose(masses.sum(), denominator, atol=1e-11),
        "Neispravna analiticka normalizacija.",
    )

    best, probability, permitted_probability = exhaustive_next(
        signal, k=k, maximum_run=2, announce=False
    )
    permitted = allowed_next(combos, 2)
    expected = int(np.argmax(np.where(permitted, masses, -1.0)))

    require(
        np.array_equal(best, combos[expected])
        and np.isclose(
            probability, masses[expected] / denominator, atol=1e-12
        )
        and np.isclose(
            permitted_probability,
            masses[permitted].sum() / denominator,
            atol=1e-12,
        ),
        "Provera NEXT maksimuma nije prosla.",
    )

    changed = signal * np.exp(0.73j)
    require(
        np.allclose(
            masses,
            np.abs(np.sum(changed[combos - 1], axis=1)) ** 2,
            atol=1e-12,
        ),
        "Globalna faza menja raspodelu.",
    )

    require(
        np.array_equal(
            history_states(dataset["history"][:3], parameters)[0],
            states[:3],
        ),
        "Istorijska memorija koristi buduce redove.",
    )

    runs = np.asarray(
        [
            [1, 2, 3, 4, 8, 12, 20],
            [1, 2, 3, 4, 5, 12, 20],
            [1, 8, 9, 10, 11, 12, 20],
            [1, 8, 15, 16, 17, 18, 19],
        ],
        dtype=np.int64,
    )
    require(
        np.array_equal(
            allowed_next(runs),
            np.asarray([True, False, False, False]),
        ),
        "Neispravno pravilo uzastopnih brojeva.",
    )


def process_csv(path):
    print()
    print(f"CSV: {path.name}", flush=True)

    draws = read_csv(path)
    full_dataset = make_dataset(draws)
    count = len(draws) - 1

    train_end = int(TRAIN_FRACTION * count)
    validation_end = int(VALIDATION_END * count)

    require(
        0 < train_end < validation_end < count,
        "Neispravna hronoloska podela.",
    )

    training = make_dataset(draws, train_end)
    selected = None

    for walk_time in WALK_TIMES:
        previous = None

        for regularization in REGULARIZATIONS:
            print(
                f"Ucenje: t={walk_time:g} | reg={regularization:g}.",
                flush=True,
            )

            try:
                parameters = fit_graph(
                    training, walk_time, regularization, previous
                )
                previous = parameters.copy()

                score = evaluate(
                    parameters, full_dataset,
                    train_end, validation_end, walk_time,
                )

                if selected is None or score < selected["score"]:
                    selected = {
                        "score": score,
                        "time": walk_time,
                        "reg": regularization,
                        "parameters": parameters.copy(),
                    }

            except (RuntimeError, FloatingPointError) as exc:
                print(
                    f"PROVERA: kandidat t={walk_time:g},"
                    f" reg={regularization:g} odbacen: {exc}",
                    flush=True,
                )

    require(selected is not None, "Nijedan kandidat nije prosao provere.")

    walk_time = selected["time"]
    regularization = selected["reg"]

    print(
        f"Validacija NLL: {selected['score']:.6f}"
        f" | t: {walk_time:g} | reg: {regularization:g}",
        flush=True,
    )
    print("Ucenje modela za test.", flush=True)

    test_parameters = fit_graph(
        make_dataset(draws, validation_end),
        walk_time,
        regularization,
        selected["parameters"],
    )

    test_score = evaluate(
        test_parameters, full_dataset,
        validation_end, count, walk_time,
    )

    print(f"Test NLL: {test_score:.6f}", flush=True)
    print("Zavrsno ucenje: svi istorijski prelazi.", flush=True)

    final_parameters = fit_graph(
        full_dataset, walk_time, regularization, test_parameters
    )

    # NEXT koristi i poslednje poznato kolo.
    all_states = history_states(
        indicators(draws), final_parameters
    )[0]
    input_state = all_states[-1]

    weights, _, laplacian = graph_from_parameters(final_parameters)
    unitary = propagator(laplacian, walk_time)[0]
    signal = unitary @ input_state

    require(
        np.isclose(np.vdot(signal, signal).real, 1.0, atol=2e-11),
        "Zavrsno stanje nije normalizovano.",
    )

    direct_weight, decay, _, _ = memory_parameters(final_parameters)
    print(
        f"Ucena memorija: tezina poslednjeg kola={direct_weight:.6f}"
        f" | korak memorije={decay:.6f}",
        flush=True,
    )

    next_combo, next_probability, allowed_probability = exhaustive_next(
        signal
    )

    print(f"NEXT: {next_combo.tolist()}", flush=True)

    if np.array_equal(next_combo, draws[-1]):
        print(
            "PROVERA: NEXT se poklapa sa poslednjim kolom;"
            " rezultat nije naknadno promenjen.",
            flush=True,
        )

    if DIAGNOSTICS:
        print(f"NEXT verovatnoca: {next_probability:.12g}")
        print(f"Masa dozvoljenog domena: {allowed_probability:.12g}")

    return {
        "path": path,
        "next": tuple(next_combo.tolist()),
        "digest": hashlib.sha256(draws.tobytes()).hexdigest(),
        "weights": weights,
        "signal": signal,
        "input": input_state,
    }


def check_repeated_results(results):
    for first, second in combinations(results, 2):
        if first["next"] != second["next"]:
            continue

        if first["digest"] == second["digest"]:
            reason = "CSV sadrzaj je identican."
        else:
            graph_difference = float(np.linalg.norm(
                first["weights"] - second["weights"]
            ))

            # Poredi matrice koje odredjuju verovatnoce.
            # Globalna kompleksna faza ne sme biti lazna razlika.
            def probability_matrix(signal):
                denominator = float(normalization(signal)[0])
                return np.real(
                    np.outer(signal, signal.conj())
                ) / denominator

            probability_difference = float(np.linalg.norm(
                probability_matrix(first["signal"])
                - probability_matrix(second["signal"])
            ))

            input_overlap = float(
                abs(np.vdot(first["input"], second["input"])) ** 2
            )

            reason = (
                "CSV sadrzaji su razliciti;"
                f" razlika grafova={graph_difference:.6g};"
                f" razlika matrica verovatnoce={probability_difference:.6g};"
                f" preklapanje ulaznih stanja={input_overlap:.6f}."
            )

        print(
            f"PROVERA: isti NEXT za {first['path'].name}"
            f" i {second['path'].name}. {reason}",
            flush=True,
        )


def main():
    print("Quantum Dirichlet Laplacian V2: uceni graf i ucena memorija.")
    print("Lokalne kvantne amplitude; tacan NLL cele kombinacije.")
    print("NEXT: najvise cetiri uzastopna broja.")
    print("Pokretanje internih provera.", flush=True)

    try:
        internal_checks()
    except Exception as exc:
        print(f"STOP: interna provera nije prosla: {exc}", flush=True)
        return 1

    print("Interne provere: prosle.", flush=True)

    results = []
    failures = 0

    for path in FILES:
        try:
            results.append(process_csv(path))
        except KeyboardInterrupt:
            print()
            print("Obrada prekinuta korisnickim zahtevom.", flush=True)
            return 130
        except Exception as exc:
            failures += 1
            print(f"STOP za ovaj CSV: {exc}", flush=True)

    check_repeated_results(results)

    if failures:
        print("Obrada nije kompletna: neki CSV-ovi nisu prosli provere.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Quantum Dirichlet Laplacian V2: uceni graf i ucena memorija.
Lokalne kvantne amplitude; tacan NLL cele kombinacije.
NEXT: najvise cetiri uzastopna broja.
Pokretanje internih provera.
Interne provere: prosle.

CSV: loto7_4720_k91.csv
Ucenje: t=1 | reg=0.1.
Ucenje: iteracija 843 | gradijent: 2.58406e-05
Ucenje: t=1 | reg=0.01.
Ucenje: t=3 | reg=0.1.
Ucenje: t=3 | reg=0.01.
Validacija NLL: 16.547586 | t: 1 | reg: 0.1
Ucenje modela za test.
Test NLL: 16.547867
Zavrsno ucenje: svi istorijski prelazi.
Ucena memorija: tezina poslednjeg kola=0.014897 | korak memorije=0.000001
Pretraga NEXT: svih 15380937 kombinacija.
NEXT: [7, x, 10, y, 22, z, 34]

CSV: loto7_4720_k91_loto_2982.csv
Ucenje: t=1 | reg=0.1.
Ucenje: iteracija 1190 | gradijent: 5.44755e-05
Ucenje: t=1 | reg=0.01.
Ucenje: t=3 | reg=0.1.
Ucenje: t=3 | reg=0.01.
Ucenje: iteracija 1435 | gradijent: 1.6122e-05
PROVERA: kandidat t=3, reg=0.01 odbacen: Optimizacija nije konvergirala; gradijent=8.37179e-06; iteracije=2200; Maximum number of iterations has been exceeded.
Validacija NLL: 16.549174 | t: 1 | reg: 0.1
Ucenje modela za test.
Test NLL: 16.547639
Zavrsno ucenje: svi istorijski prelazi.
Ucena memorija: tezina poslednjeg kola=0.004580 | korak memorije=0.000001
Pretraga NEXT: svih 15380937 kombinacija.
NEXT: [8, x, 22, y, 34, z, 38]

CSV: loto7_4720_k91_loto_plus_1738.csv
Ucenje: t=1 | reg=0.1.
Ucenje: t=1 | reg=0.01.
Ucenje: t=3 | reg=0.1.
Ucenje: t=3 | reg=0.01.
Validacija NLL: 16.578165 | t: 3 | reg: 0.1
Ucenje modela za test.
Test NLL: 16.608050
Zavrsno ucenje: svi istorijski prelazi.
Ucena memorija: tezina poslednjeg kola=0.609072 | korak memorije=0.003272
Pretraga NEXT: svih 15380937 kombinacija.
NEXT: [2, x, 15, y, 26, z, 37]
"""



"""
U V1 poslednje kolo ima fiksnu težinu, a brzina slabljenja starije istorije takođe je fiksna. U ispravci će se oba parametra učiti iz istorijskih prelaza. Za numerički zastoj koristim istu funkciju cilja sa oduzetom konstantnom početnom vrednošću, uz isti prag gradijenta — time se smanjuje gubitak preciznosti pri veoma malim poboljšanjima.

V1 je ima fiksnu težinu poslednjeg kola i fiksno slabljenje istorije. V2 uči oba parametra zajedno sa grafom.

Numerički postupak: cilj je centriran radi preciznijeg računanja malih poboljšanja, uz isti prag gradijenta. Ako BFGS zastane, sledi nastavak konjugovanim gradijentom.`
"""
