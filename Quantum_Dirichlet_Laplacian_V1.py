# Quantum_Dirichlet_Laplacian V1



#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Quantum Dirichlet Laplacian V1 — Loto 7/39.

IDEJA
    Uceni neusmereni graf sa 39 cvorova.
    Svaki par brojeva moze imati ucenu tezinu.
    Brojcana blizina ne odredjuje tezinu veze.

    L = D - A
    E(x) = x* L x
    U(t) = exp(-i t L)

    Dirichletova energija ucestvuje u obuci grafa.
    Unitarno kvantno kretanje samo po sebi ne smanjuje energiju.

ZAJEDNICKA RASPODELA
    Za sedmorku S:
        amplitude(S) = sum(psi[j] for j in S)
        P(S) = abs(amplitude(S))**2 / Z

    Tacna normalizacija preko svih k-podskupova:
        c1 = C(n-1, k-1)
        c2 = C(n-2, k-2)

        Z = (c1-c2) * sum(abs(psi)**2)
            + c2 * abs(sum(psi))**2

    Nema odvojenog biranja sedam najjacih pojedinacnih brojeva.
    Interferencija se racuna za celu kombinaciju.

KVANTNI I KLASICNI DEO
    Lokalno se simulira sest-qubitno stanje grafa.
    Stanja 39..63 nisu deo grafa i ne primaju amplitudu.
    Qiskit proverava pripremu stanja, propagaciju i preklapanje
    sa stanjem konkretne sedmorke.

    Zajednicka raspodela sedmorki dobija se analiticki iz amplituda.
    Nije izvrseno fizicko merenje svih ishoda na kvantnom racunaru.
    Obuka i iscrpna pretraga su klasicne.

    Sedmorka se moze zapisati sa sedam sest-bitnih oznaka.
    Ne formira se gust statevector od 2^42 elemenata.

PODACI
    Redovi CSV-a moraju biti od najstarijeg do najnovijeg.
    Sedam celih brojeva po redu, bez hedera.
    Cela raspoloziva istorija koristi se za zavrsno ucenje.
    Istorijski redovi se ne filtriraju.

NEXT
    Najvise cetiri uzastopna broja.
    Nema filtera za zbir, parnost, polovinu opsega ili grupisanje.
    Nema nametanja razlicitih rezultata razlicitim CSV-ovima.
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
PARAMETERS = EDGE_COUNT - 1

# Fiksiranje ukupne tezine uklanja proizvoljnu skalu L.
# Posledica je trace(L) = N, odnosno srednji stepen = 1.
TOTAL_EDGE_WEIGHT = N / 2.0

WALK_TIMES = (1.0, 3.0)
REGULARIZATIONS = (0.1, 0.01)

TRAIN_FRACTION = 0.70
VALIDATION_END = 0.85

MAX_RUN = 4
SEARCH_BATCH = 32768
MAX_ITERATIONS = 1600
GRADIENT_TOLERANCE = 1e-7

PARAMETER_SCALE = math.sqrt(EDGE_COUNT)
DIAGNOSTICS = False


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
    rows = np.arange(len(draws))[:, None]
    result[rows, draws - 1] = 1.0
    return result


def build_history_states(draws):
    """
    Realni deo: poslednje poznato kolo.
    Imaginarni deo: eksponencijalni prosek poznate istorije.

    Za cilj u redu t+1 koristi se stanje iz reda t.
    Nema buducih podataka, mesanja ili slucajnih stanja.
    """
    encoded = indicators(draws)
    states = np.empty(encoded.shape, dtype=np.complex128)

    average = encoded[0].copy()
    alpha = 2.0 / 9.0

    for t, current in enumerate(encoded):
        if t:
            average += alpha * (current - average)

        signal = current + 1j * average
        norm = float(np.linalg.norm(signal))

        require(norm > 0.0, "Prazno ulazno stanje.")
        states[t] = signal / norm

    return states


def normalization(states, k=K):
    states = np.atleast_2d(states)
    n = states.shape[1]

    require(1 <= k < n, "Neispravna velicina kombinacije.")

    c1 = math.comb(n - 1, k - 1)
    c2 = math.comb(n - 2, k - 2) if k >= 2 else 0

    norm_squared = np.sum(np.abs(states) ** 2, axis=1)
    sum_squared = np.abs(np.sum(states, axis=1)) ** 2

    result = (c1 - c2) * norm_squared + c2 * sum_squared

    require(
        np.all(np.isfinite(result)) and np.all(result > 0.0),
        "Neispravna analiticka normalizacija.",
    )
    return result


def prepare_training(states, draws):
    """
    Statistike se racunaju samo iz prosledjenog trening skupa.
    """
    require(len(states) == len(draws), "Nesaglasan trening skup.")

    covariance = (states.conj().T @ states).real / len(states)
    covariance = 0.5 * (covariance + covariance.T)

    diagonal = np.diag(covariance)
    edge_distance = (
        diagonal[EDGE_I]
        + diagonal[EDGE_J]
        - 2.0 * covariance[EDGE_I, EDGE_J]
    )

    require(
        np.min(edge_distance) >= -1e-12,
        "Negativna kvadratna razlika signala.",
    )

    return {
        "states": states,
        "draws": draws,
        "targets": indicators(draws),
        "normalization": normalization(states),
        "covariance": covariance,
        "edge_distance": np.maximum(edge_distance, 0.0),
    }


def initial_parameters(training):
    """
    Pocetni graf izvodi se iz trening signala.
    Nema slucajnih tezina ni zavisnosti od brojcanih razmaka.
    """
    distance = training["edge_distance"]
    scale = max(float(np.mean(distance)), 1e-12)

    logits = -distance / scale
    logits -= logits[-1]

    return logits[:-1].copy()


def graph_from_parameters(parameters):
    """
    Sve tezine su nenegativne; njihov zbir je fiksiran.
    Jedan logit je fiksiran na nulu radi uklanjanja redundanse.
    """
    require(
        np.shape(parameters) == (PARAMETERS,),
        "Neispravan broj parametara grafa.",
    )

    logits = np.empty(EDGE_COUNT, dtype=np.float64)
    logits[:-1] = parameters
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
        np.all(np.isfinite(weights))
        and np.all(degrees > 0.0),
        "Graf sadrzi numericki izolovan cvor.",
    )

    laplacian = np.diag(degrees)
    laplacian[EDGE_I, EDGE_J] = -weights
    laplacian[EDGE_J, EDGE_I] = -weights

    return weights, degrees, laplacian


def propagator(laplacian, walk_time):
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    phase = np.exp(-1j * walk_time * eigenvalues)

    unitary = (
        eigenvectors * phase[None, :]
    ) @ eigenvectors.T

    return unitary, eigenvalues, eigenvectors


def graph_objective(parameters, training, walk_time, regularization):
    weights, degrees, laplacian = graph_from_parameters(parameters)
    unitary, eigenvalues, eigenvectors = propagator(
        laplacian, walk_time
    )

    inputs = training["states"]
    targets = training["targets"]
    count = len(inputs)

    evolved = inputs @ unitary.T
    amplitudes = np.sum(evolved * targets, axis=1)
    observed_mass = np.abs(amplitudes) ** 2

    require(
        np.all(np.isfinite(observed_mass))
        and np.all(observed_mass > 0.0),
        "Nulta ili neispravna verovatnoca trening kombinacije.",
    )

    nll = float(np.mean(
        np.log(training["normalization"]) - np.log(observed_mass)
    ))

    # Kompleksni adjoint:
    # d loss = 2 Re Tr(G* dU).
    unitary_adjoint = -(
        targets.T
        @ (inputs.conj() / amplitudes.conj()[:, None])
    ) / count

    transformed_adjoint = (
        eigenvectors.T @ unitary_adjoint @ eigenvectors
    )

    difference = (
        eigenvalues[:, None] - eigenvalues[None, :]
    )
    midpoint = 0.5 * (
        eigenvalues[:, None] + eigenvalues[None, :]
    )

    # Stabilna podeljena razlika funkcije exp(-i t lambda).
    # Formula vazi i kada su svojstvene vrednosti jednake.
    divided_difference = (
        -1j
        * walk_time
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

    diagonal = np.diag(laplacian_gradient)
    weight_gradient = (
        diagonal[EDGE_I]
        + diagonal[EDGE_J]
        - 2.0 * laplacian_gradient[EDGE_I, EDGE_J]
    )

    # Dirichletova energija trening signala.
    energy = float(weights @ training["edge_distance"])

    # Barijera protiv izolovanih cvorova.
    degree_barrier = -float(np.mean(np.log(degrees)))

    # Kontrola koncentracije svih tezina u nekoliko grana.
    concentration_scale = EDGE_COUNT / (TOTAL_EDGE_WEIGHT ** 2)
    concentration = (
        0.5 * concentration_scale * float(weights @ weights)
    )

    graph_penalty = energy + degree_barrier + concentration

    weight_gradient += regularization * (
        training["edge_distance"]
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

    value = nll + regularization * graph_penalty
    gradient = logit_gradient[:-1]

    require(
        np.isfinite(value) and np.all(np.isfinite(gradient)),
        "Nefinitna ciljna funkcija ili gradijent.",
    )

    return value, gradient


class StationaryPointFound(Exception):
    pass


def fit_graph(training, walk_time, regularization, starting=None):
    theta0 = (
        initial_parameters(training)
        if starting is None
        else np.asarray(starting, dtype=np.float64).copy()
    )

    eta0 = theta0 / PARAMETER_SCALE

    cache = {}
    best = {}
    accepted = {}

    iterations = 0
    last_report = time.monotonic()

    def objective(eta):
        if "eta" in cache and np.array_equal(eta, cache["eta"]):
            return cache["value"], cache["gradient"]

        theta = PARAMETER_SCALE * np.asarray(eta)

        value, raw_gradient = graph_objective(
            theta, training, walk_time, regularization
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
                theta=theta.copy(),
                value=float(value),
                norm=norm,
            )

        return cache["value"], cache["gradient"]

    objective(eta0)

    if cache["norm"] <= GRADIENT_TOLERANCE:
        return cache["theta"].copy()

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
                f"Ucenje grafa: iteracija {iterations}"
                f" | gradijent: {cache['norm']:.6g}",
                flush=True,
            )
            last_report = now

    message = ""

    try:
        result = minimize(
            objective,
            eta0,
            jac=True,
            method="BFGS",
            callback=progress,
            options={
                "maxiter": MAX_ITERATIONS,
                "gtol": (
                    0.5 * PARAMETER_SCALE * GRADIENT_TOLERANCE
                ),
                "disp": False,
            },
        )

        objective(result.x)
        message = str(result.message)

        if cache["norm"] <= GRADIENT_TOLERANCE:
            accepted["theta"] = cache["theta"].copy()

    except StationaryPointFound:
        message = "Dostignut prag izvornog gradijenta."

    parameters = (
        accepted["theta"]
        if "theta" in accepted
        else best["theta"]
    )

    _, gradient = graph_objective(
        parameters, training, walk_time, regularization
    )
    norm = float(np.max(np.abs(gradient)))

    require(
        norm <= GRADIENT_TOLERANCE,
        f"Optimizacija nije konvergirala; gradijent={norm:.6g};"
        f" iteracije={iterations}; {message}",
    )

    return parameters.copy()


def evaluate(parameters, states, draws, walk_time):
    _, _, laplacian = graph_from_parameters(parameters)
    unitary = propagator(laplacian, walk_time)[0]

    evolved = states @ unitary.T
    rows = np.arange(len(draws))[:, None]
    amplitudes = np.sum(evolved[rows, draws - 1], axis=1)
    mass = np.abs(amplitudes) ** 2

    require(
        np.all(np.isfinite(mass)) and np.all(mass > 0.0),
        "Neispravna verovatnoca pri evaluaciji.",
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
            "Nefinitna masa u NEXT pretrazi.",
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
    require(best is not None, "Nema dozvoljene NEXT kombinacije.")

    require(
        math.isclose(
            total_mass / denominator,
            1.0,
            rel_tol=0.0,
            abs_tol=5e-9,
        ),
        "Zbir zajednickih verovatnoca nije jedan.",
    )

    require(
        0.0 < allowed_mass <= total_mass * (1.0 + 1e-10),
        "Neispravna masa dozvoljenog domena.",
    )

    checked_amplitude = sum(
        complex(signal[int(value) - 1]) for value in best
    )
    checked_mass = abs(checked_amplitude) ** 2

    require(
        math.isclose(
            checked_mass, best_mass,
            rel_tol=1e-11, abs_tol=1e-12,
        ),
        "Skalarna provera NEXT maksimuma nije prosla.",
    )

    require(
        len(set(best.tolist())) == k
        and np.all((best >= 1) & (best <= n))
        and bool(allowed_next(best[None, :], maximum_run)[0]),
        "NEXT nije legalna kombinacija.",
    )

    return best, best_mass / denominator, allowed_mass / denominator


def qiskit_evolved_state(input_state, unitary):
    prepared = np.zeros(DIM, dtype=np.complex128)
    prepared[:N] = input_state

    embedded = np.eye(DIM, dtype=np.complex128)
    embedded[:N, :N] = unitary

    circuit = QuantumCircuit(QUBITS)
    circuit.append(StatePreparation(prepared), range(QUBITS))
    circuit.append(UnitaryGate(embedded), range(QUBITS))

    state = np.asarray(Statevector.from_instruction(circuit).data)
    return state, circuit


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

    states = build_history_states(history)
    training = prepare_training(states[:-1], history[1:])
    parameters = initial_parameters(training)

    weights, degrees, laplacian = graph_from_parameters(parameters)

    require(
        np.all(weights >= 0.0)
        and np.all(degrees > 0.0),
        "Neispravne tezine grafa.",
    )
    require(
        np.allclose(laplacian, laplacian.T, atol=1e-12)
        and np.allclose(laplacian.sum(axis=1), 0.0, atol=1e-12)
        and np.isclose(np.trace(laplacian), N, atol=1e-12),
        "Matrica nije odgovarajuci Laplacijan.",
    )
    require(
        np.min(np.linalg.eigvalsh(laplacian)) >= -1e-11,
        "Laplacijan nije pozitivno semidefinitan.",
    )

    # Matricni i ivicni izraz Dirichletove energije.
    direct_energy = float(np.mean(np.real(np.einsum(
        "bi,ij,bj->b",
        states[:-1].conj(),
        laplacian,
        states[:-1],
        optimize=True,
    ))))
    edge_energy = float(weights @ training["edge_distance"])

    require(
        np.isclose(direct_energy, edge_energy, atol=1e-12),
        "Dva izraza Dirichletove energije nisu saglasna.",
    )

    walk_time = 1.7
    unitary, _, _ = propagator(laplacian, walk_time)

    require(
        np.allclose(
            unitary.conj().T @ unitary,
            np.eye(N),
            atol=2e-11,
        ),
        "Propagator nije unitaran.",
    )
    require(
        np.allclose(unitary @ np.ones(N), np.ones(N), atol=2e-11),
        "Propagator ne cuva konstantni mod.",
    )

    evolved = states @ unitary.T

    require(
        np.allclose(
            normalization(states),
            normalization(evolved),
            rtol=1e-11,
            atol=1e-8,
        ),
        "Propagacija menja analiticku normalizaciju.",
    )

    # Qiskit provera lokalnog kvantnog kretanja.
    qstate, circuit = qiskit_evolved_state(states[1], unitary)

    require(
        np.allclose(qstate[:N], evolved[1], atol=2e-11)
        and np.allclose(qstate[N:], 0.0, atol=2e-11),
        "Qiskit i lokalna simulacija nisu saglasni.",
    )

    # Preklapanje sa stanjem konkretne sedmorke.
    target = history[2]
    target_state = np.zeros(DIM, dtype=np.complex128)
    target_state[target - 1] = 1.0 / math.sqrt(K)

    circuit.append(
        StatePreparation(target_state).inverse(),
        range(QUBITS),
    )
    overlap = Statevector.from_instruction(circuit).data[0]
    expected_overlap = np.sum(evolved[1, target - 1]) / math.sqrt(K)

    require(
        np.allclose(overlap, expected_overlap, atol=2e-11),
        "Qiskit provera kolektivne amplitude nije prosla.",
    )

    # Analiticki gradijent kompletnog cilja.
    epsilon = 1e-6

    for regularization in (0.0, 0.1):
        _, gradient = graph_objective(
            parameters, training, walk_time, regularization
        )

        for shift in (0.2, 1.1, 2.3):
            direction = np.sin(
                np.arange(PARAMETERS) * 0.71 + shift
            )
            direction /= np.linalg.norm(direction)

            numerical = (
                graph_objective(
                    parameters + epsilon * direction,
                    training, walk_time, regularization,
                )[0]
                - graph_objective(
                    parameters - epsilon * direction,
                    training, walk_time, regularization,
                )[0]
            ) / (2.0 * epsilon)

            analytical = float(gradient @ direction)

            require(
                math.isclose(
                    numerical,
                    analytical,
                    rel_tol=5e-5,
                    abs_tol=5e-6,
                ),
                "Analiticki gradijent nije prosao proveru.",
            )

    # Provera i pri ponovljenim svojstvenim vrednostima:
    # kompletan graf sa jednakim tezinama.
    equal_parameters = np.zeros(PARAMETERS)
    _, gradient = graph_objective(
        equal_parameters, training, walk_time, 0.1
    )

    direction = np.cos(np.arange(PARAMETERS) * 0.47)
    direction /= np.linalg.norm(direction)

    numerical = (
        graph_objective(
            equal_parameters + epsilon * direction,
            training, walk_time, 0.1,
        )[0]
        - graph_objective(
            equal_parameters - epsilon * direction,
            training, walk_time, 0.1,
        )[0]
    ) / (2.0 * epsilon)

    require(
        math.isclose(
            numerical,
            float(gradient @ direction),
            rel_tol=5e-5,
            abs_tol=5e-6,
        ),
        "Gradijent nije stabilan kod ponovljenih svojstvenih vrednosti.",
    )

    # Tacna normalizacija i maksimum na malom domenu.
    small_n, small_k = 9, 4
    index = np.arange(small_n, dtype=np.float64)

    signal = (
        np.cos(0.37 * index + 0.2)
        + 1j * np.sin(0.63 * index + 0.4)
    )
    signal /= np.linalg.norm(signal)

    combos = np.asarray(
        list(combinations(range(1, small_n + 1), small_k)),
        dtype=np.int64,
    )
    masses = np.abs(np.sum(signal[combos - 1], axis=1)) ** 2
    denominator = float(normalization(signal, small_k)[0])

    require(
        np.isclose(masses.sum(), denominator, atol=1e-11),
        "Analiticka normalizacija nije tacna.",
    )

    best, probability, allowed_probability = exhaustive_next(
        signal,
        k=small_k,
        maximum_run=2,
        announce=False,
    )

    permitted = allowed_next(combos, 2)
    expected_index = int(np.argmax(
        np.where(permitted, masses, -1.0)
    ))

    require(
        np.array_equal(best, combos[expected_index]),
        "Iscrpni NEXT maksimum nije tacan.",
    )
    require(
        np.isclose(
            probability,
            masses[expected_index] / denominator,
            atol=1e-12,
        )
        and np.isclose(
            allowed_probability,
            masses[permitted].sum() / denominator,
            atol=1e-12,
        ),
        "Neispravne zajednicke verovatnoce.",
    )

    # Kompleksna faza celog stanja ne sme menjati raspodelu.
    changed = signal * np.exp(0.73j)
    changed_masses = np.abs(
        np.sum(changed[combos - 1], axis=1)
    ) ** 2

    require(
        np.allclose(masses, changed_masses, atol=1e-12),
        "Globalna faza menja verovatnoce.",
    )

    # Numericke oznake nisu koordinatna udaljenost grafa.
    permutation = np.arange(N)[::-1]
    inverse_permutation = np.argsort(permutation)

    changed_laplacian = laplacian[
        np.ix_(permutation, permutation)
    ]
    changed_unitary = propagator(changed_laplacian, walk_time)[0]

    changed_state = changed_unitary @ states[1, permutation]
    changed_target = inverse_permutation[target - 1]

    require(
        np.isclose(
            abs(np.sum(changed_state[changed_target])) ** 2,
            abs(np.sum(evolved[1, target - 1])) ** 2,
            atol=1e-11,
        ),
        "Model zavisi od brojcanog rasporeda oznaka grafa.",
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
        "Pravilo uzastopnih brojeva nije tacno.",
    )
    require(
        np.array_equal(
            build_history_states(history[:3]), states[:3]
        ),
        "Ulazna stanja koriste buduce redove.",
    )


def process_csv(path):
    print()
    print(f"CSV: {path.name}", flush=True)

    draws = read_csv(path)
    history_states = build_history_states(draws)

    inputs = history_states[:-1]
    targets = draws[1:]
    count = len(targets)

    train_end = int(TRAIN_FRACTION * count)
    validation_end = int(VALIDATION_END * count)

    require(
        0 < train_end < validation_end < count,
        "Neispravna hronoloska podela.",
    )

    training = prepare_training(
        inputs[:train_end], targets[:train_end]
    )

    selected = None

    for walk_time in WALK_TIMES:
        previous = None

        for regularization in REGULARIZATIONS:
            print(
                f"Ucenje grafa: t={walk_time:g}"
                f" | reg={regularization:g}.",
                flush=True,
            )

            try:
                parameters = fit_graph(
                    training,
                    walk_time,
                    regularization,
                    previous,
                )
                previous = parameters.copy()

                score = evaluate(
                    parameters,
                    inputs[train_end:validation_end],
                    targets[train_end:validation_end],
                    walk_time,
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

    require(
        selected is not None,
        "Nijedan validacioni kandidat nije prosao provere.",
    )

    walk_time = selected["time"]
    regularization = selected["reg"]

    print(
        f"Validacija NLL: {selected['score']:.6f}"
        f" | t: {walk_time:g}"
        f" | reg: {regularization:g}",
        flush=True,
    )

    print("Ucenje modela za test.", flush=True)

    test_training = prepare_training(
        inputs[:validation_end], targets[:validation_end]
    )
    test_parameters = fit_graph(
        test_training,
        walk_time,
        regularization,
        selected["parameters"],
    )

    test_score = evaluate(
        test_parameters,
        inputs[validation_end:],
        targets[validation_end:],
        walk_time,
    )

    print(f"Test NLL: {test_score:.6f}", flush=True)
    print("Zavrsno ucenje: svi istorijski prelazi.", flush=True)

    final_training = prepare_training(inputs, targets)
    final_parameters = fit_graph(
        final_training,
        walk_time,
        regularization,
        test_parameters,
    )

    weights, _, laplacian = graph_from_parameters(final_parameters)
    unitary = propagator(laplacian, walk_time)[0]
    signal = unitary @ history_states[-1]

    require(
        np.isclose(np.vdot(signal, signal).real, 1.0, atol=2e-11),
        "Zavrsno kvantno stanje nije normalizovano.",
    )

    next_combo, next_probability, allowed_probability = exhaustive_next(
        signal
    )

    # Sedam sest-bitnih oznaka izlazne kombinacije.
    basis_label = sum(
        (int(value) - 1) << (QUBITS * position)
        for position, value in enumerate(next_combo)
    )
    recovered = [
        ((basis_label >> (QUBITS * position)) & (DIM - 1)) + 1
        for position in range(K)
    ]

    require(
        recovered == next_combo.tolist(),
        "Provera oznaka sedam registara nije prosla.",
    )

    print(f"NEXT: {next_combo.tolist()}", flush=True)

    if DIAGNOSTICS:
        print(f"Istorijski prelazi: {count}")
        print(f"NEXT verovatnoca: {next_probability:.12g}")
        print(f"Masa dozvoljenog domena: {allowed_probability:.12g}")

    return {
        "path": path,
        "next": tuple(next_combo.tolist()),
        "digest": hashlib.sha256(draws.tobytes()).hexdigest(),
        "weights": weights,
        "signal": signal,
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
            state_difference = float(np.linalg.norm(
                first["signal"] - second["signal"]
            ))

            reason = (
                "CSV sadrzaji su razliciti;"
                f" razlika grafova={graph_difference:.6g};"
                f" razlika amplituda={state_difference:.6g}."
            )

        print(
            f"PROVERA: isti NEXT za {first['path'].name}"
            f" i {second['path'].name}. {reason}",
            flush=True,
        )


def main():
    print("Quantum Dirichlet Laplacian V1: uceni graf i kolektivne amplitude.")
    print("Lokalna kvantna simulacija; tacan NLL cele kombinacije.")
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
Quantum Dirichlet Laplacian V1: uceni graf i kolektivne amplitude.
Lokalna kvantna simulacija; tacan NLL cele kombinacije.
NEXT: najvise cetiri uzastopna broja.
Pokretanje internih provera.
Interne provere: prosle.

CSV: loto7_4720_k91.csv
Ucenje grafa: t=1 | reg=0.1.
Ucenje grafa: t=1 | reg=0.01.
Ucenje grafa: t=3 | reg=0.1.
Ucenje grafa: t=3 | reg=0.01.
Validacija NLL: 16.597941 | t: 1 | reg: 0.1
Ucenje modela za test.
Test NLL: 16.628047
Zavrsno ucenje: svi istorijski prelazi.
Pretraga NEXT: svih 15380937 kombinacija.
NEXT: [2, x, 25, y, 27, z, 37]

CSV: loto7_4720_k91_loto_2982.csv
Ucenje grafa: t=1 | reg=0.1.
Ucenje grafa: t=1 | reg=0.01.
Ucenje grafa: t=3 | reg=0.1.
Ucenje grafa: t=3 | reg=0.01.
Validacija NLL: 16.605457 | t: 1 | reg: 0.1
Ucenje modela za test.
Test NLL: 16.618889
Zavrsno ucenje: svi istorijski prelazi.
Pretraga NEXT: svih 15380937 kombinacija.
NEXT: [1, x, 4, y, 15, z, 26]

CSV: loto7_4720_k91_loto_plus_1738.csv
Ucenje grafa: t=1 | reg=0.1.
Ucenje grafa: t=1 | reg=0.01.
Ucenje grafa: t=3 | reg=0.1.
Ucenje grafa: t=3 | reg=0.01.
PROVERA: kandidat t=3, reg=0.01 odbacen: Optimizacija nije konvergirala; gradijent=3.39796e-07; iteracije=670; Desired error not necessarily achieved due to precision loss.
Validacija NLL: 16.604616 | t: 1 | reg: 0.1
Ucenje modela za test.
Test NLL: 16.570112
Zavrsno ucenje: svi istorijski prelazi.
Pretraga NEXT: svih 15380937 kombinacija.
NEXT: [2, x, 25, y, 27, z, 37]
PROVERA: isti NEXT za loto7_4720_k91.csv i loto7_4720_k91_loto_plus_1738.csv. CSV sadrzaji su razliciti; razlika grafova=0.620618; razlika amplituda=0.299476.
"""



"""
The Laplacian Quadratic Form Is Dirichlet Energy. 
The Laplacian quadratic form measures the total variation of a graph signal across edges. 
Small differences between neighboring nodes correspond to low Dirichlet energy and a smooth signal, while large differences produce higher energy and greater variation. 


Učenje grafa odnosa između brojeva, uz kontrolu koliko se signal menja preko naučenih veza.

Za neusmeren graf sa nenegativnim težinama:
\[
E(x)=x^\top Lx=\sum_{\{i,j\}\in E}w_{ij}(x_i-x_j)^2.
\]
Ovde su „susedi” čvorovi povezani granom — ne nužno uzastopni brojevi. Veza može spajati, na primer, 4 i 29. 


- 39 čvorova predstavlja brojeve. Veze i njihove težine uče se iz istorijskih podataka.
- Istorija određuje signal na grafu. Model se obučava prema narednim izvlačenjima, kako graf ne bi opisivao samo istovremeno pojavljivanje.
- Laplacijan upravlja kvantnim prenosom amplituda. Mogući propagator je \(U(t)=e^{-itL}\); njegovu vezu sa zajedničkom raspodelom sedam različitih brojeva posebno izvedeni.

Samo minimizovanje Dirichletove energije nije dovoljno. Konstantan signal ima nultu energiju, a pri učenju težina trivijalno rešenje može biti graf bez grana. Zato prediktivni cilj i ograničenja grafa moraju sprečiti ta rešenja. Postoje konkretne metode za učenje grafa iz signala. 

Vrednost ideje je u učenoj strukturi veza i dinamici amplituda. 


V1 uči težine veza između svih 39 brojeva. Laplacijan upravlja kvantnim kretanjem amplituda, a verovatnoća sedmorke računa se iz njihovog zajedničkog zbira, bez sedam odvojenih rang-lista. Normalizaciju izvodim analitički.


V1 uči graf iz istorijskih prelaza. Dirichletova energija učestvuje u učenju, a Laplacijan upravlja kvantnim kretanjem amplituda. Brojčani susedi nisu unapred povezani posebnim pravilom.

Verovatnoća sedmorke računa se kolektivno, iz amplituda svih njenih brojeva. Normalizacija je analitička; NEXT pretražuje svih 15.380.937 kombinacija.

Ugrađene provere proveravaju gradijent, Qiskit simulaciju, normalizaciju i izbor maksimuma. 
"""
