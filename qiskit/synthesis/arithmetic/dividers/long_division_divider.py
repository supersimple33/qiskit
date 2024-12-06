# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Compute the quotient and remainder of two qubit registers using yuan et al's long division method."""

from __future__ import annotations

import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister


def ctrl_eql_sbt(N: int) -> QuantumCircuit:
    qr_a = QuantumRegister(N, "a")
    qr_b = QuantumRegister(N, "b")
    qr_c = QuantumRegister(1, "c")
    circuit = QuantumCircuit(qr_a, qr_b, qr_c)

    # Invert the a qubits
    for i in range(N):
        circuit.cx(qr_c[0], qr_a[i])

    # 3: Calculate the binary add of a and b
    for i in range(N - 1, 0, -1):
        circuit.cx(qr_a[i], qr_b[i])

    # 4: shift ups
    for i in range(N - 1):
        circuit.ccx(qr_a[i], qr_b[i], qr_a[i + 1])

    # 5: controlled sub
    for i in range(N - 1, 0, -1):
        circuit.ccx(qr_c[0], qr_a[i], qr_b[i])
        circuit.ccx(qr_a[i - 1], qr_b[i - 1], qr_a[i])

    # circuit.ccx(qr_c[0], qr_a[0], qr_b[0])

    # Undo 3:
    for i in range(1, num_state_qubits):
        circuit.cx(qr_a[i], qr_b[i])

    # Invert all the qubits
    for i in range(num_state_qubits):
        circuit.cx(qr_c[0], qr_a[i])
        circuit.cx(qr_c[0], qr_b[i])

    return circuit


def ctrl_uneql_sbt(N: int, M: int):
    qr_a = QuantumRegister(N, "a")
    qr_b = QuantumRegister(N, "b")
    qr_c = QuantumRegister(1, "c")
    circuit = QuantumCircuit(qr_a, qr_b, qr_c)

    # Invert the a qubits
    for i in range(N):
        circuit.cx(qr_c[0], qr_a[i])

    # 3: Calculate the binary add of a and b
    for i in range(M - 1, 0, -1):
        circuit.cx(qr_a[i], qr_b[i])

    # 4: shift ups
    for i in range(N - 1):
        circuit.ccx(qr_a[i], qr_b[i], qr_a[i + 1])

    # 5: controlled sub
    for i in range(N - 1, 0, -1):
        circuit.ccx(qr_c[0], qr_a[i], qr_b[i])
        # if i <= M:
        circuit.ccx(qr_a[i - 1], qr_b[i - 1], qr_a[i])

    # Undo 3:
    for i in range(1, M):
        circuit.cx(qr_a[i], qr_b[i])

    # Invert all the qubits
    for i in range(N):
        circuit.cx(qr_c[0], qr_a[i])
        circuit.cx(qr_c[0], qr_b[i])

    return circuit


def cmpr(N: int):
    qr_a = QuantumRegister(N, "a")
    qr_b = QuantumRegister(N, "b")
    qr_aux = QuantumRegister(2, "aux")
    circuit = QuantumCircuit(qr_a, qr_b, qr_aux)

    # 1: Prepare the aux

    # 2: Computer xor
    for i in range(N):
        circuit.cx(qr_b[i], qr_a[i])

    # 3: start ancilla
    circuit.ccx(qr_a[0], qr_b[0], qr_aux[0])
    circuit.cx(qr_b[1], qr_aux[0])
    j = 1

    # 4: controlled xor
    for i in range(N - 1, 0, -1):
        circuit.x(qr_a[i])
        circuit.ccx(qr_aux[not j], qr_a[i], qr_aux[j])
        circuit.x(qr_a[i])
        circuit.cx(qr_b[i], qr_aux[j])
        j = not j

    # 4: UnComputer xor
    for i in range(N):
        circuit.cx(qr_b[i], qr_a[i])

    return circuit


def long_division_divider(
    num_dividend_qubits: int,
    num_divisor_qubits: int | None = None,
) -> QuantumCircuit:
    r"""A long division divider circuit to compute the quotient and remainder of two qubit registers.

    Division in this circuit is implemented using the procedure of Table. 1 in [1], where
    the long division method is implemented using comparators and subtractors.


    weighted sum rotations are implemented as given in Fig. 5 in [1]. QFT is used on the output
    register and is followed by rotations controlled by input registers. The rotations
    transform the state into the product of two input registers in QFT base, which is
    reverted from QFT base using inverse QFT.
    For example, on 4 qubit dividend and 2 qubit divisor, a full divider is given by:

    .. plot::
        :include-source:

        from qiskit.synthesis.arithmetic import long_division_divider

    Args:
        num_dividend_qubits: The number of qubits in the dividend register for
            state :math:`|d\rangle`.
        num_divisor_qubits: The number of qubits in the divisor register for
                        state :math:`|q\rangle`. Default value is ``num_dividend_qubits``
                        to keep divisor and dividend registers equal in size.

    Raises:
        ValueError: If ``num_divisor_qubits`` is given and not valid, meaning it is
            greater than ``num_dividend_qubits``.

    **References:**

    [1] Yuan et al., A novel fault-tolerant quantum divider and its simulation, 2022.
    `doi:10.1007/s11128-022-03523-8 <https://doi.org/10.1007/s11128-022-03523-8>`_

    """
    # define the registers
    if num_divisor_qubits is None:
        num_divisor_qubits = num_dividend_qubits
    elif num_divisor_qubits > num_dividend_qubits:
        raise ValueError(
            "Number of divisor qubits cannot be greater than number of dividend qubits."
        )

    qr_d = QuantumRegister(num_dividend_qubits, "d")
    qr_q = QuantumRegister(num_divisor_qubits, "q")
    qr_s = QuantumRegister(num_dividend_qubits - num_divisor_qubits + 1, "s")

    # build the division circuit
    circuit = QuantumCircuit(qr_d, qr_q, qr_s)

    # prepare comparators and subtractors
    from qiskit.circuit.library.arithmetic import IntegerComparator

    comparator = IntegerComparator(num_divisor_qubits)

    for i in range(num_dividend_qubits - num_divisor_qubits + 1):
        pass
