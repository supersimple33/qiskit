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
from qiskit.circuit.library import DraperQFTAdder


def alt_adder(N: int) -> QuantumCircuit:
    """Ripple carry adder circuit.

    Thapliyal, H., & Ranganathan, N. (2013). Design of efficient reversible logic-based binary and BCD adder circuits. ACM Journal on Emerging Technologies in Computing Systems (JETC), 9(3), 1-31.
    """

    qr_a = QuantumRegister(N, "a")
    qr_b = QuantumRegister(N, "b")
    circuit = QuantumCircuit(qr_a, qr_b, name=f"ctrl_sbt_{N}")

    # Step 1
    for i in range(1, N):
        circuit.cx(qr_a[i], qr_b[i])

    # Step 2
    for i in range(N - 2, 0, -1):
        circuit.cx(qr_a[i], qr_a[i + 1])

    # Step 3
    for i in range(N - 1):
        circuit.ccx(qr_a[i], qr_b[i], qr_a[i + 1])

    # Step 4
    # circuit.cx(qr_a[N - 2], qr_b[N - 2])
    for i in range(N - 2, -1, -1):
        circuit.ccx(qr_a[i], qr_b[i], qr_a[i + 1])
        circuit.cx(qr_a[i], qr_b[i])

    # Step 5
    for i in range(1, N - 1):
        circuit.cx(qr_a[i], qr_a[i + 1])

    for i in range(1, N):
        circuit.cx(qr_a[i], qr_b[i])

    return circuit


def adder(N: int) -> QuantumCircuit:
    """Ripple carry adder circuit. Based on the control adder without a control."""

    qr_a = QuantumRegister(N, "a")
    qr_b = QuantumRegister(N, "b")
    circuit = QuantumCircuit(qr_a, qr_b, name=f"add_{N}")

    # # Invert the a qubits # FOR SUBTRACTION
    # for i in range(N):
    #     circuit.cx(qr_c[0], qr_b[i])

    # 3: Calculate the binary add of a and b
    for i in range(N - 1, 0, -1):
        circuit.cx(qr_a[i], qr_b[i])

    # 3b: step
    for i in range(N - 2, 0, -1):
        circuit.cx(qr_a[i], qr_a[i + 1])

    # 4: shift ups
    for i in range(N - 1):
        circuit.ccx(qr_a[i], qr_b[i], qr_a[i + 1])

    # 5: controlled sub
    for i in range(N - 1, 0, -1):
        circuit.cx(qr_a[i], qr_b[i])
        circuit.ccx(qr_a[i - 1], qr_b[i - 1], qr_a[i])

    circuit.cx(qr_a[0], qr_b[0])

    # Undo 3b:
    for i in range(1, N - 1):
        circuit.cx(qr_a[i], qr_a[i + 1])

    # Undo 3a:
    for i in range(1, N):
        circuit.cx(qr_a[i], qr_b[i])

    # FOR SUBTRACTION
    # for i in range(N):
    #     circuit.cx(qr_c[0], qr_b[i])

    return circuit


def ctrl_add(N: int) -> QuantumCircuit:
    """Controlled Adder circuit.

    Muñoz-Coreas, E., & Thapliyal, H. (2017). T-count optimized design of quantum integer multiplication. arXiv preprint arXiv:1706.05113.
    """
    qr_a = QuantumRegister(N, "a")
    qr_b = QuantumRegister(N, "b")
    qr_c = QuantumRegister(1, "c")
    circuit = QuantumCircuit(qr_a, qr_b, qr_c, name=f"ctrl_add_{N}")

    # # Invert the a qubits # FOR SUBTRACTION
    # for i in range(N):
    #     circuit.cx(qr_c[0], qr_b[i])

    # 3: Calculate the binary add of a and b
    for i in range(N - 1, 0, -1):
        circuit.cx(qr_a[i], qr_b[i])

    # 3b: step
    for i in range(N - 2, 0, -1):
        circuit.cx(qr_a[i], qr_a[i + 1])

    # 4: shift ups
    for i in range(N - 1):
        circuit.ccx(qr_a[i], qr_b[i], qr_a[i + 1])

    # 5: controlled sub
    for i in range(N - 1, 0, -1):
        circuit.ccx(qr_c[0], qr_a[i], qr_b[i])
        circuit.ccx(qr_a[i - 1], qr_b[i - 1], qr_a[i])

    circuit.ccx(qr_c[0], qr_a[0], qr_b[0])

    # Undo 3b:
    for i in range(1, N - 1):
        circuit.cx(qr_a[i], qr_a[i + 1])

    # Undo 3a:
    for i in range(1, N):
        circuit.cx(qr_a[i], qr_b[i])

    # FOR SUBTRACTION
    # for i in range(N):
    #     circuit.cx(qr_c[0], qr_b[i])

    return circuit


def cmpr(N: int):
    qr_a = QuantumRegister(N, "a")
    qr_b = QuantumRegister(N, "b")
    qr_aux = QuantumRegister(2, "aux")
    circuit = QuantumCircuit(qr_a, qr_b, qr_aux, name=f"cmpr_{N}")

    # 1: Prepare the aux

    # 2: Computer xor
    for i in range(N):
        circuit.cx(qr_b[i], qr_a[i])

    # cnots going up
    circuit.cx(qr_b[1], qr_aux[0])
    for i in range(2, N):
        circuit.cx(qr_b[i], qr_b[i - 1])
    circuit.cx(qr_b[N - 1], qr_aux[1])

    circuit.ccx(qr_a[0], qr_b[0], qr_aux[0])
    for i in range(1, N):
        circuit.ccx(
            qr_b[i - 1] if i != 1 else qr_aux[0],
            qr_a[i],
            qr_b[i] if i != N - 1 else qr_aux[1],
            ctrl_state="01",
        )

    for i in range(1, N - 1):
        circuit.ccx(
            qr_b[i - 1] if i != 1 else qr_aux[0],
            qr_a[i],
            qr_b[i] if i != N - 1 else qr_aux[1],
            ctrl_state="01",
        )
    circuit.ccx(qr_a[0], qr_b[0], qr_aux[0])

    for i in range(2, N):
        circuit.cx(qr_b[i], qr_b[i - 1])
    circuit.cx(qr_b[1], qr_aux[0])

    for i in range(N):
        circuit.cx(qr_b[i], qr_a[i])

    return circuit


def long_division_divider(N: int, draper=False) -> QuantumCircuit:
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

    [1] Thapliyal, Himanshu et al. “Quantum Circuit Designs of Integer Division Optimizing T-count and T-depth.”
    IEEE Transactions on Emerging Topics in Computing 9 (2018): 1045-1056.

    """

    qr_a = QuantumRegister(N, "a")  # qr_q
    qr_b = QuantumRegister(N, "b")
    qr_r = QuantumRegister(N, "r")

    circuit = QuantumCircuit(qr_a, qr_b, qr_r)

    add = adder(N)
    cadd = ctrl_add(N)
    if draper:
        add = DraperQFTAdder(N, kind="fixed")
        cadd = DraperQFTAdder(N, kind="fixed").control()

    for i in range(1, N):
        # 1: Prepare the Y register
        # 1.1 Create the Y register
        Y = qr_a[N - i :] + qr_r[: N - i]
        print(len(Y))

        # 1.2 Subtract off b from Y
        circuit.x(Y)
        circuit.append(add, qr_b[:] + Y)
        circuit.x(Y)

        # 2: prepare r_n-i for use
        circuit.cx(Y[N - 1], qr_r[N - i])

        # 3: Controlled add
        if draper:
            circuit.append(cadd, [qr_r[N - i]] + qr_b[:] + Y[:])
        else:
            circuit.append(cadd, qr_b[:] + Y[:] + [qr_r[N - i]])

        # 4: apply not gate
        circuit.x(qr_r[N - i])

    # 1: Subtraction
    circuit.x(qr_a)
    circuit.append(add, qr_b[:] + qr_a[:])
    circuit.x(qr_a)

    # 2: setup r_0
    circuit.cx(qr_a[N - 1], qr_r[0])

    # 3: Controlled add
    if draper:
        circuit.append(cadd, [qr_r[0]] + qr_b[:] + qr_a[:])
    else:
        circuit.append(cadd, qr_b[:] + qr_a[:] + [qr_r[0]])

    # 4: final not gate
    circuit.x(qr_r[0])

    return circuit


def other_divider(
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
    qr_aux = QuantumRegister(3, "aux")  # N

    # build the division circuit
    circuit = QuantumCircuit(qr_d, qr_q, qr_s, qr_aux)

    # prepare comparators and subtractors
    # from qiskit.circuit.library.arithmetic import IntegerComparator

    comparator = cmpr(num_divisor_qubits)
    sub_eql = ctrl_sbt(num_divisor_qubits)
    sub_uneql = ctrl_sbt(num_divisor_qubits + 1)

    for i in range(num_dividend_qubits - num_divisor_qubits + 1):
        # print(len(qr_d[num_dividend_qubits - num_divisor_qubits - i : num_dividend_qubits - i]))
        circuit.append(
            comparator,
            qr_d[num_dividend_qubits - num_divisor_qubits - i : num_dividend_qubits - i]
            + qr_q[:]
            + qr_aux[:2],
        )
        circuit.x(qr_aux[1])
        circuit.cx(qr_aux[1], qr_s[num_dividend_qubits - num_divisor_qubits - i])  #  CTRL STATE

        # if i == num_dividend_qubits - num_divisor_qubits:
        #     break

        circuit.append(
            sub_eql,
            qr_q[:]
            + qr_d[num_dividend_qubits - num_divisor_qubits - i : num_dividend_qubits - i]
            + [qr_aux[1]],
        )
        circuit.x(qr_aux[1])

        circuit.ccx(qr_d[num_dividend_qubits - 1 - i], qr_aux[1], qr_aux[0])
        # print(len(qr_d[num_dividend_qubits - num_divisor_qubits - i - 1 : num_dividend_qubits - i]))

        circuit.cx(
            qr_d[num_dividend_qubits - 1 - i],
            qr_s[num_dividend_qubits - num_divisor_qubits - i - 1],
        )

        if i == num_dividend_qubits - num_divisor_qubits:
            break

        print(len(qr_d[num_dividend_qubits - num_divisor_qubits - i - 1 : num_dividend_qubits - i]))
        circuit.append(
            sub_uneql,
            (
                qr_q[:]
                + [qr_aux[2]]
                + qr_d[num_dividend_qubits - num_divisor_qubits - i - 1 : num_dividend_qubits - i]
                + [qr_aux[0]]
            ),
        )

        circuit.reset(qr_aux[:2])

    circuit.x(qr_aux[1])
    return circuit
