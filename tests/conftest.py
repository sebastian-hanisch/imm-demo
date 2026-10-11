"""Gemeinsame Fixtures: Mini-Beispiele, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mix_example():
    """Zwei Modelle mit gleicher Wahrscheinlichkeit 0,5, Wechselwahrscheinlichkeit 0,2, Schätzungen x_1 = (0; 0) und x_2 = (10; 0), Kovarianz je I.

    Handrechnung: Mischgewichte w[i, j] = p_ij mu_i / c_j mit c_j = 0,5: [[0,8; 0,2], [0,2; 0,8]].
    Gemischter Anfang Modell 1: 0,8 * 0 + 0,2 * 10 = 2, Modell 2: 0,2 * 0 + 0,8 * 10 = 8.
    Kovarianz Modell 1 (Position): 1 + 0,8 * 2^2 + 0,2 * 8^2 = 1 + 3,2 + 12,8 = 17.
    """
    xs = [np.array([0.0, 0.0]), np.array([10.0, 0.0])]
    Ps = [np.eye(2), np.eye(2)]
    mu = np.array([0.5, 0.5])
    return xs, Ps, mu
