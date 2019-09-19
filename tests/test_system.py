import numpy as np

import simulation.system as system

import pytest

sys_a = None
sys_b = None
sys_c = None

def setup():
    global sys_a
    global sys_b
    global sys_c

    pos_a = np.array([
        [1, 2],
        [2, 4],
        [.1, .3],
        [2, -1]
    ], dtype=float)
    pos_b = np.array([
        [4, 2],
        [5, 3],
    ])
    pos_c = np.array([
        [1, 2, 3],
        [1, 4, 4],
        [2, 2, 1]
    ])

    mass_a = np.array([1, 2, 3, 4])
    mass_b = np.array([6, 7])
    mass_c = np.array([3, 3, 3])

    sys_a = system.System(
        pos=pos_a,
        mass=mass_a
    )
    sys_b = system.System(
        pos=pos_b,
        mass=mass_b
    )
    sys_c = system.System(
        pos=pos_c,
        mass=mass_c
    )

def test_sys_appending():
    global sys_a
    global sys_b
    global sys_c

    assert sys_a.N == 4
    assert sys_b.N == 2
    assert sys_c.N == 3

    sys_a += sys_b
    assert sys_a.N == 4 + sys_b.N
    with pytest.raises(system.SystemError) as _:
        sys_b.append(sys_c)
    
    assert sys_b.N == 2
    assert sys_c.N == 3
    
    add_b = {'pos': [[1, 4], [1, 1]], 'mass': [1, 1]}
    add_bad = {'pos': [[1, 4], [1, 1]], 'mass': [1, 1, 1]}

    sys_b += add_b
    assert sys_b.N == 2 + 2
    with pytest.raises(system.SystemError) as _:
        sys_b += add_bad
    
    assert sys_b.N == 2 + 2

    prev_n = sys_a.N
    sys_a += sys_a
    assert sys_a.N == 2 * prev_n



def test_sys_modifying():
    global sys_a
    global sys_b
    global sys_c

    assert sys_a.N == 4
    original_pos = sys_a.pos.copy()
    sys_a.set('pos', [-1, -1], [0])
    assert np.all(sys_a.pos[0] == [-1, -1])
    assert np.all(sys_a.pos[1] == original_pos[1])
    assert sys_a.N == 4

    d = sys_a[:2]
    e = sys_a[1]
    assert np.all(d.pos[1] == e.pos[0])
    assert d.N == 2
    assert e.N == 1
    m = sys_a.get('mass')
    assert np.all(sys_a.mass == m)
    m *= 2
    assert np.all(sys_a.mass * 2 == m)
    sys_a.set('mass', m)
    assert np.all(sys_a.mass == m)
    sys_a.set('mass', [2], index=[0])
    assert sys_a.mass[0] == np.array([2])
    pass
