def test_import_shmistogram():
    import shmistogram as shm

    assert "__version__" in dir(shm)


def test_import_empdens():
    import empdens as dens

    assert "__version__" in dir(dens)
