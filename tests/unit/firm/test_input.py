from firm.Input import Solution


def test_example_test():
    assert isinstance(Solution, type)


def test_example_failing_test():
    assert 1 == 0, "This will fail"
