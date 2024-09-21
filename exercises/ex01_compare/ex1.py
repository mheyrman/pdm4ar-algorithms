from typing import NewType, Tuple

ComparisonOutcome = NewType("ComparisonOutcome", str)
""" The type of comparison outcomes """
FIRST_PREFERRED = ComparisonOutcome("first_preferred")
""" We prefer the first option. """
SECOND_PREFERRED = ComparisonOutcome("second_preferred")
""" We prefer the second option. """
INDIFFERENT = ComparisonOutcome("indifferent")
""" We are indifferent among the two options. """


def compare_lexicographic(a: Tuple[float], b: Tuple[float]) -> ComparisonOutcome:
    """
    Implement here your solution.
    The two tuples represent two vectors of outcomes (e.g. different cost function realizations) for two different decisions.
    Which one is preferred?

    Note that the terms are sorted lexicographically by importance.
    For example, the term in position 1 is less important than the one in position 0,
    but more important than the one in position 2
    """
    # todo
    
    outcome = INDIFFERENT
    found = False
    i = 0
    while not found and i < len(a):
        if a[i] == b[i]:
            found = False
        elif a[i] < b[i]:
            outcome = FIRST_PREFERRED
            found = True
        else:
            outcome = SECOND_PREFERRED
            found = True
        i = i + 1

    return outcome