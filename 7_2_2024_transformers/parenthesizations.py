import itertools
import functools


@functools.lru_cache(maxsize=None)
def valid(n):
    """
    Returns a list of all possible parenthesizations of length n.

    Parameters:
      n (int): The length of the parenthesizations.

    Returns:
      A list of strings, where each inner string represents a valid parenthesization of length n.

    Example:
    >>> parenthesizations(3)
    {'((()))', '(()())', '(())()', '()(())', '()()()'}
    """
    if n == 0:
        return set([""])
    else:
        result = set()
        for i in range(n):
            for c in valid(i):
                for d in valid(n - i - 1):
                    result.add("(" + c + ")" + d)
        return result


def all(n):
    """
    Generates a set of all  parenthesizations of length 2n.

    Parameters:
      n (int): The length of the parenthesizations.

    Returns:
      set: A set of strings, where each string represents an invalid parenthesization of length 2n.

    Example:
    >>> invalid(3)
    {"())())", ")))(((", ...}
    """
    return set("".join(p) for p in itertools.product("()", repeat=2 * n))


def all_balanced(n):
    return set(
        "".join(["(" if i in pos else ")" for i in range(2 * n)])
        for pos in itertools.combinations(range(2 * n), n)
    )
