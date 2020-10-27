from dataclasses import dataclass
from typing import Optional


@dataclass
class BinarySearchHint:
    '''Class representing the incremental step of a binary search test.'''

    '''True if the target of the search has been found.'''
    found: bool = False

    '''True if the parameter value was determined to be too low.'''
    tooLow: bool = False


@dataclass
class BinarySearchResult:
    '''Class representing the output of a binary search.'''

    '''True if the target of the search has been found.'''
    found: bool = True

    '''If found=True, this field contains the output value of the search.
    Otherwise, the value is undefined, but set to None by default. If
    found=True and value=None, then None is the value found during the search.
    '''
    value: Optional[float] = None


def binary_search(data, test):
    '''
    Perform a binary search for a given parameter value.
    Call the type of the value being searched for T.
    Args:
    - data: a list of T
    - test: a callable T -> BinarySearchHint
    Returns:
      An instance of BinarySearchResult
    '''
    current_min = 0
    current_max = len(data) - 1

    while current_min <= current_max:
        tested_index = (current_min + current_max) // 2
        tested_value = data[tested_index]
        hint = test(tested_value)
        if hint.found:
            return BinarySearchResult(found=True, value=tested_value)
        elif hint.tooLow:
            current_min = tested_index + 1
        else:
            current_max = tested_index - 1

    return BinarySearchResult(found=False, value=None)
