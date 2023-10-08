from typing import List


def reverse_strings(strings: List[str]) -> List[str]:
    """Reverse the strings in a list."""
    reversed_strings = [s[::-1] for s in strings]
    return reversed_strings


def find_even_numbers(numbers: List[int]) -> List[int]:
    """Find and return even numbers from a list of integers.

    Args:
        numbers (List[int]): List of integers.

    Returns:
        List[int]: List of even numbers.
    """
    even_numbers = [num for num in numbers if num % 2 == 0]
    return even_numbers
