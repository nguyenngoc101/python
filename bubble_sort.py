def bubble_sort(arr: list) -> list:
    """
    Sort a list using the bubble sort algorithm.
    
    Bubble sort repeatedly steps through the list, compares adjacent elements,
    and swaps them if they are in the wrong order. This process repeats until
    the list is sorted.
    
    Time Complexity: O(n²) - where n is the number of elements
    Space Complexity: O(1) - in-place sorting
    
    Args:
        arr: The list to be sorted
        
    Returns:
        The sorted list (sorted in-place)
    """
    n = len(arr)
    
    for i in range(n):
        # Track if any swaps occurred in this pass
        swapped = False
        
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            # Compare adjacent elements
            if arr[j] > arr[j + 1]:
                # Swap if the element is greater than the next
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        # If no swaps occurred, array is already sorted
        if not swapped:
            break
    
    return arr


# Example usage and demonstration
if __name__ == "__main__":
    # Test with different arrays
    test_cases = [
        [64, 34, 25, 12, 22, 11, 90],
        [5, 1, 4, 2, 8],
        [1, 2, 3, 4, 5],  # Already sorted
        [5, 4, 3, 2, 1],  # Reverse sorted
        [],               # Empty array
        [1],              # Single element
    ]
    
    print("Bubble Sort Demonstration")
    print("=" * 40)
    
    for arr in test_cases:
        original = arr.copy()
        sorted_arr = bubble_sort(arr)
        print(f"Original: {original}")
        print(f"Sorted:   {sorted_arr}")
        print("-" * 40)
