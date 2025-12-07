"""
10 Popular Sorting Algorithms Implementation
============================================

This module contains implementations of 10 popular sorting algorithms:
1. Bubble Sort
2. Selection Sort
3. Insertion Sort
4. Merge Sort
5. Quick Sort
6. Heap Sort
7. Counting Sort
8. Radix Sort
9. Bucket Sort
10. Shell Sort

Each algorithm includes:
- Time complexity analysis
- Space complexity analysis
- A brief description
"""

from typing import List
import math


# =============================================================================
# 1. BUBBLE SORT
# =============================================================================
def bubble_sort(arr: List[int]) -> List[int]:
    """
    Bubble Sort Algorithm
    
    Repeatedly steps through the list, compares adjacent elements,
    and swaps them if they are in the wrong order.
    
    Time Complexity:
        - Best: O(n) when array is already sorted
        - Average: O(n²)
        - Worst: O(n²)
    
    Space Complexity: O(1)
    Stable: Yes
    """
    arr = arr.copy()
    n = len(arr)
    
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        # Optimization: if no swapping occurred, array is sorted
        if not swapped:
            break
    
    return arr


# =============================================================================
# 2. SELECTION SORT
# =============================================================================
def selection_sort(arr: List[int]) -> List[int]:
    """
    Selection Sort Algorithm
    
    Divides the array into sorted and unsorted regions.
    Repeatedly selects the smallest element from unsorted region
    and moves it to the sorted region.
    
    Time Complexity:
        - Best: O(n²)
        - Average: O(n²)
        - Worst: O(n²)
    
    Space Complexity: O(1)
    Stable: No
    """
    arr = arr.copy()
    n = len(arr)
    
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    
    return arr


# =============================================================================
# 3. INSERTION SORT
# =============================================================================
def insertion_sort(arr: List[int]) -> List[int]:
    """
    Insertion Sort Algorithm
    
    Builds the sorted array one item at a time by repeatedly
    picking the next item and inserting it into its correct position.
    
    Time Complexity:
        - Best: O(n) when array is already sorted
        - Average: O(n²)
        - Worst: O(n²)
    
    Space Complexity: O(1)
    Stable: Yes
    """
    arr = arr.copy()
    
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    
    return arr


# =============================================================================
# 4. MERGE SORT
# =============================================================================
def merge_sort(arr: List[int]) -> List[int]:
    """
    Merge Sort Algorithm
    
    Divide and conquer algorithm that divides the array into halves,
    recursively sorts them, and then merges the sorted halves.
    
    Time Complexity:
        - Best: O(n log n)
        - Average: O(n log n)
        - Worst: O(n log n)
    
    Space Complexity: O(n)
    Stable: Yes
    """
    arr = arr.copy()
    
    if len(arr) <= 1:
        return arr
    
    def merge(left: List[int], right: List[int]) -> List[int]:
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)


# =============================================================================
# 5. QUICK SORT
# =============================================================================
def quick_sort(arr: List[int]) -> List[int]:
    """
    Quick Sort Algorithm
    
    Divide and conquer algorithm that picks an element as pivot
    and partitions the array around the pivot.
    
    Time Complexity:
        - Best: O(n log n)
        - Average: O(n log n)
        - Worst: O(n²) when array is already sorted
    
    Space Complexity: O(log n) for recursion stack
    Stable: No
    """
    arr = arr.copy()
    
    def partition(low: int, high: int) -> int:
        # Using median-of-three pivot selection for better performance
        mid = (low + high) // 2
        if arr[low] > arr[mid]:
            arr[low], arr[mid] = arr[mid], arr[low]
        if arr[low] > arr[high]:
            arr[low], arr[high] = arr[high], arr[low]
        if arr[mid] > arr[high]:
            arr[mid], arr[high] = arr[high], arr[mid]
        
        # Move median to high-1 position
        arr[mid], arr[high - 1] = arr[high - 1], arr[mid]
        pivot = arr[high - 1]
        
        i = low
        j = high - 1
        
        while True:
            i += 1
            while arr[i] < pivot:
                i += 1
            j -= 1
            while arr[j] > pivot:
                j -= 1
            if i >= j:
                break
            arr[i], arr[j] = arr[j], arr[i]
        
        arr[i], arr[high - 1] = arr[high - 1], arr[i]
        return i
    
    def quick_sort_helper(low: int, high: int):
        if high - low < 2:
            if high > low and arr[low] > arr[high]:
                arr[low], arr[high] = arr[high], arr[low]
            return
        
        pivot_idx = partition(low, high)
        quick_sort_helper(low, pivot_idx - 1)
        quick_sort_helper(pivot_idx + 1, high)
    
    if len(arr) > 1:
        quick_sort_helper(0, len(arr) - 1)
    
    return arr


# =============================================================================
# 6. HEAP SORT
# =============================================================================
def heap_sort(arr: List[int]) -> List[int]:
    """
    Heap Sort Algorithm
    
    Builds a max-heap from the input data, then repeatedly extracts
    the maximum element and rebuilds the heap.
    
    Time Complexity:
        - Best: O(n log n)
        - Average: O(n log n)
        - Worst: O(n log n)
    
    Space Complexity: O(1)
    Stable: No
    """
    arr = arr.copy()
    n = len(arr)
    
    def heapify(size: int, root: int):
        largest = root
        left = 2 * root + 1
        right = 2 * root + 2
        
        if left < size and arr[left] > arr[largest]:
            largest = left
        
        if right < size and arr[right] > arr[largest]:
            largest = right
        
        if largest != root:
            arr[root], arr[largest] = arr[largest], arr[root]
            heapify(size, largest)
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(n, i)
    
    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(i, 0)
    
    return arr


# =============================================================================
# 7. COUNTING SORT
# =============================================================================
def counting_sort(arr: List[int]) -> List[int]:
    """
    Counting Sort Algorithm
    
    Non-comparison based sorting algorithm that works by counting
    the occurrences of each unique element.
    
    Time Complexity:
        - Best: O(n + k) where k is the range of input
        - Average: O(n + k)
        - Worst: O(n + k)
    
    Space Complexity: O(n + k)
    Stable: Yes
    
    Note: Works best for arrays with small range of positive integers.
    """
    if not arr:
        return []
    
    arr = arr.copy()
    
    # Handle negative numbers by shifting
    min_val = min(arr)
    max_val = max(arr)
    range_val = max_val - min_val + 1
    
    # Create count array
    count = [0] * range_val
    output = [0] * len(arr)
    
    # Count occurrences
    for num in arr:
        count[num - min_val] += 1
    
    # Calculate cumulative count
    for i in range(1, len(count)):
        count[i] += count[i - 1]
    
    # Build output array (iterate in reverse for stability)
    for i in range(len(arr) - 1, -1, -1):
        output[count[arr[i] - min_val] - 1] = arr[i]
        count[arr[i] - min_val] -= 1
    
    return output


# =============================================================================
# 8. RADIX SORT
# =============================================================================
def radix_sort(arr: List[int]) -> List[int]:
    """
    Radix Sort Algorithm
    
    Non-comparison based sorting algorithm that sorts numbers
    digit by digit, starting from the least significant digit.
    
    Time Complexity:
        - Best: O(nk) where k is the number of digits
        - Average: O(nk)
        - Worst: O(nk)
    
    Space Complexity: O(n + k)
    Stable: Yes
    
    Note: This implementation handles positive integers.
    """
    if not arr:
        return []
    
    arr = arr.copy()
    
    # Handle negative numbers
    negatives = [-x for x in arr if x < 0]
    positives = [x for x in arr if x >= 0]
    
    def radix_sort_positive(nums: List[int]) -> List[int]:
        if not nums:
            return []
        
        max_val = max(nums)
        exp = 1
        
        while max_val // exp > 0:
            # Counting sort for current digit
            count = [0] * 10
            output = [0] * len(nums)
            
            for num in nums:
                digit = (num // exp) % 10
                count[digit] += 1
            
            for i in range(1, 10):
                count[i] += count[i - 1]
            
            for i in range(len(nums) - 1, -1, -1):
                digit = (nums[i] // exp) % 10
                output[count[digit] - 1] = nums[i]
                count[digit] -= 1
            
            nums = output
            exp *= 10
        
        return nums
    
    sorted_negatives = [-x for x in reversed(radix_sort_positive(negatives))]
    sorted_positives = radix_sort_positive(positives)
    
    return sorted_negatives + sorted_positives


# =============================================================================
# 9. BUCKET SORT
# =============================================================================
def bucket_sort(arr: List[int]) -> List[int]:
    """
    Bucket Sort Algorithm
    
    Distributes elements into buckets, sorts each bucket individually
    (using insertion sort), and then concatenates all buckets.
    
    Time Complexity:
        - Best: O(n + k) when elements are uniformly distributed
        - Average: O(n + k)
        - Worst: O(n²) when all elements are in one bucket
    
    Space Complexity: O(n + k)
    Stable: Yes (when using stable sort for buckets)
    """
    if not arr:
        return []
    
    arr = arr.copy()
    n = len(arr)
    
    # Handle the case where all elements are the same
    min_val = min(arr)
    max_val = max(arr)
    
    if min_val == max_val:
        return arr
    
    # Create buckets
    bucket_count = n
    bucket_range = (max_val - min_val) / bucket_count
    buckets = [[] for _ in range(bucket_count)]
    
    # Distribute elements into buckets
    for num in arr:
        if num == max_val:
            bucket_idx = bucket_count - 1
        else:
            bucket_idx = int((num - min_val) / bucket_range)
        buckets[bucket_idx].append(num)
    
    # Sort each bucket using insertion sort
    def insertion_sort_bucket(bucket: List[int]) -> List[int]:
        for i in range(1, len(bucket)):
            key = bucket[i]
            j = i - 1
            while j >= 0 and bucket[j] > key:
                bucket[j + 1] = bucket[j]
                j -= 1
            bucket[j + 1] = key
        return bucket
    
    # Concatenate sorted buckets
    result = []
    for bucket in buckets:
        result.extend(insertion_sort_bucket(bucket))
    
    return result


# =============================================================================
# 10. SHELL SORT
# =============================================================================
def shell_sort(arr: List[int]) -> List[int]:
    """
    Shell Sort Algorithm
    
    Generalization of insertion sort that allows the exchange of items
    that are far apart. Uses a gap sequence to determine the stride.
    
    Time Complexity:
        - Best: O(n log n)
        - Average: O(n^1.25) to O(n^1.5) depending on gap sequence
        - Worst: O(n²) with original gap sequence
    
    Space Complexity: O(1)
    Stable: No
    
    Note: This implementation uses Knuth's gap sequence (3^k - 1) / 2
    """
    arr = arr.copy()
    n = len(arr)
    
    # Calculate initial gap using Knuth's sequence
    gap = 1
    while gap < n // 3:
        gap = gap * 3 + 1
    
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 3
    
    return arr


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def verify_sort(arr: List[int], sorted_arr: List[int]) -> bool:
    """Verify that the sorted array is correct."""
    return sorted_arr == sorted(arr)


def print_algorithm_info():
    """Print information about all sorting algorithms."""
    algorithms = [
        ("Bubble Sort", "O(n²)", "O(1)", "Yes"),
        ("Selection Sort", "O(n²)", "O(1)", "No"),
        ("Insertion Sort", "O(n²)", "O(1)", "Yes"),
        ("Merge Sort", "O(n log n)", "O(n)", "Yes"),
        ("Quick Sort", "O(n log n)*", "O(log n)", "No"),
        ("Heap Sort", "O(n log n)", "O(1)", "No"),
        ("Counting Sort", "O(n + k)", "O(n + k)", "Yes"),
        ("Radix Sort", "O(nk)", "O(n + k)", "Yes"),
        ("Bucket Sort", "O(n + k)*", "O(n + k)", "Yes"),
        ("Shell Sort", "O(n^1.5)", "O(1)", "No"),
    ]
    
    print("\n" + "=" * 70)
    print(" SORTING ALGORITHMS COMPARISON")
    print("=" * 70)
    print(f"{'Algorithm':<18} {'Avg Time':<14} {'Space':<12} {'Stable':<8}")
    print("-" * 70)
    for name, time, space, stable in algorithms:
        print(f"{name:<18} {time:<14} {space:<12} {stable:<8}")
    print("=" * 70)
    print("* Quick Sort worst case: O(n²), Bucket Sort worst case: O(n²)")
    print("  k = range of input for Counting/Radix, number of buckets for Bucket")


# =============================================================================
# DEMONSTRATION
# =============================================================================
if __name__ == "__main__":
    import random
    
    # Test array
    test_array = [64, 34, 25, 12, 22, 11, 90, 5, 77, 30]
    print(f"Original array: {test_array}")
    print()
    
    # Test all sorting algorithms
    algorithms = {
        "Bubble Sort": bubble_sort,
        "Selection Sort": selection_sort,
        "Insertion Sort": insertion_sort,
        "Merge Sort": merge_sort,
        "Quick Sort": quick_sort,
        "Heap Sort": heap_sort,
        "Counting Sort": counting_sort,
        "Radix Sort": radix_sort,
        "Bucket Sort": bucket_sort,
        "Shell Sort": shell_sort,
    }
    
    print("Testing all sorting algorithms:")
    print("-" * 50)
    
    all_passed = True
    for name, func in algorithms.items():
        result = func(test_array)
        passed = verify_sort(test_array, result)
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:<18}: {result} {status}")
        if not passed:
            all_passed = False
    
    print("-" * 50)
    print(f"\nAll tests passed: {all_passed}")
    
    # Test with negative numbers
    print("\n" + "=" * 50)
    print("Testing with negative numbers:")
    print("=" * 50)
    test_negative = [-5, 3, -1, 0, -8, 7, 2, -3]
    print(f"Original: {test_negative}")
    
    for name, func in algorithms.items():
        result = func(test_negative)
        passed = verify_sort(test_negative, result)
        status = "✓" if passed else "✗"
        print(f"{name:<18}: {result} {status}")
    
    # Test with random large array
    print("\n" + "=" * 50)
    print("Performance test with 1000 random elements:")
    print("=" * 50)
    
    import time
    
    random_array = [random.randint(-1000, 1000) for _ in range(1000)]
    
    for name, func in algorithms.items():
        start = time.perf_counter()
        result = func(random_array)
        end = time.perf_counter()
        passed = verify_sort(random_array, result)
        print(f"{name:<18}: {(end - start) * 1000:.3f} ms {'✓' if passed else '✗'}")
    
    # Print algorithm comparison
    print_algorithm_info()
