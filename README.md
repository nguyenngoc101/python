# 🔄 Sorting Algorithms Collection

A comprehensive Python implementation of **10 popular sorting algorithms** with detailed documentation, complexity analysis, and benchmarks.

## 📋 Algorithms Included

| Algorithm | Time Complexity (Avg) | Space | Stable | Best For |
|-----------|----------------------|-------|--------|----------|
| **Bubble Sort** | O(n²) | O(1) | ✓ | Small datasets, nearly sorted data |
| **Selection Sort** | O(n²) | O(1) | ✗ | Small datasets, memory-constrained |
| **Insertion Sort** | O(n²) | O(1) | ✓ | Small/nearly sorted datasets |
| **Merge Sort** | O(n log n) | O(n) | ✓ | Large datasets, stable sort needed |
| **Quick Sort** | O(n log n) | O(log n) | ✗ | General purpose, large datasets |
| **Heap Sort** | O(n log n) | O(1) | ✗ | Memory-constrained, guaranteed O(n log n) |
| **Counting Sort** | O(n + k) | O(n + k) | ✓ | Integer data with small range |
| **Radix Sort** | O(nk) | O(n + k) | ✓ | Fixed-length integers/strings |
| **Bucket Sort** | O(n + k) | O(n + k) | ✓ | Uniformly distributed data |
| **Shell Sort** | O(n^1.5) | O(1) | ✗ | Medium datasets, in-place needed |

## 🚀 Quick Start

### Run the Demo
```bash
python3 sorting_algorithms.py
```

### Use in Your Code
```python
from sorting_algorithms import quick_sort, merge_sort, heap_sort

# Sort an array
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = quick_sort(arr)
print(sorted_arr)  # [11, 12, 22, 25, 34, 64, 90]

# Works with negative numbers too
arr = [-5, 3, -1, 0, 7, -3]
sorted_arr = merge_sort(arr)
print(sorted_arr)  # [-5, -3, -1, 0, 3, 7]
```

## 📊 Performance Benchmarks

Results from sorting 1000 random integers:

```
Algorithm          Time (ms)    
─────────────────────────────────
Counting Sort      0.27   ⚡ Fastest
Bucket Sort        0.35
Quick Sort         0.51
Radix Sort         0.53
Shell Sort         0.81
Heap Sort          1.44
Merge Sort         1.66
Insertion Sort     12.19
Selection Sort     12.18
Bubble Sort        28.18  🐢 Slowest
```

## ✨ Features

- **Non-mutating**: All functions return a new sorted array without modifying the original
- **Negative number support**: All algorithms handle negative integers correctly
- **Well-documented**: Each function includes detailed docstrings with complexity analysis
- **Type hints**: Full Python type annotations for better IDE support
- **Verification utilities**: Built-in functions to verify sorting correctness

## 📁 File Structure

```
python/
├── sorting_algorithms.py   # All 10 sorting algorithms
└── README.md               # This file
```

## 🔧 Requirements

- Python 3.6+
- No external dependencies

## 📖 Algorithm Details

### Comparison-Based Sorts
- **Bubble Sort**: Simple but inefficient; good for educational purposes
- **Selection Sort**: Minimizes swaps; useful when writes are expensive
- **Insertion Sort**: Excellent for small or nearly sorted arrays
- **Merge Sort**: Guaranteed O(n log n); requires extra space
- **Quick Sort**: Fastest in practice; uses median-of-three pivot selection
- **Heap Sort**: In-place with guaranteed O(n log n) performance
- **Shell Sort**: Improved insertion sort with gap sequences (Knuth's sequence)

### Non-Comparison Sorts
- **Counting Sort**: Linear time for integers with bounded range
- **Radix Sort**: Sorts digit by digit; great for fixed-length keys
- **Bucket Sort**: Distributes elements into buckets; ideal for uniform distributions

## 📄 License

MIT License - feel free to use in your projects!
