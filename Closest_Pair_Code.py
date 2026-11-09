"""Closest Pair of Points: brute force vs. O(n log n) divide and conquer.

Run: python closest_pair_demo.py
Both algorithms return two closest points, the Euclidean distance,
and the number of squared-distance evaluations.
"""

from math import sqrt
from random import Random
from time import perf_counter


def distance_squared(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def brute_force(points):
    """Check all n(n-1)/2 pairs. Time O(n^2), extra space O(1)."""
    if len(points) < 2:
        raise ValueError("Provide at least two points")

    best = float("inf")
    closest = None
    comparisons = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            comparisons += 1
            candidate = distance_squared(points[i], points[j])
            if candidate < best:
                best = candidate
                closest = (points[i], points[j])
    return closest, sqrt(best), comparisons


def divide_and_conquer(points):
    """Presort by x and y. Time O(n log n), extra space O(n)."""
    if len(points) < 2:
        raise ValueError("Provide at least two points")

    # Unique IDs also make duplicate coordinates safe when splitting by membership.
    tagged = [(x, y, index) for index, (x, y) in enumerate(points)]
    by_x = sorted(tagged, key=lambda p: (p[0], p[1], p[2]))
    by_y = sorted(tagged, key=lambda p: (p[1], p[0], p[2]))
    comparisons = 0

    def checked_distance(p, q):
        nonlocal comparisons
        comparisons += 1
        return distance_squared(p, q)

    def solve(px, py):
        n = len(px)
        if n <= 3:
            best = (float("inf"), None, None)
            for i in range(n):
                for j in range(i + 1, n):
                    candidate = (checked_distance(px[i], px[j]), px[i], px[j])
                    if candidate[0] < best[0]:
                        best = candidate
            return best

        mid = n // 2
        left_x, right_x = px[:mid], px[mid:]
        middle_x = right_x[0][0]
        left_ids = {p[2] for p in left_x}
        left_y = [p for p in py if p[2] in left_ids]
        right_y = [p for p in py if p[2] not in left_ids]

        best_left = solve(left_x, left_y)
        best_right = solve(right_x, right_y)
        best = best_left if best_left[0] <= best_right[0] else best_right

        # py is already sorted by y; no extra O(n log n) sort is needed here.
        strip = [p for p in py if (p[0] - middle_x) ** 2 < best[0]]
        for i in range(len(strip)):
            # Geometry guarantees we need at most the next 7 y-neighbors.
            for j in range(i + 1, min(i + 8, len(strip))):
                if (strip[j][1] - strip[i][1]) ** 2 >= best[0]:
                    break
                candidate = checked_distance(strip[i], strip[j])
                if candidate < best[0]:
                    best = (candidate, strip[i], strip[j])
        return best

    best_squared, first, second = solve(by_x, by_y)
    pair = ((first[0], first[1]), (second[0], second[1]))
    return pair, sqrt(best_squared), comparisons


def sample_points(n, seed=42):
    """Fixed seed makes each experiment reproducible."""
    rng = Random(seed)
    return [(rng.randrange(100_000), rng.randrange(100_000)) for _ in range(n)]


def demonstrate():
    example = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
    print("EXAMPLE: 6 points")
    for title, method in (("Brute force", brute_force), ("Divide & conquer", divide_and_conquer)):
        pair, distance, checks = method(example)
        print(f"{title:18} closest={pair}, distance={distance:.5f}, comparisons={checks}")

    print("\nBONUS: TIME / COMPARISON DEMO (seed 42)")
    print(f"{'Points':>7}  {'Method':<17} {'Distance checks':>16} {'Time (ms)':>12}")
    print("-" * 59)
    for n in (100, 1000, 3000):
        points = sample_points(n)
        expected = None
        for title, method in (("Brute force", brute_force), ("Divide & conquer", divide_and_conquer)):
            start = perf_counter()
            pair, distance, checks = method(points)
            elapsed_ms = (perf_counter() - start) * 1000
            if expected is None:
                expected = distance
            else:
                assert abs(expected - distance) < 1e-9, "Algorithms disagree!"
            print(f"{n:7}  {title:<17} {checks:16,} {elapsed_ms:12.3f}")


if __name__ == "__main__":
    demonstrate()
