#!/usr/bin/env python3
import math


def main() -> None:
    n = 8
    total = 0
    for k in range(1, n + 1):
        c = math.comb(n, k)
        total += c
        print(f"k={k}: {c} (cumulative={total})")
    print(f"Total (1..{n}): {total}")
    print(f"Total (2..{n}): {total - math.comb(n, 1)}")


if __name__ == "__main__":
    main()
