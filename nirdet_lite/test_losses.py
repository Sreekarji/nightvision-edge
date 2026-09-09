"""
test_losses.py — NIRDetLoss self-test runner.

Runs the five built-in self-tests from losses.py's __main__ block:
  T1  empty GT          -> no regression loss, finite total
  T2  loss direction    -> matched predictions < random baseline
  T3  NaN/Inf guard     -> 25 random batches all finite
  T4  collision         -> smallest-area GT wins
  T5  geometry          -> grid sizes derived from resolution
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Import the private test helpers from losses
import losses as _losses_mod

def test_losses():
    print("=" * 62)
    print("  losses.py self-tests (via test_losses.py)")
    print("=" * 62)

    _losses_mod._t_empty()
    _losses_mod._t_direction()
    _losses_mod._t_nan()
    _losses_mod._t_collision()
    _losses_mod._t_resolution()

    print("=" * 62)
    print("PASS")
    print("=" * 62)


if __name__ == "__main__":
    test_losses()
