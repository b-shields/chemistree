"""Benchmark harness for chemistree.

Runs headless Claude Code once per case, in an isolated context session, and
records per-case metrics (token footprint, cost, turns) and the final molecule.
See ``README.md`` for the design.
"""
