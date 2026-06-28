"""Compatibility shim — the implementation moved into the ``cembedding`` package.

This file keeps ``python server.py`` working for the git-clone / development
install path (and preserves that path's entry point). The real server lives in
``cembedding/server.py``; for PyPI / uvx use the ``cembedding`` console script
or ``python -m cembedding``.

Running ``python server.py`` from the repository root puts the root on
``sys.path``, so ``import cembedding`` resolves to the package directory beside
this file.
"""

from cembedding.server import run

if __name__ == "__main__":
    run()
