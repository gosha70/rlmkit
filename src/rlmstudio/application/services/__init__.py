"""Application services — pure functions and stateless helpers.

Modules in this package must not depend on FastAPI, AppState, or any
infrastructure adapter.  They are unit-testable without I/O and are
called from the ``server/routes/`` layer after the HTTP request has
been translated into plain data structures.
"""
