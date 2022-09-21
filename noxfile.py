"""Nox sessions for linting, docs, and testing."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import nox

DIR = Path(__file__).parent.resolve()

nox.options.sessions = ["lint", "pylint", "tests"]


@nox.session
def lint(session: nox.Session) -> None:
    """Run the linter.

    Includes all the pre-commit checks on all the files.
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session
def pylint(session: nox.Session) -> None:
    """
    Run PyLint.

    This needs to be installed into the package environment, and is slower
    than a pre-commit check

    """
    session.install(".", "pylint")
    session.run("pylint", "src", *session.posargs)


@nox.session
def tests(session: nox.Session) -> None:
    """Run the unit and regular tests."""
    session.install(".[test]")
    session.run(
        "pytest",
        "--cov=bluebonnet",
        "--cov-append",
        "--cov-report=xml",
        *session.posargs,
    )


@nox.session
def docs(session: nox.Session) -> None:
    """Build the docs. Pass "serve" to serve."""
    session.install(".[docs]")
    session.chdir("docs")
    session.run("sphinx-build", "-M", "html", ".", "_build")

    if session.posargs:
        if "serve" in session.posargs:
            print("Launching docs at http://localhost:8000/ - use Ctrl-C to quit")
            session.run("python", "-m", "http.server", "8000", "-d", "_build/html")
        else:
            session.warn("Unsupported argument to docs")


@nox.session
def build(session: nox.Session) -> None:
    """Build an SDist and a wheel."""
    build_p = DIR.joinpath("build")
    if build_p.exists():
        shutil.rmtree(build_p)

    session.install("build")
    session.run("python", "-m", "build")


@nox.session
def paper(session: nox.Sesson) -> None:
    """Build the JOSS paper draft."""
    paper_dir = DIR.joinpath("paper")
    session.run(
        "docker",
        "run",
        "--rm",
        "--volume",
        f"{paper_dir}:/data",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--env",
        "JOURNAL=joss",
        "openjournals/inara",
    )
