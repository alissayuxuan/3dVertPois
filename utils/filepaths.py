from __future__ import annotations

import os
import warnings
from itertools import chain
from pathlib import Path


def search_path(basepath: str | Path, query: str, verbose: bool = False, suppress: bool = False) -> list[Path]:
    """Searches from basepath with query
    Args:
        basepath: ground path to look into
        query: search query, can contain wildcards like *.npz or **/*.npz
        verbose:
        suppress: if true, will not throwing warnings if nothing is found

    Returns:
        All found paths
    """
    basepath = str(basepath)
    assert os.path.exists(basepath), f"basepath for search_path() doesnt exist, got {basepath}"
    if not basepath.endswith("/"):
        basepath += "/"
    print(f"search_path: in {basepath}{query}") if verbose else None
    paths = sorted(list(chain(list(Path(f"{basepath}").glob(f"{query}")))))
    if len(paths) == 0 and not suppress:
        warnings.warn(f"did not find any paths in {basepath}{query}", UserWarning)
    return paths


def search_path_single(basepath: str | Path, query: str, verbose: bool = False) -> Path:
    """Searches from basepath with query
    Args:
        basepath: ground path to look into
        query: search query, can contain wildcards like *.npz or **/*.npz
        verbose:

    Returns:
        First found path
    """
    paths = search_path(basepath, query, verbose=verbose)
    if len(paths) == 0:
        raise FileNotFoundError(f"did not find any paths in {basepath}{query}")
    if len(paths) > 1:
        raise FileExistsError(f"found multiple paths in {basepath}{query}, got {paths}")
    return paths[0]
