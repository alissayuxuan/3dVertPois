# Choose a license before publishing

This repository currently has **no license**, which means nobody may legally reuse it.
Pick one before making the repository public.

## What constrains the choice

This code is a derivative of
<https://github.com/doppelplusungut/3dVertPois> (Daniel-Jordi Regenbrecht's master's
thesis code). **Check that repository's license first** - a derivative work cannot be
released under terms more permissive than its source. If it carries no license at all,
you need the author's explicit permission before publishing this.

It also vendors code adapted from [MONAI](https://github.com/Project-MONAI/MONAI)
(Apache-2.0) in `src/vertpois/models/densenet.py` and `subm_densenet.py`, and depends on
[TPTBox](https://github.com/Hendrik-code/TPTBox) (Apache-2.0). Apache-2.0 requires that
you retain the attribution already present in those files' module docstrings.

## Suggested

Apache-2.0, matching TPTBox and MONAI, if the upstream repository permits it. Then:

1. download the full text from <https://www.apache.org/licenses/LICENSE-2.0.txt> and save
   it as `LICENSE` (the full text is required - a short notice is not enough);
2. confirm `license = "Apache-2.0"` in `pyproject.toml` still matches your choice;
3. delete this file.
