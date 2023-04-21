# Lightning Sample project/package

This is starter project template which shall simplify initial steps for each new PL project...

[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/)
[![Build Status](https://dev.azure.com/Lightning-AI/compatibility/_apis/build/status%2Faccelerators%2FLightning-AI.lightning-Graphcore?branchName=main)](https://dev.azure.com/Lightning-AI/compatibility/_build/latest?definitionId=48&branchName=main)
[![General checks](https://github.com/Lightning-AI/lightning-graphcore/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-graphcore/actions/workflows/ci-checks.yml)
[![Documentation Status](https://readthedocs.org/projects/lightning-graphcore/badge/?version=latest)](https://lightning-graphcore.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-graphcore/main.svg?badge_token=mqheL1-cTn-280Vx4cJUdg)](https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-graphcore/main?badge_token=mqheL1-cTn-280Vx4cJUdg)

\* the Read-The-Docs is failing as this one leads to the public domain which requires the repo to be public too

## To be Done aka cross-check

You still need to enable some external integrations such as:

- [ ] lock the main breach in GH setting - no direct push without PR
- [ ] init Read-The-Docs (add this new project)
- [ ] add credentials for releasing package to PyPI

## Tests / Docs notes

- We are using [Napoleon style,](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) and we shall use static types...
- It is nice to se [doctest](https://docs.python.org/3/library/doctest.html) as they are also generated as examples in documentation
- For wider and edge cases testing use [pytest parametrization](https://docs.pytest.org/en/stable/parametrize.html) :\]
