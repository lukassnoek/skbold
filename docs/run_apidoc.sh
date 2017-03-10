#!/usr/bin/env bash
rm source/*.rst
sphinx-apidoc -o source -fMeT ../skbold ../skbold/*/tests ../skbold/*/*/tests ../skbold/*/*/*/tests ../skbold/data
