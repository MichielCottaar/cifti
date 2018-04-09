#!/bin/bash

pip install numpy nibabel>=2.2 six nose coverage
git clone https://github.com/demianw/nibabel-nitest-cifti2.git
mv nibabel-nitest-cifti2 nitest-cifti2
