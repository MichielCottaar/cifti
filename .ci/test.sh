#!/bin/bash

export NIBABEL_DATA_DIR=`pwd`

nosetests --with-coverage --cover-package=cifti --where=cifti
