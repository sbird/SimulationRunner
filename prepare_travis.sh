#!/bin/bash
#Script to fetch and build dependencies for the tests, for travis.

wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b -p $HOME/miniconda
export PATH=$HOME/miniconda/bin:$PATH
conda update --yes conda
conda create --yes -n test python=3.6
source $HOME/miniconda/bin/activate test
conda install --yes -c bccp nbodykit matplotlib numpy bigfile
conda install --yes gfortran_linux-64 nose configobj scipy
cd $HOME/miniconda/envs/test/bin/
#Set up a symlink to gfortran
ln -s x86_64-conda_cos6-linux-gnu-gfortran gfortran
cd -
#Clone stuff.
mkdir tests
