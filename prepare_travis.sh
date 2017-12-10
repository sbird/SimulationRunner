#!/bin/bash
#Script to fetch and build dependencies for the tests, for travis.

mkdir tests
mkdir depends
cd depends
##Get and make CAMB.
git clone https://github.com/sbird/camb.git
cd camb
cd pycamb
alias gfortran=gfortran-6
#Maybe this works?
cat "alias gfortran=gfortran-6" >> ~/.bashrc
python3 setup.py install --user

#Get and make N-GenIC.
cd ../
git clone https://github.com/sbird/S-GenIC.git
cd S-GenIC
git submodule update --init --recursive
make

#Get and make GenPK.
cd ../
git clone https://github.com/sbird/GenPK.git
cd GenPK
git submodule update --init --recursive
make
