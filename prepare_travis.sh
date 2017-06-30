#!/bin/bash
#Script to fetch and build dependencies for the tests, for travis.

mkdir tests
mkdir depends
cd depends
##Get and make CAMB.
git clone https://github.com/sbird/camb.git
cd camb
make camb

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
