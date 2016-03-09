"""Integration tests for the neutrinosimulation module"""

import filecmp
import shutil
import os
import h5py
import numpy as np
from neutrinosimulation import NeutrinoSemiLinearSim, NeutrinoPartSim

def test_neutrino_part():
    """Create a full simulation with particle neutrinos."""
    test_dir = "./test_nu/"
    Sim = NeutrinoPartSim(test_dir,box = 256,npart = 256, m_nu = 0.45, redshift = 99, redend=0, separate_gas=False)
    Sim.make_simulation()
    assert os.path.exists(test_dir)
    #Check these files have not changed
    for f in ("_genic_params.ini",):
        assert filecmp.cmp(os.path.join(test_dir,f), os.path.join("./testdata/test_nu/",f))
    #Check that the output has neutrino particles
    f = h5py.File(os.path.join(test_dir,"ICS/256_256_99.0.hdf5"),'r')
    assert f["Header"].attrs["NumPart_Total"][2] == 256**3
    #Clean the test directory if test was successful
    #Check the mass is correct
    mcdm = f["Header"].attrs["MassTable"][1]
    mnu = f["Header"].attrs["MassTable"][2]
    #The mass ratio should be given by the ratio of omega_nu by omega_cdm
    assert np.abs(mnu/mcdm / ( Sim.omeganu/(Sim.omegac+Sim.omegab)) - 1) < 1e-5
    assert np.abs(f["Header"].attrs["MassTable"][1] / 7.17244023 - 1) < 1e-5
    f.close()
    #shutil.rmtree("./test_nu/")

def test_neutrino_semilinear():
    """Create a full simulation with semi-linear neutrinos.
    The important thing here is to test that OmegaNu is correctly set."""
    test_dir = "./test_nu_semilin/"
    Sim = NeutrinoSemiLinearSim(test_dir,box = 256,npart = 256, m_nu = 0.45, redshift = 99, redend=0, separate_gas=False)
    Sim.make_simulation()
    assert os.path.exists(test_dir)
    #Check these files have not changed
    for f in ("_camb_params.ini", "_genic_params.ini", "gadget3.param"):
        assert filecmp.cmp(os.path.join(test_dir,f), os.path.join("./testdata/test_nu_semilin/",f))
    #Check that the output has no neutrino particles
    f = h5py.File(os.path.join(test_dir, "ICS/256_256_99.0.hdf5"),'r')
    assert f["Header"].attrs["NumPart_Total"][2] == 0
    #Check the mass is correct: the CDM particles should have the same mass as in the particle simulation
    assert np.abs(f["Header"].attrs["MassTable"][1] / 7.17244023 - 1) < 1e-5
    f.close()
    #shutil.rmtree("./test_nu/")
