"""Integration tests for the neutrinosimulation module"""

import filecmp
import shutil
import os
import h5py
import numpy as np
import configobj
from . import neutrinosimulation as nus

def test_neutrino_part():
    """Create a full simulation with particle neutrinos."""
    test_dir = "./test_nu/"
    Sim = nus.NeutrinoPartSim(outdir=test_dir,box = 256,npart = 256, m_nu = 0.45, redshift = 99, redend=0, separate_gas=False, do_build=False)
    Sim.make_simulation()
    assert os.path.exists(test_dir)
    #Check these we are writing reasonable values.
    config = configobj.ConfigObj(os.path.join(test_dir,"_genic_params.ini"))
    assert abs(float(config['OmegaDM_2ndSpecies']) - 0.009860074585986426) < 1e-7
    assert config['Omega'] == "0.288"
    assert config['OmegaLambda'] == "0.712"
    assert config['NNeutrino'] == "256"
    assert config['NU_KSPACE'] == "0"
    assert config['NU_On'] == "1"
    assert config['NU_Vtherm_On'] == "1"
    assert config['NU_PartMass_in_ev'] == "0.45"
    #Check that the output has neutrino particles
    f = h5py.File(os.path.join(test_dir,"ICS/256_256_99.0.hdf5"),'r')
    assert f["Header"].attrs["NumPart_Total"][2] == 256**3
    #Clean the test directory if test was successful
    #Check the mass is correct
    mcdm = f["Header"].attrs["MassTable"][1]
    mnu = f["Header"].attrs["MassTable"][2]
    #The mass ratio should be given by the ratio of omega_nu by omega_cdm
    assert np.abs(mnu/mcdm / ( Sim.omeganu/(Sim.omegac+Sim.omegab)) - 1) < 1e-5
    assert np.abs(f["Header"].attrs["MassTable"][1] / 7.71977292 - 1) < 1e-5
    f.close()
    #shutil.rmtree("./test_nu/")

def test_neutrino_semilinear():
    """Create a full simulation with semi-linear neutrinos.
    The important thing here is to test that OmegaNu is correctly set."""
    test_dir = "./test_nu_semilin/"
    Sim = nus.NeutrinoSemiLinearSim(outdir=test_dir,box = 256,npart = 256, m_nu = 0.45, redshift = 99, redend=0, separate_gas=False, do_build=False)
    Sim.make_simulation()
    assert os.path.exists(test_dir)
    #Check these files have not changed
    config = configobj.ConfigObj(os.path.join(test_dir,"_genic_params.ini"))
    assert abs(float(config['OmegaDM_2ndSpecies']) - 0.009860074585986426) < 1e-7
    assert config['Omega'] == "0.288"
    assert config['OmegaLambda'] == "0.712"
    assert config['NNeutrino'] == "0"
    assert config['NU_KSPACE'] == "0"
    assert config['NU_On'] == "0"
    assert config['NU_Vtherm_On'] == "1"
    assert config['NU_PartMass_in_ev'] == "0"

    config = configobj.ConfigObj(os.path.join(test_dir,"_camb_params.ini"))
    assert abs(float(config['ombh2']) - 0.023127999999999996) < 1e-7
    assert abs(float(config['omch2']) - 0.11316056345286662) < 1e-7
    assert abs(float(config['omnuh2']) - 0.004831436547133348) < 1e-7
    assert config['massless_neutrinos'] == "0.046"
    assert config['massive_neutrinos'] == "3"

    test_files = ("gadget3.param",)
    match, mismatch, errors = filecmp.cmpfiles(test_dir, "./testdata/test_nu_semilin/",test_files)
    assert len(errors) == 0
    assert len(mismatch) == 0
    #Check that the output has no neutrino particles
    f = h5py.File(os.path.join(test_dir, "ICS/256_256_99.0.hdf5"),'r')
    assert f["Header"].attrs["NumPart_Total"][2] == 0
    #Check the mass is correct: the CDM particles should have the same mass as in the particle simulation
    assert np.abs(f["Header"].attrs["MassTable"][1] / 7.71977292 - 1) < 1e-5
    f.close()
    #shutil.rmtree("./test_nu/")
