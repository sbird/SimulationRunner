"""Integration tests for the Simulation module"""

import os
import re
import configobj
from SimulationRunner import simulationics

def test_full_integration():
    """Create a full simulation snapshot and check it corresponds to the saved results"""
    defaultpath = os.path.join(os.path.dirname(__file__), "tests/test1")
    Sim = simulationics.SimulationICs(outdir=defaultpath,box = 256,npart = 128, redshift = 99, code_args={'redend':0})
    Sim.make_simulation()
    #Check the following files were created
    assert os.path.exists(defaultpath)
    for ff in ("_camb_params.ini", "TREECOOL", "mpi_submit", "camb_linear", "ICS", "output", "camb_linear/ics_matterpow_99.dat", "SimulationICs.json", "mpgadget.param", "_genic_params.ini"):
        assert os.path.exists(os.path.join(defaultpath, ff))
    #Clean the test directory if test was successful
    #shutil.rmtree(defaultpath)

def test_only_DM():
    """Create a full simulation with no gas"""
    outdir = os.path.join(os.path.dirname(__file__),"tests/test2")
    Sim = simulationics.SimulationICs(outdir=outdir, box = 256, npart = 128, redshift = 99, separate_gas=False, code_args={'redend':0},hubble=0.71)
    Sim.make_simulation()
    assert os.path.exists(outdir)

    Sim2 = simulationics.SimulationICs(outdir=outdir, box=128, npart=128)
    Sim2.load_txt_description()
    assert Sim2.box == Sim.box
    assert Sim2.hubble == Sim.hubble
    assert Sim.code_class_name is Sim2.code_class_name
    assert Sim.code_args == Sim2.code_args
    #shutil.rmtree(outdir)
