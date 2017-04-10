"""Integration tests for the MP-Simulation: we remake the ICs, because the N-GenIC output format changed."""

import os
import re
import configobj
from . import simulationics
from . import mpsimulation

def test_full_integration():
    """Create a full simulation snapshot and check it corresponds to the saved results"""
    defaultpath = os.path.join(os.path.dirname(__file__), "test1-mp")
    Sim = simulationics.SimulationICs(outdir=defaultpath,box = 256,npart = 256, redshift = 99, code_class=mpsimulation.MPSimulation, code_args={'redend':0,})
    Sim.make_simulation()
    #Check the following files were created
    assert os.path.exists(defaultpath)
    for ff in ("TREECOOL", "mpi_submit", "camb_linear", "ICS", "output", "camb_linear/ics_matterpow_99.dat", "ICS/PK-DM-256_256_99", "SimulationICs.json"):
        assert os.path.exists(os.path.join(defaultpath, ff))
    #Check these files have not changed
    testdatadir = os.path.join(os.path.dirname(__file__),"testdata/test1-mp/")
    for f in "mpgadget.param":
        config_new = configobj.ConfigObj(os.path.join(defaultpath,f))
        config_old = configobj.ConfigObj(os.path.join(testdatadir,f))
        for key in config_old.keys():
            if re.match("/home/spb",config_old[key]):
                continue
            assert config_old[key] == config_new[key]
    #Clean the test directory if test was successful
    #shutil.rmtree(defaultpath)

def test_only_MP_DM():
    """Create a full simulation with no gas"""
    outdir = os.path.join(os.path.dirname(__file__),"test2-mp")
    Sim = simulationics.SimulationICs(outdir=outdir, box = 256, npart = 256, redshift = 99, separate_gas=False, code_class=mpsimulation.MPSimulation, code_args={'redend':0})
    Sim.make_simulation()
    assert os.path.exists(outdir)
    #shutil.rmtree(outdir)
