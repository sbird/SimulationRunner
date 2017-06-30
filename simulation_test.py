"""Integration tests for the Simulation module"""

import filecmp
# import shutil
import os
import re
import configobj
from SimulationRunner import simulationics

def test_full_integration():
    """Create a full simulation snapshot and check it corresponds to the saved results"""
    defaultpath = os.path.join(os.path.dirname(__file__), "tests/test1")
    Sim = simulationics.SimulationICs(outdir=defaultpath,box = 256,npart = 256, redshift = 99, code_args={'redend':0})
    Sim.make_simulation()
    #Check the following files were created
    assert os.path.exists(defaultpath)
    for ff in ("times.txt", "TREECOOL", "mpi_submit", "camb_linear", "ICS", "output", "camb_linear/ics_matterpow_99.dat", "ICS/PK-DM-256_256_99", "SimulationICs.json"):
        assert os.path.exists(os.path.join(defaultpath, ff))
    #Check these files have not changed
    testdatadir = os.path.join(os.path.dirname(__file__),"testdata/test1/")
    for f in ("_camb_params.ini", "_genic_params.ini"):
        config_new = configobj.ConfigObj(os.path.join(defaultpath,f))
        config_old = configobj.ConfigObj(os.path.join(testdatadir,f))
        for key in config_old.keys():
            if re.match("/home/spb",config_old[key]):
                continue
            assert config_old[key] == config_new[key]
    for f in ("Config.sh", "gadget3.param"):
        assert filecmp.cmp(os.path.join(defaultpath,f), os.path.join(testdatadir,f))
    #Clean the test directory if test was successful
    #shutil.rmtree(defaultpath)

def test_only_DM():
    """Create a full simulation with no gas"""
    outdir = os.path.join(os.path.dirname(__file__),"tests/test2")
    Sim = simulationics.SimulationICs(outdir=outdir, box = 256, npart = 256, redshift = 99, separate_gas=False, code_args={'redend':0},hubble=0.71)
    Sim.make_simulation()
    assert os.path.exists(outdir)

    Sim2 = simulationics.SimulationICs(outdir=outdir, box=128, npart=256)
    Sim2.load_txt_description()
    assert Sim2.box == Sim.box
    assert Sim2.hubble == Sim.hubble
    assert Sim.code_class_name is Sim2.code_class_name
    assert Sim.code_args == Sim2.code_args
    #shutil.rmtree(outdir)
