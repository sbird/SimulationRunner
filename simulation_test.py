"""Integration tests for the Simulation module"""

import filecmp
import shutil
import os
from simulation import Simulation

def test_full_integration():
    """Create a full simulation snapshot and check it corresponds to the saved results"""
    Sim = Simulation("./test1",box = 256,npart = 256, redshift = 99, redend=0)
    Sim.make_simulation()
    #Check the following files were created
    assert os.path.exists("./test1")
    for ff in ("times.txt", "TREECOOL", "mpi_submit", "camb_linear", "ICS", "output", "camb_linear/ics_matterpow_99.dat", "ICS/PK-DM-256_256_99", "Simulation.json"):
        assert os.path.exists(os.path.join("./test1/", ff))
    #Check these files have not changed
    for f in ("_camb_params.ini", "_genic_params.ini", "Config.sh", "gadget3.param"):
        assert filecmp.cmp(os.path.join("./test1/",f), os.path.join("./testdata/test1/",f))
    #Clean the test directory if test was successful
    #shutil.rmtree("./test1/")

def test_only_DM():
    """Create a full simulation with no gas"""
    Sim = Simulation("./test2",box = 256,npart = 256, redshift = 99, redend=0, separate_gas=False)
    Sim.make_simulation()
    assert os.path.exists("./test2")
