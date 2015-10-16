"""Integration tests for the Simulation module"""

from simulation import Simulation
import filecmp
import shutil
import os

def test_full_integration():
    """Create a full simulation snapshot and check it corresponds to the saved results"""
    #First remove the test directory
    Sim = Simulation("./test1",box = 256,npart = 256, redshift = 99, redend=0)
    Sim.make_simulation()
    #Check the following files were created
    assert os.path.exists("./test1")
    for ff in ("times.txt", "TREECOOL", "mpi_submit", "camb_linear", "ICS", "output", "camb_linear/ics_matterpow_99.dat", "ICS/PK-DM-256_256_99", "Simulation.json"):
        assert os.path.exists(os.path.join("./test1/", ff))
    #Check these files have not changed
    for f in ("_camb_params.ini", "_genic_params.ini", "Config.sh", "gadget3.param"):
        assert filecmp.cmp(os.path.join("./test1/",f), os.path.join("./testdata/test1/",f))
    #shutil.rmtree("./test1/")
