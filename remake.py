"""This module rebuilds the Gadget binary for all runs in a directory.
It must be very compatible, as it will run on the cluster.
Requires python 2.5."""

import glob
import filecmp
import subprocess
import shutil
import os
import os.path as path

def rebuild(rundir, codedir, config_file="Config.sh", binary="P-Gadget3"):
    """Rebuild all Gadget binaries in subdirectories of rundir.
    Arguments:
    rundir: Parent of simulation directories
    codedir: Location of the Makefile.
    binary: name of file to rebuild.
    config_file: Name of configuration file which specifies compile flags. Should be within the rundir."""
    #Find all subdirs with config files.
    configs = glob.glob(path.join(path.join(rundir, "*"),config_file))
    #First run.
    first = True
    for cc in configs:
        directory = path.dirname(cc)
        #Is the config file already there identical to the next compile?
        #If so, don't bother recompiling.
        if first or not filecmp.cmpfiles(codedir, directory, config_file):
            #Link in new config file
            codeconf = path.join(codedir, config_file)
            #Remove the old link if present and if it is a symlink.
            if path.exists(codeconf):
                if path.islink(codeconf):
                    os.remove(codeconf)
                else:
                    raise OSError("File:",codeconf," exists, and is not symlink. Not deleting")
            #Make symlink
            os.symlink(cc, path.join(codedir,config_file))
            #check_output is new in python 2.7, so not used.
            make_retcode = subprocess.call(["make", "-j4"], cwd=codedir)
            if make_retcode:
                raise RuntimeError("make failed on ",cc)
            first = False
        #Note that if dst is a symlink, this will overwrite the contents
        #of the symlink instead of breaking it.
        shutil.copy2(path.join(codedir, binary), path.join(directory, binary))
    return configs
