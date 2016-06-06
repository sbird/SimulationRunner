"""This module rebuilds the Gadget binary for all runs in a directory.
It must be very compatible, as it will run on the cluster.
Requires python 2.6."""

from __future__ import print_function
import glob
import filecmp
import subprocess
import shutil
import re
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
    rundir = path.expanduser(rundir)
    codedir = path.expanduser(codedir)
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

def resub(rundir, script_file="mpi_submit", submit_command="qsub"):
    """Submit all jobs in the emulator to the queueing system"""
    #Find all subdirs with config files.
    rundir = path.expanduser(rundir)
    configs = glob.glob(path.join(path.join(rundir, "*"),script_file))
    for cc in configs:
        cdir = path.dirname(cc)
        subprocess.call([submit_command, script_file], cwd=cdir)

def _check_single_status(fname, endz):
    """Given a file, check whether it shows the
        simulation reached the desired redshift."""
    #Get the last line of the file:
    #need to open in binary to get negative seeks fom the end.
    with open(fname, 'rb') as fh:
        #Start at the end and seek backwards until we find a newline.
        fh.seek(-2,os.SEEK_END)
        while fh.read(1) != b'\n':
            fh.seek(-2,os.SEEK_CUR)
        #This should be before the final redshift.
        last = fh.readline().decode()
    #Parse it to find the redshift
    match = re.search(r"Redshift: ([0-9]{1,3}\.?[0-9]*)",last)
    redshift = float(match.group(1))
    return redshift <= endz

def check_status(rundir, output_file="output/info.txt", endz=2):
    """Get completeness status for every directory in the suite.
    Ultimately this should work out whether there
    was an error or just a timeout."""
    rundir = path.expanduser(rundir)
    outputs = glob.glob(path.join(path.join(rundir, "*"),output_file))
    completes = [_check_single_status(cc, endz) for cc in outputs]
    return outputs, completes

def print_status(rundir, output_file="output/info.txt", endz=2):
    """Get completeness status for every directory in the suite.
    Ultimately this should work out whether there
    was an error or just a timeout."""
    outputs, completes = check_status(rundir, output_file, endz)
    for oo, cc in zip(outputs, completes):
        print(oo[len(rundir):-len(output_file)-1]," : ",end="")
        if not cc:
            print("NOT ",end="")
        print("COMPLETE")

def resub_not_complete(rundir, output_file="output/info.txt", endz=2, script_file="mpi_submit", resub_command="qsub", paramfile="gadget3.param"):
    """Resubmit incomplete simulations to the queue.
    We also edit the script file to add a RestartFlag"""
    outputs, completes = check_status(rundir, output_file, endz)
    #Pathnames for incomplete simulations
    for oo,cc in zip(outputs,completes):
        if cc:
            continue
        #Remove the output_file from the output, getting the directory.
        odir = oo[:-len(output_file)]
        script_file_resub = script_file+"_resub"
        with open(path.join(odir, script_file),'r') as ifile:
            with open(path.join(odir, script_file_resub),'w') as ofile:
                line = ifile.readline()
		while line != '':
                    #Find the actual submission line and add a '1' after the paramfile.
                    if re.search("mpirun|mpiexec", line):
                        line = re.sub(paramfile, paramfile+" 1",line)
                    #Write each line straight through to the output by default.
                    ofile.write(line)
                    line = ifile.readline()
        print("Re-submitting: ",path.join(odir, script_file_resub))
        subprocess.call([resub_command, script_file_resub], cwd=odir)





