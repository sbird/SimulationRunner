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
import distutils.spawn

def rebuild_MP(rundir, codedir, config_file="Options.mk", binary=None):
    """rebuild, but with defaults appropriate for MP-Gadget."""
    if binary is None:
        binary=["gadget/MP-Gadget", "genic/MP-GenIC"]
    return rebuild(rundir, codedir,config_file=config_file, binary=binary)

def rebuild(rundir, codedir, config_file="Config.sh", binary=None):
    """Rebuild all Gadget binaries in subdirectories of rundir.
    Arguments:
    rundir: Parent of simulation directories
    codedir: Location of the Makefile.
    binary: name of file to rebuild.
    config_file: Name of configuration file which specifies compile flags. Should be within the rundir."""
    if binary is None:
        binary = ["P-Gadget3",]
    #Find all subdirs with config files.
    rundir = path.expanduser(rundir)
    codedir = path.expanduser(codedir)
    configs = glob.glob(path.join(path.join(rundir, "*"),config_file))
    configs += glob.glob(path.join(rundir,config_file))
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
        for bi in binary:
            shutil.copy2(path.join(codedir, bi), path.join(directory, os.path.basename(bi)))
    return configs

def detect_submit():
    """Auto-detect the resubmission command. """
    #Python 3.3 has shutils.which, but we need python 2 compatibility here.
    #Use sbatch if it exists
    if distutils.spawn.find_executable('sbatch') is not None:
        return 'sbatch'
    #Try for qsub
    if distutils.spawn.find_executable('qsub') is not None:
        return 'qsub'
    #Otherwise not sure what to do.
    raise ValueError("Could not find sbatch or qsub")

def resub(rundir, script_file="mpi_submit", submit_command=None):
    """Submit all jobs in the emulator to the queueing system"""
    #Find all subdirs with config files.
    if submit_command is None:
        submit_command = detect_submit()
    rundir = path.expanduser(rundir)
    configs = glob.glob(path.join(path.join(rundir, "*"),script_file))
    for cc in configs:
        cdir = path.dirname(cc)
        subprocess.call([submit_command, script_file], cwd=cdir)

def _check_single_status(fname, regex):
    """Given a file, check whether it shows the
        simulation reached the desired redshift."""
    #Get the last line of the file:
    #need to open in binary to get negative seeks fom the end.
    redshift = 1100.
    fname = sorted(glob.glob(fname))[-1]
    with open(fname, 'rb') as fh:
        match = None
        #Start at the end and seek backwards until we find a newline.
        fh.seek(0,os.SEEK_END)
        while match is None:
            fh.seek(-2,os.SEEK_CUR)
            while fh.read(1) != b'\n':
                if fh.tell() < 2:
                    fh.seek(0)
                    break
                fh.seek(-2,os.SEEK_CUR)
            #Complete line
            last = fh.readline().decode()
            #Seek back to beginning
            fh.seek(-len(last),os.SEEK_CUR)
            #Parse it to find the redshift
            match = re.search(regex, last)
        redshift = float(match.group(1))
        #Convert to z from a if needed
        if re.search("Time", regex):
            redshift = 1./redshift - 1.
    return redshift

def _check_single_status_snap(outdir, output_file, snap="PART_"):
    """Get the final redshift of a simulation from the last written snapshot"""
    try:
        snapnum = _find_snap(outdir, output_file,snap=snap)
    except IOError:
        return 1100
    snapdir = path.join(path.join(outdir,output_file),snap+str(snapnum).rjust(3,'0'))
    return _get_redshift_snapshot(snapdir)

def _get_redshift_snapshot(snapshot):
    """Get the redshift of a BigFile snapshot"""
    fname = os.path.join(snapshot,"Header/attr-v2")
    with open(fname,'r') as fh:
        for line in fh:
            if re.search("Time",line) is not None:
                m = re.search(r"#HUMANE \[ ([\d\.]*) \]",line)
                return 1./float(m.groups()[0])-1
    raise IOError("No redshift in file")

def _find_snap(outputs,output_file, snap="PART_"):
    """Find the last written snapshot"""
    written = glob.glob(path.join(path.join(outputs, output_file),snap+"[0-9][0-9][0-9]"))
    if not written:
        raise IOError("No snapshots for",outputs)
    matches = [re.search(snap+"([0-9][0-9][0-9])",wr) for wr in written]
    snapnums = [int(mm.group(1)) for mm in matches]
    return sorted(snapnums)[-1]

def _get_regex(odir, output_file):
    """Determine which file type we are parsing: Gadget-3 or MP-Gadget."""
    output_txt = path.join(output_file, "info.txt")
    output = glob.glob(path.join(odir,output_txt))
    regex = r"Redshift: ([0-9]{1,3}\.?[0-9]*)"
    #If no info.txt, probably we are MP-Gadget and need cpu.txt instead
    if not output:
        output_txt = path.join(output_file, "cpu.tx*")
        output = glob.glob(path.join(odir,output_txt))
        if not output:
            return "", regex
        return output_txt, r"Step [0-9]*, Time: ([0-9]{1,3}\.?[0-9]*)"
    return output_txt, regex

def check_status(rundir, output_file="output", endz=2, use_file=True, snap="PART_"):
    """Get completeness status for every directory in the suite.
    Ultimately this should work out whether there
    was an error or just a timeout."""
    rundir = path.expanduser(rundir)
    odirs = glob.glob(path.join(rundir, "*"+os.path.sep))
    if not odirs:
        raise IOError(rundir +" is empty.")
    if use_file:
        redshifts = [_check_single_status_snap(cc,output_file=output_file,snap=snap) for cc in odirs]
    else:
        #Check for info.txt or cpu.txt:
        output_txt, regex = _get_regex(odirs[0], output_file)
        #If we are handed a single directory rather than a set.
        if not output_txt:
            odirs = glob.glob(path.join(rundir, "out*"))
            output_txt, regex = _get_regex(odirs[0], output_file="")
        #If the simulation didn't start yet
        if not output_txt:
            return odirs, [False for _ in odirs], [1100. for _ in odirs]
        redshifts = [_check_single_status(path.join(cc,output_txt), regex) for cc in odirs]
    return odirs, [zz <= endz for zz in redshifts], redshifts

def print_status(rundir, output_file="output", endz=2.01):
    """Get completeness status for every directory in the suite.
    Ultimately this should work out whether there
    was an error or just a timeout."""
    outputs, completes, redshifts = check_status(rundir, output_file, endz)
    for oo, cc,zz in zip(outputs, completes, redshifts):
        print(oo," : ",end="")
        if not cc:
            print("NOT COMPLETE: z=",zz)
        else:
            print("COMPLETE")

def resub_not_complete(rundir, output_file="output", endz=2.01, script_file="mpi_submit", resub_command=None, paramfile="mpgadget.param", restart=1, snap="PART_"):
    """Resubmit incomplete simulations to the queue.
    We also edit the script file to add a RestartFlag"""
    if resub_command is None:
        resub_command = detect_submit()
    outputs, completes, _ = check_status(rundir, output_file, endz)
    #Pathnames for incomplete simulations
    for odir,cc in zip(outputs,completes):
        if cc:
            continue
        rest = " "+str(restart)
        if restart == 2:
            snapnum = _find_snap(odir, output_file,snap=snap)
            rest += " "+str(snapnum)
        script_file_resub = script_file+"_resub"
        found = False
        with open(path.join(odir, script_file),'r') as ifile:
            with open(path.join(odir, script_file_resub),'w') as ofile:
                line = ifile.readline()
                while line != '':
                    #Find the actual submission line and add a '1' after the paramfile.
                    if re.search("mpirun|mpiexec|ibrun", line):
                        nline = re.sub(paramfile, paramfile+rest,line)
                        assert nline != line
                        line = nline
                        found = True
                    #Write each line straight through to the output by default.
                    ofile.write(line)
                    line = ifile.readline()
        if found:
            print("Re-submitting: ",path.join(odir, script_file_resub))
            subprocess.call([resub_command, script_file_resub], cwd=odir)
        else:
            print("ERROR: no change, not re-submitting: ",path.join(odir, script_file_resub))

def check_status_ics(rundir, icdir="ICS"):
    """Get IC generation status for every directory in the suite."""
    rundir = path.expanduser(rundir)
    odirs = glob.glob(path.join(rundir, "*"+os.path.sep))
    if not odirs:
        raise IOError(rundir +" is empty.")
    icex = lambda odir: bool(glob.glob(path.join(path.join(odir, icdir),"*/Header/attr-v2")))
    exists = [icex(cc) for cc in odirs]
    return odirs, exists

def resub_not_complete_genic(rundir, icdir="ICS", script_file="mpi_submit_genic", resub_command=None):
    """Resubmit failed IC generations to the queue."""
    if resub_command is None:
        resub_command = detect_submit()
    outputs, completes = check_status_ics(rundir, icdir)
    #Pathnames for incomplete simulations
    for odir,cc in zip(outputs,completes):
        if cc:
            continue
        print("Re-submitting: ",path.join(odir, script_file))
        subprocess.call([resub_command, script_file], cwd=odir)

def _check_spectra_single(odir, output="output", specdir="SPECTRA_", partdir="PART_"):
    """Check that a single simulation has all its spectra"""
    parts = glob.glob(path.join(odir, path.join(output, partdir+"*")))
    specs = [re.sub(partdir, specdir, part) for part in parts]
    for part,spec in zip(parts,specs):
        #Check whether the spectra exist
        exists = bool(glob.glob(os.path.join(spec, "lya_forest_spectra.hdf5")))
        #If they don't, check whether this is because we have a strange redshift
        if not exists:
            red = _get_redshift_snapshot(part)
            #Is it a forest redshift range?
            if red <= 5.5 and abs(red * 5 - int(red * 5)) < 0.002:
                return False
    return True

def check_status_spectra(rundir, output="output", specdir="SPECTRA_", partdir="PART_"):
    """Get spectral generation status for every directory in the suite."""
    rundir = path.expanduser(rundir)
    odirs = glob.glob(path.join(rundir, "*"+os.path.sep))
    if not odirs:
        raise IOError(rundir +" is empty.")
    exists = [_check_spectra_single(odir, output=output, specdir=specdir, partdir=partdir) for odir in odirs]
    return odirs, exists

def resub_not_complete_spectra(rundir, output="output", specdir="SPECTRA_", partdir="PART_", script_file="spectra_submit", resub_command=None):
    """Resubmit failed IC generations to the queue."""
    if resub_command is None:
        resub_command = detect_submit()
    outputs, completes = check_status_spectra(rundir, output=output, specdir=specdir, partdir=partdir)
    #Pathnames for incomplete simulations
    for odir,cc in zip(outputs,completes):
        if cc:
            continue
        print("Re-submitting: ",path.join(odir, script_file))
        subprocess.call([resub_command, script_file], cwd=odir)
