"""Specialised module to contain functions to specialise the simulation run to different clusters"""
import os.path

class ClusterClass(object):
    """Generic class implementing some general defaults for cluster submissions."""
    def __init__(self, gadget="P-Gadget3", param="gadget3.param", nproc=256, timelimit=24):
        """CPU parameters (walltime, number of cpus, etc):
        these are specified to a default here, but should be over-ridden in a machine-specific decorator."""
        self.nproc = nproc
        self.email = "sbird4@jhu.edu"
        self.timelimit = timelimit
        #Maximum memory available for an MPI task
        self.memory = 1800
        self.gadgetexe = gadget
        self.gadgetparam = param

    def generate_mpi_submit(self, outdir):
        """Generate a sample mpi_submit file.
        The prefix argument is a string at the start of each line.
        It separates queueing system directives from normal comments"""
        with open(os.path.join(outdir, "mpi_submit"),'w') as mpis:
            mpis.write("#!/bin/bash\n")
            mpis.write(self._queue_directive())
            mpis.write(self._mpi_program())

    def _mpi_program(self):
        """String for MPI program to execute"""
        qstring = "mpirun -np "+str(self.nproc)+" "+self.gadgetexe+" "+self.gadgetparam+"\n"
        return qstring

    def _queue_directive(self, prefix="#PBS"):
        """Write the part of the mpi_submit file that directs the queueing system.
        This is usually specific to a given cluster.
        The prefix argument is a string at the start of each line.
        It separates queueing system directives from normal comments"""
        qstring = prefix+" -j eo\n"
        qstring += prefix+" -m bae\n"
        qstring += prefix+" -M "+self.email+"\n"
        qstring += prefix+" -l walltime="+str(self.timelimit)+":00:00\n"
        return qstring

    def cluster_config_options(self,config, prefix=""):
        """Config options that might be specific to a particular cluster"""
        _ = (config, prefix)
        #isend/irecv is quite slow on some clusters because of the extra memory allocations.
        #Maybe test this on your specific system and see if it helps.
        #config.write(prefix+"NO_ISEND_IRECV_IN_DOMAIN\n")
        #config.write(prefix+"NO_ISEND_IRECV_IN_PM\n")
        #config.write(prefix+"NOTYPEPREFIX_FFTW\n")
        return

class ComaClass(ClusterClass):
    """Subclassed for specific properties of the Coma cluster at CMU.
    __init__ and _queue_directive are changed."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = 1200

    def _queue_directive(self, prefix="#PBS"):
        """Generate mpi_submit with coma specific parts"""
        qstring = super()._queue_directive(prefix)
        qstring += prefix+" -q amd\n"
        qstring += prefix+" -l nodes="+str(int(self.nproc/16))+":ppn=16\n"
        return qstring

    def cluster_config_options(self,config, prefix=""):
        """Config options that might be specific to a particular cluster"""
        _ = prefix
        #isend/irecv is quite slow on some clusters because of the extra memory allocations.
        #Maybe test this on your specific system and see if it helps.
        config.write("NO_ISEND_IRECV_IN_DOMAIN\n")
        config.write("NO_ISEND_IRECV_IN_PM\n")
        #config.write("NOTYPEPREFIX_FFTW\n")
        return

class HipatiaClass(ClusterClass):
    """Subclassed for specific properties of the Hipatia cluster in Barcelona.
    __init__ and _queue_directive are changed."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = 2500

    def _queue_directive(self, prefix="#PBS"):
        """Generate mpi_submit with coma specific parts"""
        qstring = super()._queue_directive(prefix)
        qstring += prefix+" -l nodes="+str(int(self.nproc/16))+":ppn=16\n"
        qstring += prefix+" -l mem="+str(int(self.memory*self.nproc/1000))+"g\n"
        #Pass environment to child processes
        qstring += prefix+" -V\n"
        return qstring

    def _mpi_program(self):
        """String for MPI program to execute. Hipatia is weird because PBS_JOBID needs to be unset for the job to launch."""
        #Change to current directory
        qstring = "cd $PBS_O_WORKDIR\n"
        #Don't ask me why this works, but it is necessary.
        qstring += "unset PBS_JOBID\n"
        qstring += "mpirun -np "+str(self.nproc)+" "+self.gadgetexe+" "+self.gadgetparam+"\n"
        return qstring

class HHPCClass(ClusterClass):
    """Subclassed for specific properties of the HHPCv2 cluster at JHU.
    This has 12 cores per node, and 4GB per core.
    You must use full nodes, preferrably more than 1 hour per job,
    and pass PBS_NODEFILE to mpirun.
    __init__ and _queue_directive are changed."""
    def __init__(self, *args, nproc=252,timelimit=8,**kwargs):
        super().__init__(*args, **kwargs,nproc=nproc,timelimit=timelimit)
        self.memory = 3000

    def _queue_directive(self, prefix="#PBS"):
        """Generate mpi_submit with coma specific parts"""
        qstring = super()._queue_directive(prefix)
        qstring += prefix+" -l nodes="+str(int(self.nproc/12))+":ppn=12\n"
        #Pass environment to child processes
        return qstring

    def _mpi_program(self):
        """String for MPI program to execute. PBS_NODEFILE needs to be passed to mpirun for HHPC."""
        #Change to current directory
        qstring = "cd $PBS_O_WORKDIR\n"
        #Required.
        qstring += "export MPI_NPROCS=`wc -l $PBS_NODEFILE`\n"
        qstring += "mpirun -machinefile $PBS_NODEFILE "+self.gadgetexe+" "+self.gadgetparam+"\n"
        return qstring

class HypatiaClass(ClusterClass):
    """Subclass for Hypatia cluster in UCL"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #hypatia has no timelimit

    def generate_mpi_submit(self, outdir):
        """Generate a sample mpi_submit file.
        The prefix argument is a string at the start of each line.
        It separates queueing system directives from normal comments"""
        with open(os.path.join(outdir, "mpi_submit"),'w') as mpis:
            mpis.write("#!/bin/csh -f\n")
            mpis.write(self._queue_directive())
            mpis.write(self._mpi_program())

    def _queue_directive(self, prefix="#PBS"):
        """Generate Hypatia-specific mpi_submit"""
        qstring = prefix+" -m bae\n"
        qstring += prefix+" -r n\n"
        qstring += prefix+" -q smp\n"
        qstring += prefix+" -N "+os.path.basename(self.gadgetexe)+"\n"
        qstring += prefix+" -M "+self.email+"\n"
        qstring += prefix+" -l nodes=1:ppn="+str(self.nproc)+"\n"
        #Pass environment to child processes
        qstring += prefix+" -V\n"
        return qstring

    def _mpi_program(self):
        """String for MPI program to execute. Hipatia is weird because PBS_JOBID needs to be unset for the job to launch."""
        #Change to current directory
        qstring = "cd $PBS_O_WORKDIR\n"
        #Don't ask me why this works, but it is necessary.
        qstring += ". /opt/torque/etc/openmpi-setup.sh\n"
        qstring += "mpirun -v -hostfile $PBS_NODEFILE -npernode "+str(self.nproc)+" "+self.gadgetexe+" "+self.gadgetparam+"\n"
        return qstring
