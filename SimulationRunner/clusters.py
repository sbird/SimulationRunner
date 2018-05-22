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
        name = os.path.basename(os.path.normpath(outdir))
        with open(os.path.join(outdir, "mpi_submit"),'w') as mpis:
            mpis.write("#!/bin/bash\n")
            mpis.write(self._queue_directive(name))
            mpis.write(self._mpi_program())

    def _mpi_program(self):
        """String for MPI program to execute"""
        qstring = "mpirun -np "+str(self.nproc)+" "+self.gadgetexe+" "+self.gadgetparam+"\n"
        return qstring

    def _queue_directive(self, name, prefix="#PBS"):
        """Write the part of the mpi_submit file that directs the queueing system.
        This is usually specific to a given cluster.
        The prefix argument is a string at the start of each line.
        It separates queueing system directives from normal comments"""
        _ = name
        qstring = prefix+" -j eo\n"
        qstring += prefix+" -m bae\n"
        qstring += prefix+" -M "+self.email+"\n"
        qstring += prefix+" -l walltime="+str(self.timelimit)+":00:00\n"
        return qstring

    def cluster_runtime(self):
        """Runtime options for cluster. Here memory."""
        return {}

    def cluster_config_options(self,config, prefix=""):
        """Config options that might be specific to a particular cluster"""
        _ = (config, prefix)
        #isend/irecv is quite slow on some clusters because of the extra memory allocations.
        #Maybe test this on your specific system and see if it helps.
        #config.write(prefix+"NO_ISEND_IRECV_IN_DOMAIN\n")
        #config.write(prefix+"NO_ISEND_IRECV_IN_PM\n")
        #config.write(prefix+"NOTYPEPREFIX_FFTW\n")
        return

    def cluster_optimize(self):
        """Compiler optimisation options for a specific cluster.
        Only MP-Gadget pays attention to this."""
        return "-fopenmp -O3 -g -Wall -ffast-math -march=native"

class ComaClass(ClusterClass):
    """Subclassed for specific properties of the Coma cluster at CMU.
    __init__ and _queue_directive are changed."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = 1200

    def _queue_directive(self, name, prefix="#PBS"):
        """Generate mpi_submit with coma specific parts"""
        qstring = super()._queue_directive(name=name, prefix=prefix)
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

    def _queue_directive(self, name, prefix="#PBS"):
        """Generate mpi_submit with coma specific parts"""
        qstring = super()._queue_directive(name=name, prefix=prefix)
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
        super().__init__(*args, nproc=nproc,timelimit=timelimit, **kwargs)
        self.memory = 3000

    def _queue_directive(self, name, prefix="#PBS"):
        """Generate mpi_submit with coma specific parts"""
        qstring = super()._queue_directive(name=name, prefix=prefix)
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

class MARCCClass(ClusterClass):
    """Subclassed for the MARCC cluster at JHU.
    This has 24 cores per node, shared memory of 128GB pr node.
    Ask for complete nodes.
    Uses SLURM."""
    def __init__(self, *args, nproc=48,timelimit=8,**kwargs):
        #Complete nodes!
        assert nproc % 24 == 0
        super().__init__(*args, nproc=nproc,timelimit=timelimit, **kwargs)
        self.memory = 5000

    def _queue_directive(self, name, prefix="#SBATCH"):
        """Generate mpi_submit with coma specific parts"""
        qstring = prefix+" --partition=parallel\n"
        qstring += prefix+" --job-name="+name+"\n"
        qstring += prefix+" --time="+str(int(self.timelimit))+":00:0\n"
        qstring += prefix+" --nodes="+str(int(self.nproc/24))+"\n"
        #Number of tasks (processes) per node
        qstring += prefix+" --ntasks-per-node=24\n"
        #Number of cpus (threads) per task (process)
        qstring += prefix+" --cpus-per-task=1\n"
        #Max 128 GB per node (24 cores)
        qstring += prefix+" --mem-per-cpu="+str(self.memory)+"\n"
        qstring += prefix+" --mail-type=end\n"
        qstring += prefix+" --mail-user="+self.email+"\n"
        return qstring

    def _mpi_program(self):
        """String for MPI program to execute.
        Note that this assumes you aren't using threads!"""
        #Change to current directory
        qstring = "export OMP_NUM_THREADS=1\n"
        #This is for threads
        #qstring += "export OMP_NUM_THREADS = $SLURM_CPUS_PER_TASK\n"
        #Adjust for thread/proc balance per socket.
        #qstring += "mpirun --map-by ppr:3:socket:PE=4 "+self.gadgetexe+" "+self.gadgetparam+"\n"
        #So we pick up fftw2
        qstring += "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.locallibs/lib\n"
        qstring += "mpirun --map-by core "+self.gadgetexe+" "+self.gadgetparam+"\n"
        return qstring

    def cluster_optimize(self):
        """Compiler optimisation options for a specific cluster.
        Only MP-Gadget pays attention to this."""
        return "-fopenmp -O3 -g -Wall -march=native"

class BIOClass(ClusterClass):
    """Subclassed for the biocluster at UCR.
    This has 32 cores per node, shared memory of 128GB per node.
    Ask for complete nodes.
    Uses SLURM."""
    def __init__(self, *args, nproc=128,timelimit=2,**kwargs):
        #Complete nodes!
        assert nproc % 32 == 0
        super().__init__(*args, nproc=nproc,timelimit=timelimit, **kwargs)
        self.memory = 4

    def _queue_directive(self, name, prefix="#SBATCH"):
        """Generate mpi_submit with coma specific parts"""
        qstring = prefix+" --partition=short\n"
        qstring += prefix+" --job-name="+name+"\n"
        qstring += prefix+" --time="+str(int(self.timelimit))+":00:0\n"
        qstring += prefix+" --nodes="+str(int(self.nproc/32))+"\n"
        #Number of tasks (processes) per node
        qstring += prefix+" --ntasks-per-node=32\n"
        #Number of cpus (threads) per task (process)
        qstring += prefix+" --cpus-per-task=1\n"
        #Max 128 GB per node (24 cores)
        qstring += prefix+" --mem-per-cpu=4G\n"
        qstring += prefix+" --mail-type=end\n"
        qstring += prefix+" --mail-user="+self.email+"\n"
        return qstring

    def _mpi_program(self):
        """String for MPI program to execute.
        Note that this assumes you aren't using threads!"""
        #Change to current directory
        qstring = "export OMP_NUM_THREADS=1\n"
        #This is for threads
        #qstring += "export OMP_NUM_THREADS = $SLURM_CPUS_PER_TASK\n"
        #Adjust for thread/proc balance per socket.
        #qstring += "mpirun --map-by ppr:3:socket:PE=4 "+self.gadgetexe+" "+self.gadgetparam+"\n"
        #So we pick up fftw2
        qstring += "mpirun --map-by core "+self.gadgetexe+" "+self.gadgetparam+"\n"
        return qstring

    def cluster_runtime(self):
        """Runtime options for cluster. Here memory."""
        return {'MaxMemSizePerNode': 4 * 32 * 950}

    def cluster_optimize(self):
        """Compiler optimisation options for a specific cluster.
        Only MP-Gadget pays attention to this."""
        return "-fopenmp -O3 -g -Wall -ffast-math -march=corei7"

class HypatiaClass(ClusterClass):
    """Subclass for Hypatia cluster in UCL"""
    def generate_mpi_submit(self, outdir):
        """Generate a sample mpi_submit file.
        The prefix argument is a string at the start of each line.
        It separates queueing system directives from normal comments"""
        with open(os.path.join(outdir, "mpi_submit"),'w') as mpis:
            mpis.write("#!/bin/csh -f\n")
            mpis.write(self._queue_directive())
            mpis.write(self._mpi_program())

    def _queue_directive(self, name, prefix="#PBS"):
        """Generate Hypatia-specific mpi_submit"""
        qstring = prefix+" -m bae\n"
        qstring += prefix+" -r n\n"
        qstring += prefix+" -q smp\n"
        qstring += prefix+" -N "+name+"\n"
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
