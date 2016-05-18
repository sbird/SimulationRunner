"""Specialised module to contain functions to specialise the simulation run to different clusters"""

def coma_mpi_decorate(class_name, nproc=256, timelimit=24):
    """This is a class decorator: it creates a new class which subclasses a given class to contain the information
        specific to using the COMA cluster.
        __init__ and _queue_directive are subclassed"""
    newdoc = class_name.__doc__ + """
    Subclassed for specific properties of the Coma cluster at CMU.
    __init__ and _queue_directive are changed."""
    class ComaClass(class_name):
        """Docstring should be specified in newdoc"""
        __doc__ = newdoc
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.memory = 1200
            self.timelimit = timelimit
            self.nproc = nproc

        def _queue_directive(self, prefix="#PBS"):
            """Generate mpi_submit with coma specific parts"""
            qstring = super()._queue_directive(prefix)
            qstring += prefix+" -q amd\n"
            qstring += prefix+" -l nodes="+str(int(self.nproc/16))+":ppn=16\n"
            return qstring

        def _cluster_config_options(self,config):
            """Config options that might be specific to a particular cluster"""
            #isend/irecv is quite slow on some clusters because of the extra memory allocations.
            #Maybe test this on your specific system and see if it helps.
            config.write("NO_ISEND_IRECV_IN_DOMAIN\n")
            config.write("NO_ISEND_IRECV_IN_PM\n")
            #config.write("NOTYPEPREFIX_FFTW\n")
            return


    return ComaClass

def hipatia_mpi_decorate(class_name, nproc=256, timelimit=24):
    """This is a class decorator: it creates a new class which subclasses a given class to contain the information
        specific to using the Hipatia cluster.
        __init__ and _queue_directive are subclassed"""
    newdoc = class_name.__doc__ + """
    Subclassed for specific properties of the Hipatia cluster in Barcelona.
    __init__ and _queue_directive are changed."""
    class HipatiaClass(class_name):
        """Docstring should be specified in newdoc"""
        __doc__ = newdoc
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.memory = 2500
            self.timelimit = timelimit
            self.nproc = nproc

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

    return HipatiaClass

def hhpc_mpi_decorate(class_name, nproc=252, timelimit=8):
    """This is a class decorator: it creates a new class which subclasses a given class to contain the information
        specific to using the HHPCv2 cluster at JHU.
        __init__ and _queue_directive are subclassed"""
    newdoc = class_name.__doc__ + """
    Subclassed for specific properties of the HHPCv2 cluster at JHU.
    This has 12 cores per node, and 4GB per core.
    You must use full nodes, preferrably more than 1 hour per job,
    and pass PBS_NODEFILE to mpirun.
    __init__ and _queue_directive are changed."""
    class HHPCClass(class_name):
        """Docstring should be specified in newdoc"""
        __doc__ = newdoc
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.memory = 3000
            self.timelimit = timelimit
            self.nproc = nproc

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

    return HHPCClass

def hypatia_mpi_decorate(class_name, nproc=60):
    """This is a class decorator: it creates a new class which subclasses a given class to contain the information
        specific to using the Hypatia cluster at UCL.
        __init__ and _queue_directive are subclassed"""
    newdoc = class_name.__doc__ + """
    Subclassed for specific properties of the Hypatia cluster in UCL.
    __init__ and _queue_directive are changed."""
    class HypatiaClass(class_name):
        """Docstring should be specified in newdoc"""
        __doc__ = newdoc
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            #hypatia has no timelimit
            self.nproc = nproc

        def generate_mpi_submit(self):
            """Generate a sample mpi_submit file.
            The prefix argument is a string at the start of each line.
            It separates queueing system directives from normal comments"""
            with open(os.path.join(self.outdir, "mpi_submit"),'w') as mpis:
                mpis.write("#!/bin/csh -f\n")
                mpis.write(self._queue_directive())
                mpis.write(self._mpi_program())

        def _queue_directive(self, prefix="#PBS"):
            """Generate Hypatia-specific mpi_submit"""
            qstring += prefix+" -m bae\n"
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

    return HypatiaClass
