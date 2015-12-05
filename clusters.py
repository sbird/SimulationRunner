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
            """String for MPI program to execute. Hipatia is weird because the default (torque) program launch module is broken.
            Instead we need to launch programs via ssh (which means we need a passwordless ssh key)"""
            #Change to current directory
            qstring = "cd $PBS_O_WORKDIR\n"
            #mca plm rsh launches programs via ssh
            qstring += "mpirun -mca plm rsh -np "+str(self.nproc)+" "+self.gadgetexe+" "+self.gadgetparam+"\n"
            return qstring

    return HipatiaClass

