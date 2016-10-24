"""Module to store some utility functions."""
import glob
import os.path
import subprocess

def find_exec(executable):
    """Simple function to locate a binary in a nearby directory"""
    possible = [executable,]+glob.glob("depends/*/"+executable)+ glob.glob(os.path.join(os.path.join(os.path.dirname(__file__),"../*/"), executable))
    exists = [ex for ex in possible if os.path.exists(ex) and os.path.isfile(ex) ]
    if len(exists) > 1:
        print("Warning: found multiple possibilities: ",exists)
    if len(exists) > 0:
        return exists[0]
    raise ValueError(executable+" not found")

def get_git_hash(path):
    """Get the git hash of a file."""
    rpath = os.path.realpath(path)
    if not os.path.isdir(rpath):
        rpath = os.path.dirname(rpath)
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd = rpath, universal_newlines=True)
    return commit_hash
