import socket
import getpass
import subprocess

def get_pc_and_version():
    hostname = socket.gethostname()
    username = getpass.getuser()

    # Store the version of the session
    commit_hash = subprocess.check_output(["git", "describe", '--always']).strip().decode('ascii')
    try:
        subprocess.check_output(["git", "diff", "--quiet"])
    except:
        commit_hash += '-dirty'
    version = commit_hash

    return hostname, username, version
