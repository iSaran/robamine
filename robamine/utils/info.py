import socket
import getpass
import subprocess
import logging
from tensorboard import default, program
import os
from datetime import datetime

logger = logging.getLogger('robamine.utils.misc')

def get_pc_and_version():
    '''Returns current PC's hostname, username and code's current version.'''
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

def port_is_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_tensorboard_server(logdir):
    port = 6006
    while port_is_in_use(port):
        port += 1
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    tb = program.TensorBoard(default.get_plugins(), default.get_assets_zip_provider())
    tb.configure(argv=[None, '--logdir', logdir, '--port', str(port)])
    url = tb.launch()
    logger.info('TensorBoard plots at %s' % url)
    return url

def get_dir_size(dir = '.'):
    '''Returns the size of the given directory in bytes.'''
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def bytes2human(bytes):
    '''Returns a string with human readable form of the given bytes.'''
    if bytes > 1024 * 1024 * 1024 * 1024:
        return str("{:.2f} TB".format(bytes / (1024 * 1024 * 1024 * 1024)))
    elif bytes > 1024 * 1024 * 1024:
        return str("{:.2f} GB".format(bytes / (1024 * 1024 * 1024)))
    elif bytes > 1024 * 1024:
        return str("{:.2f} MB".format(bytes / (1024 * 1024)))
    elif bytes > 1024:
        return str("{:.2f} KB".format(bytes / 1024))
    else:
        return str(bytes) + ' Bytes'

def get_now_timestamp():
    """
    Returns a timestamp for the current datetime as a string for using it in
    log file naming.
    """
    now_raw = datetime.now()
    return str(now_raw.year) + '.' + \
           '{:02d}'.format(now_raw.month) + '.' + \
           '{:02d}'.format(now_raw.day) + '.' + \
           '{:02d}'.format(now_raw.hour) + '.' \
           '{:02d}'.format(now_raw.minute) + '.' \
           '{:02d}'.format(now_raw.second) + '.' \
           '{:02d}'.format(now_raw.microsecond)
