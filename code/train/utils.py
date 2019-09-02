from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os


def parent_directory(path, times=1):
    backup_path = path
    for i in range(times):
        backup_path = os.path.dirname(backup_path)
    return backup_path
