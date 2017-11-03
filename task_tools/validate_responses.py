import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from voxelproto.crowd_task_pb2 import CrowdTask, CrowdTaskResponse

def keep_response(response):
    if response.completed != CrowdTaskResponse.COMPLETED:
        return False

    # Old-style names from the dev portal
    if (len(response.assignment_id) == 11 and
        len(response.worker_id) == 11 and
        response.assignment_id.startswith('A') and
        response.worker_id.startswith('W')):
        return False

    # New-style names from the dev portal
    # Note that real IDs are always uppercase, so this will not result in
    # random conflicts
    if 'test' in response.assignment_id or 'test' in response.worker_id:
        return False

    try:
        response.data_struct['annotation']
    except ValueError:
        return True

    if len(response.data_struct['annotation']) < 5:
        return False

    # I annotate some of my responses this way, though this should be redundant
    # with the dev portal names
    if 'Nikita' in response.data_struct['annotation']:
        return False

    return True
