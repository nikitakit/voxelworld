"""
This file takes responses to annotation task
(see `generate_whereis_tasks_synthetic.py`), generates potential distractor
locations for Misty, and writes out the resulting tasks. These tasks can then
be given to humans to evaluate annotation quality and human performance.
"""

# %cd ~/dev/mctest/task_tools
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from util.monkeypatch_voxelregion import monkeypatch
monkeypatch()
import util.struct_tools
from voxelproto.voxelregion_pb2 import VoxelRegion
import math
import random
import numpy as np

from lrpc.lrpc import get_lrpc
import asyncio
from voxelproto import voxelregion_pb2
from voxelproto import world_service_pb2
from voxelproto import crowd_task_pb2
from voxelproto import task_service_pb2
from voxelproto.crowd_task_pb2 import CrowdTask, CrowdTaskResponse
from voxelproto.task_service_pb2 import TaskServiceRequest, NamedTask
from voxelproto.world_service_pb2 import RegionRequest
from voxelproto.world_snapshot_pb2 import WorldSnapshot

from task_tools.validate_responses import keep_response

ANNOTATE_SEARCH = "/whereis_synth4/"
TASK_NAME_FORMAT = "/guess_whereis_synth34_2/{annotation_basename}-{annotation_num}"
PROMPT_ACCEPT = True


def sample_candidates_two_step(voxels, candidates, misty_location):
    candidates_mask = np.zeros_like(voxels, dtype=bool)
    for cx, cy, cz in candidates:
        candidates_mask[cx, cy, cz] = True

    # pad
    pad_count = 2
    candidates_mask = np.pad(candidates_mask, [[pad_count,pad_count]]*3, mode='constant')
    misty_location = [a+pad_count for a in misty_location]

    def nonzero_adj(coords, grid):
        nz = grid[coords[0]-2:coords[0]+3, coords[1]-2:coords[1]+3,coords[2]-2:coords[2]+3]
        nz[2,2,2] = False

        nz = nz.nonzero()
        for nx, ny, nz in zip(*nz):
            yield nx-2+coords[0], ny-2+coords[1], nz-2+coords[2]

    def shuffled(l):
        l2 = list(l)
        random.shuffle(l2)
        return l2

    choice_candidates = [tuple(misty_location)]

    for x, y, z in shuffled(zip(*candidates_mask.nonzero())):
        if not candidates_mask[x,y,z]:
            continue # just in case, handle mutation while iterating

        too_close = False

        for dxyz in choice_candidates:
            if np.abs(np.array([x,y,z]) - np.array(dxyz)).max() <= 3:
                too_close = True

        if too_close:
            continue

        choice_candidates.append((x,y,z))
        candidates_mask[x,y,z] = False
        if len(choice_candidates) == 3:
            break
    else:
        print("WARNING: not enough initial distractors")
        return None

    for x,y,z in choice_candidates[:3]:
        for dx, dy, dz in shuffled(nonzero_adj((x,y,z), candidates_mask)):
            choice_candidates.append((dx,dy,dz))
            candidates_mask[dx,dy,dz] = False
            break
        else:
            print("WARNING: not enough secondary distractors")
            return None

    random.shuffle(choice_candidates)

    #unpad
    choice_candidates = [(x-pad_count, y-pad_count, z-pad_count) for (x,y,z) in choice_candidates]

    return choice_candidates

def sample_candidates_landmarks(voxels, candidates, misty_location):
    # 0 is air, the others are floor/walls/ceiling
    landmarks = (voxels != 0) & (voxels != 1) & (voxels != 7) & (voxels != 4)

    candidates_mask = np.zeros_like(voxels, dtype=bool)
    for cx, cy, cz in candidates:
        candidates_mask[cx, cy, cz] = True

    # pad
    landmarks = np.pad(landmarks, [[1,1],[1,1],[1,1]], mode='constant')
    candidates_mask = np.pad(candidates_mask, [[1,1],[1,1],[1,1]], mode='constant')
    misty_location = [a+1 for a in misty_location]

    def nonzero_adj(coords, grid, diag=False):
        nz = grid[coords[0]-1:coords[0]+2, coords[1]-1:coords[1]+2,coords[2]-1:coords[2]+2]
        if not diag:
            nz[0,0,0] = nz[0,0,-1] = False
            nz[-1,0,0] = nz[-1,0,-1] = False
            nz[-1,-1,0] = nz[-1,-1,-1] = False
            nz[0,-1,0] = nz[0,-1,-1] = False
            nz[1,1,1] = False
        nz = nz.nonzero()
        for nx, ny, nz in zip(*nz):
            yield nx-1+coords[0], ny-1+coords[1], nz-1+coords[2]

    def clear_adj(coords, grid):
        for nx, ny, nz in nonzero_adj(coords, grid):
            grid[nx,ny,nz] = False


    def shuffled(l):
        l2 = list(l)
        random.shuffle(l2)
        return l2

    for lx, ly, lz in shuffled(nonzero_adj(misty_location, landmarks)):
        candidates_adj = list(nonzero_adj((lx,ly,lz), candidates_mask))
        if len(candidates_adj) >= 2:
            break
    else:
        for lx, ly, lz in shuffled(nonzero_adj(misty_location, landmarks)):
            candidates_adj = list(nonzero_adj((lx,ly,lz), candidates_mask, diag=True))
            if len(candidates_adj) >= 2:
                break
        else:
            print(landmarks[misty_location[0]-1:misty_location[0]+2, misty_location[1]-1:misty_location[1]+2,misty_location[2]-1:misty_location[2]+2])
            print(candidates_mask[misty_location[0]-1:misty_location[0]+2, misty_location[1]-1:misty_location[1]+2,misty_location[2]-1:misty_location[2]+2])
            assert False, "nothing else near misty's landmark"

    distractors = [tuple(pt) for pt in candidates_adj if tuple(pt) != tuple(misty_location)]
    distractors = random.sample(distractors, 1)
    clear_adj(distractors[0], landmarks)
    clear_adj(misty_location, landmarks)
    candidates_mask[tuple(misty_location)] = False
    candidates_mask[tuple(distractors[0])] = False

    for lx, ly, lz in shuffled(zip(*landmarks.nonzero())):
        if not landmarks[lx,ly,lz]:
            continue # handle mutation while iterating

        distractors_adj = list(nonzero_adj((lx,ly,lz), candidates_mask, diag=True))
        if len(distractors_adj) < 2:
            continue

        new_distractors = random.sample(distractors_adj, 2)
        for dx, dy, dz in new_distractors:
            candidates_mask[dx,dy,dz] = False
        landmarks[lx, ly, lz] = False

        distractors.extend(new_distractors)
        if len(distractors) == 5:
            break
    else:
        print("WARNING: not enough distractors")
        for (dx, dy, dz) in shuffled(zip(*candidates_mask.nonzero())):
            if (dx, dy, dz) not in distractors:
                distractors.append((dx,dy,dz))
                if len(distractors) == 5:
                    break
        else:
            assert False, "not enough distractors {}".format(len(distractors))

    choice_candidates = distractors + [misty_location]
    random.shuffle(choice_candidates)

    #unpad
    choice_candidates = [(x-1, y-1, z-1) for (x,y,z) in choice_candidates]

    return choice_candidates

def sample_candidates_simple(voxels, candidates, misty_location):
    distractors = [xyz for xyz in candidates if np.abs(np.array(xyz) - np.array(misty_location)).sum() > 1]
    if len(distractors) < 4:
        print("ERROR: could not find 4 distractors for task: {}".format(task_template.name))
        return None
    distractors = random.sample(distractors, 4)

    # Mix misty_location with distractors
    # This is needed so that the order of presentation to workers can
    # be deterministically recovered from the file

    choice_candidates = distractors + [misty_location]
    random.shuffle(choice_candidates)

    return choice_candidates

async def main():
    print("[main] started")
    names_req = TaskServiceRequest.FindTaskNames()
    names_req.paths.extend([ANNOTATE_SEARCH])
    names_response = await task_rpc.FindNames(names_req)

    tasks_per_chunk = 20
    for i in range(0, len(names_response.names), tasks_per_chunk):
        req = TaskServiceRequest.FindTasks()
        req.names.extend(names_response.names[i:i+tasks_per_chunk])
        req.return_responses = True
        response = await task_rpc.Find(req)
        for task in response.tasks:
            print("Got task", task.name)

            annotation_name = task.name
            while annotation_name.endswith('/'):
                annotation_name = annotation_name[:-1]
            annotation_basename = annotation_name.split('/')[-1]

            req = TaskServiceRequest.SubmitTasks()
            task_template = NamedTask()
            task_template.base_task.MergeFrom(task.base_task)
            task_template.base_task.game = "whereis_guess_v2"

            data = task_template.base_task.data_static_world.data
            data['task_name'] = task.name

            for annotation_num, response in enumerate(task.responses):
                if not keep_response(response):
                    continue

                task_template.name = TASK_NAME_FORMAT.format(annotation_basename=annotation_basename, annotation_num=annotation_num)

                data['annotation'] = response.data_struct['annotation']

                candidates = data['candidates']
                candidates = iter([int(x) for x in candidates])
                candidates = list(zip(candidates, candidates, candidates))
                misty_location = (int(data['x']), int(data['y']), int(data['z']))

                dx, dy, dz = task_template.base_task.data_static_world.snapshot.regions[0].dimensions
                voxels = np.reshape(task_template.base_task.data_static_world.snapshot.regions[0].voxels_u32, [dx, dy, dz])

                choice_candidates = sample_candidates_two_step(voxels, candidates, misty_location)
                if choice_candidates is None:
                    continue

                data.get_or_create_list('choice_candidates').extend(np.array(choice_candidates).flatten().tolist())

                print()
                print("Task: {}".format(task_template.name))
                print("Annotation")
                print()
                print(data['annotation'])
                print()
                if PROMPT_ACCEPT:
                    do_accept = None
                    while do_accept not in ["", "y", "n", "q"]:
                        do_accept = input("Accept? (Y/n/i/q)")
                        if do_accept == "i":
                            tracker_rpc.Activate(task_template.base_task)

                    if do_accept == "q":
                        print("[main] quitting")
                        return
                    elif do_accept == "n":
                        continue

                req.tasks.extend([task_template])
                req.save = True
                req.return_processed = True
                response = await task_rpc.Submit(req)

                print('Saved the task...')
                await asyncio.sleep(0.2)
                break # Only allow one annotation per task

    print("[main] done")

lrpc = get_lrpc()
tracker_rpc = crowd_task_pb2.TrackerService(lrpc)
task_rpc = task_service_pb2.TaskService(lrpc)

lrpc.add_to_event_loop()
# from util import ipyloop
asyncio.get_event_loop().run_until_complete(main())
