if __name__ != "__main__":
    raise ImportError("This file is designed to run as a script")

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from voxelproto.task_service_pb2 import NamedTask
from voxelproto.crowd_task_pb2 import CrowdTaskResponse
from google.protobuf import json_format


import sqlite3
import tornado
from tornado.options import define, options
import os
import json
import sys

define('db', default=os.path.join(os.path.dirname(__file__), "mturk_server.db"),
    help='path to the sqlite database', type=str)

define('create', default=None,
    help='path to json file describing a pool', type=str)

define('add', default=None,
    help='path to json file subtasks to add to a pool', type=str)

define('extract', default=None, multiple=True,
    help='name of pool to extract from', type=str)

define('taskroot', default=os.path.abspath('../task_tools/tasks'),
    help='root folder for task definitions', type=str)

define('responseroot', default=os.path.abspath('../task_tools/task_responses'),
    help='root folder for storing task responses', type=str)

tornado.options.parse_command_line()

## Make sure options are valid
CREATE_JSON = None
ADD_JSON = None

if options.create is None and options.add is None and options.extract is None:
    print('No options given (see --help)')
    sys.exit(0)

if options.create is not None and options.add is not None:
    raise Exception("Can specify --create or --add, but not both")

if options.create is not None:
    with open(options.create) as f:
        CREATE_JSON = json.load(f)

if options.add is not None:
    with open(options.add) as f:
        ADD_JSON = json.load(f)

## Connect to the database
conn = sqlite3.connect(options.db)
c = conn.cursor()

## Ensure all tables exist
c.execute("""CREATE TABLE IF NOT EXISTS assignments
             (assignment_id TEXT PRIMARY KEY, worker_id TEXT, pool TEXT, token TEXT, completed int)
""")
c.execute("""CREATE TABLE IF NOT EXISTS tasks
             (assignment_id TEXT, worker_id TEXT, task_template_row int, completed int, data TEXT)
""")
c.execute("""CREATE TABLE IF NOT EXISTS task_templates
             (pool TEXT, name TEXT, data TEXT)
""")
c.execute("""CREATE TABLE IF NOT EXISTS pools
             (pool TEXT UNIQUE,
             description_title TEXT, description TEXT,
             portal_header_html TEXT, portal_html TEXT,
             count INT)
""")

## Functions for creating pools and adding task templates

def create_pool(name, description_title, description, portal_header_html, portal_html, count):
    c.execute("""INSERT INTO pools (pool, description_title, description, portal_header_html, portal_html, count)
                 VALUES (?, ?, ?, ?, ?, ?)""",
        (name, description_title, description, portal_header_html, portal_html, count))

def add_task_template(pool_name, template):
    if isinstance(template, str):
        if not template.endswith('.pb'):
            template = template + '.pb'

        with open(os.path.join(options.taskroot, template), 'rb') as f:
            raw_task = NamedTask.FromString(f.read())
        task_name = raw_task.name
        crowd_task = raw_task.processed_task
        template = json_format.MessageToJson(crowd_task)
    else:
        template = json.dumps(template)
        task_name = ""
    c.execute("INSERT INTO task_templates (pool, name, data) VALUES (?, ?, ?)",
        (pool_name, task_name, template))

def extract_task_responses(pools):
    name_to_responses = {}

    for pool in pools:
        c.execute("""SELECT task_templates.name, tasks.assignment_id, tasks.worker_id, tasks.completed, tasks.data
                     FROM tasks INNER JOIN task_templates
                     ON task_templates.rowid == tasks.task_template_row
                     WHERE pool = ? AND completed != 0
                    """, (pool,))

        for name, assignment_id, worker_id, completed, data in c.fetchall():
            if name not in name_to_responses:
                name_to_responses[name] = NamedTask()
                name_to_responses[name].name = name

            response = name_to_responses[name].responses.add()
            response.assignment_id = assignment_id
            response.worker_id = worker_id
            response.completed = completed
            try:
                json_format.Parse(data, response)
            except:
                pass

    for task_name, responses in name_to_responses.items():
        folder = os.path.join(options.responseroot, os.path.dirname(task_name))
        filename = os.path.join(folder, os.path.basename(task_name) + '.pb')
        print('Saving task responses to:', filename)

        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(filename, 'wb') as f:
            f.write(responses.SerializeToString())

conn.commit()
try:
    if CREATE_JSON is not None:
        for k, v in CREATE_JSON.items():
            create_pool(k,
                v['description_title'], v['description'],
                v['portal_header_html'], v['portal_html'],
                v['count'])
            for task_template in v['task_templates']:
                add_task_template(k, task_template)
    elif ADD_JSON is not None:
        for k, v in ADD_JSON.items():
            for task_template in v['task_templates']:
                add_task_template(k, task_template)
    elif options.extract is not None:
        extract_task_responses(options.extract)
    conn.commit()
except:
    conn.rollback()
    raise
finally:
    conn.close()
