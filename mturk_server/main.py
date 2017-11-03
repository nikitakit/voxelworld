import tornado.ioloop
import tornado.web
import tornado.httpserver
import tornado.websocket
import logging
import sqlite3
from tornado.options import define, options
import ssl
import os
import json
import hmac, binascii
from enum import Enum
import boto3
import os
import sys
from threading import Thread
import random
from datetime import datetime, timezone

define('port', default=8081, help='run on the given port', type=int)
define('db', default=os.path.join(os.path.dirname(__file__), "mturk_server.db"),
    help='path to the sqlite database', type=str)
define('env', default=os.path.join(os.path.dirname(__file__), "aws_env.json"),
    help='path to json file with AWS environment vars', type=str)
define('key', default=os.path.join(os.path.dirname(__file__), "mykey.key"),
    help='path to SSL key file', type=str)
define('cert', default=os.path.join(os.path.dirname(__file__), "mycert.pem"),
    help='path to SSL cert file', type=str)


serverlog = logging.getLogger('mturk_server')
serverlog.setLevel(logging.DEBUG)

def load_sqs_credentials():
    global options
    if not options.env:
        return
    with open(options.env) as f:
        env_dict = json.load(f)

    for k, v in env_dict.items():
        os.environ[k] = v

sqs_running = True
def receive_messages():
    """
    Runs in a thread
    """
    global sqs_running
    if 'MTURK_SQS_QUEUE_NAME' not in os.environ:
        # We don't have a queue, so stop
        serverlog.warning("No SQS queue configured: not using SQS")
        sqs_running = False
        return

    queue_name = os.environ['MTURK_SQS_QUEUE_NAME']
    if not queue_name:
        serverlog.warning("No SQS queue configured: not using SQS")
        sqs_running = False
        return

    sqs = boto3.resource('sqs')
    mturk_actions_queue = sqs.get_queue_by_name(QueueName=queue_name)

    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('nose').setLevel(logging.WARNING)

    count = 0
    while sqs_running:
        count += 1
        if count % 10 == 0:
            serverlog.debug('Polling for SQS messages is alive')
            count = 0
        for msg in mturk_actions_queue.receive_messages():
            msg_dict = json.loads(msg.body)
            for event in msg_dict.get('Events', []):
                if event['EventType'] =='AssignmentReturned':
                    code = CompletionCode.returned
                elif event['EventType'] =='AssignmentAbandoned':
                    code = CompletionCode.abandoned
                else:
                    continue

                assignment_id = event['AssignmentId']
                serverlog.debug("Got SQS notification re: assignment {}".format(assignment_id))

                # Handle the event inside the main event loop, not in this thread
                tornado.ioloop.IOLoop.current().add_callback(on_assignment_changed, assignment_id, code)
            msg.delete()

def on_assignment_changed(assignment_id, code):
    worker_id = DB.get_worker_for_assignment(assignment_id)
    serverlog.info("[wId: %s; aId: %s] Changing assignment to status %s", worker_id, assignment_id, code)
    DB.complete_assignment(assignment_id, code)

    worker_id = DB.get_worker_for_assignment(assignment_id)
    if worker_id in WORKER_TO_DASHBOARD_WS:
        handler = WORKER_TO_DASHBOARD_WS[worker_id]
        if handler is not None:
            handler.prompt_returned()

# This should match the values in crowd_task.proto
class CompletionCode(Enum):
    active = 0
    completed = 1
    no_work = 2
    generic_error = 3
    work_rejected = 4
    returned = 5
    abandoned = 6

class AssignmentDatabase():
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        self.c = self.conn.cursor()
        self.SECRET = b"voxelworld1;p,.r" # Used to approve HITs

    def assign_worker(self, assignment_id, worker_id, pool, token):
        self.c.execute("INSERT OR IGNORE INTO assignments (assignment_id, worker_id, pool, token, completed) VALUES (?, ?, ?, ?, 0)",
            (assignment_id, worker_id, pool, token))

        self.c.execute("SELECT task_template_row FROM tasks WHERE assignment_id = ? and worker_id = ?", (assignment_id, worker_id))
        if self.c.fetchone() is None:
            have_work = self.assign_from_pool(assignment_id, worker_id, pool)
            if not have_work:
                self.c.execute("UPDATE assignments SET completed = ? WHERE assignment_id = ?", (CompletionCode.no_work.value, assignment_id,))

        self.conn.commit()

    def get_worker_for_assignment(self, assignment_id):
        self.c.execute("SELECT worker_id FROM assignments WHERE assignment_id = ?", (assignment_id,))
        res = self.c.fetchone()
        if res is None:
            return None
        return res[0]

    def assign_from_pool(self, assignment_id, worker_id, pool):
        self.c.execute("SELECT count FROM pools WHERE pool = ?", (pool,))
        (count,) = self.c.fetchone()
        self.c.execute("""SELECT task_templates.rowid
                          FROM task_templates
                          LEFT OUTER JOIN tasks ON tasks.task_template_row = task_templates.rowid
                          WHERE pool = ?
                            AND (tasks.completed = 0 OR tasks.completed = 1 OR tasks.completed IS NULL)
                            AND task_templates.rowid NOT IN (SELECT task_template_row FROM tasks WHERE worker_id = ?)
                          GROUP BY task_templates.rowid
                          ORDER BY COUNT(tasks.completed), RANDOM()
                          LIMIT ?""",
                       (pool, worker_id, count))
        task_template_rows = self.c.fetchall()
        serverlog.info('[wId: {}; aId: {}] Assigning {} tasks: {}'.format(worker_id, assignment_id, count, task_template_rows))
        if len(task_template_rows) < count:
            return False
        for (task_template_row,) in task_template_rows:
            self.c.execute("INSERT INTO tasks (assignment_id, worker_id, task_template_row, completed, data) VALUES (?, ?, ?, 0, '')", (assignment_id, worker_id, task_template_row))
        return True

    def get_assignment(self, worker_id):
        self.c.execute("SELECT assignment_id FROM assignments WHERE worker_id = ? AND completed = 0", (worker_id,))
        res = self.c.fetchone()
        if res is not None:
            res = res[0]
        return res

    def get_assignment_pool_data(self, assignment_id):
        self.c.execute("SELECT pool FROM assignments WHERE assignment_id = ?", (assignment_id,))
        pool = self.c.fetchone()
        if pool is None:
            return None
        pool = pool[0]
        return self.get_pool_data(pool)

    def get_pool_data(self, pool):
        self.c.execute("SELECT description_title, description, portal_header_html, portal_html FROM pools WHERE pool = ?", (pool,))
        res = self.c.fetchone()
        if res is None:
            return None
        return {
            "name": res[0],
            "description": res[1],
            "portal_header_html": res[2],
            "portal_html": res[3]
        }

    def get_task(self, worker_id):
        assignment_id = self.get_assignment(worker_id)
        if assignment_id is None:
            return None
        # TODO: use a join
        self.c.execute("SELECT rowid, task_template_row FROM tasks WHERE assignment_id = ? AND completed = 0", (assignment_id,))
        res = self.c.fetchone()
        if res is None:
            serverlog.warning("No task active despite assignment not being marked as completed")
            return None
        task_id, task_template_row = res[0], res[1]
        self.c.execute("SELECT data FROM task_templates WHERE rowid = ?", (task_template_row,))
        res = self.c.fetchone()
        if res is None:
            serverlog.error("invalid foreign key")
            return None
        res = res[0]
        res = json.loads(res)
        res['task_id'] = task_id
        return res

    def get_all_tasks(self, worker_id):
        # TODO: use a join
        self.c.execute("SELECT assignment_id, completed FROM tasks WHERE worker_id = ?", (worker_id,))
        res = []
        for item in self.c.fetchall():
            res.append({'assignment_id': item[0], 'completed': bool(item[1])})
        return res

    def complete_task(self, task_id, data, code=CompletionCode.completed):
        if isinstance(code, int):
            code = CompletionCode(code)

        self.c.execute("SELECT completed FROM tasks WHERE rowid = ?", (task_id,))
        current_code = self.c.fetchone()
        if current_code is None:
            serverlog.error("Attempt to complete invalid task id")
            return
        current_code = current_code[0]
        if current_code != CompletionCode.active.value:
            # This can happen when an assignment is returned, but the dash is not
            # updated
            serverlog.warning("Submitting task a second time")
            return

        serverlog.debug('completing task {} {} {}'.format(task_id, data, code))
        self.c.execute("UPDATE tasks SET completed = ?, data = ? WHERE rowid = ?", (code.value, json.dumps(data), task_id))

        # TODO: use a join
        self.c.execute("SELECT assignment_id FROM tasks WHERE rowid = ?", (task_id,))
        assignment_id = self.c.fetchone()[0]

        self.c.execute("SELECT rowid FROM tasks WHERE assignment_id = ? AND completed = 0", (assignment_id,))
        if self.c.fetchone() is None or code != CompletionCode.completed:
            serverlog.debug('completing assignment {} {}'.format(assignment_id, code))
            self.c.execute("UPDATE assignments SET completed = ? WHERE assignment_id = ?", (code.value, assignment_id,))
            self.conn.commit()

            return assignment_id
        else:
            self.conn.commit()
            return None

    def complete_assignment(self, assignment_id, code, recurse_tasks=True):
        if isinstance(code, int):
            code = CompletionCode(code)
        self.c.execute("UPDATE assignments SET completed = ? WHERE assignment_id = ?",
            (code.value, assignment_id,))
        if recurse_tasks:
            self.c.execute("UPDATE tasks SET completed = ? WHERE assignment_id = ? AND completed = 0",
                (code.value, assignment_id,))
        self.conn.commit()


    def get_completion_code(self, assignment_id):
        self.c.execute("SELECT completed FROM assignments WHERE assignment_id = ?", (assignment_id,))
        res = self.c.fetchone()
        if res is None:
            return CompletionCode.active # TODO: check this

        try:
            return CompletionCode(res[0])
        except:
            return CompletionCode.generic_error

    def get_token_response(self, assignment_id=None, token=None):
        if token is None and assignment_id is not None:
            self.c.execute("SELECT token FROM assignments WHERE assignment_id = ?", (assignment_id,))
            res = self.c.fetchone()
            if res is None:
                return None
            token = res[0]
        elif token is None and assignment_id is None:
            raise ValueError("need assignment_id or token")
        return hmac.new(self.SECRET, binascii.unhexlify(token)).hexdigest()

ASSIGNMENT_TO_PORTAL_WS = {}
WORKER_TO_DASHBOARD_WS = {}

class RootHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("""
        If you have been linked here, please report this error.
        All content on this server is at a different URL.
        """)

class DevPortalHandler(tornado.web.RequestHandler):
    def get(self):
        pool = self.get_argument('pool', default=None)
        confirm_pool = self.get_argument('confirm_pool', default="")
        if pool is None:
            self.write("""
<html>
    <body>
        <form action="/devportal" method="get">
            Pool name: <input name="pool" value="{}"></input>
            <button>Submit</button>
        </form>
    </body>
</html>
            """.format(confirm_pool))
            return
        random_id = '%010x' % random.randrange(16**10)
        assignment_id = 'test_A_' + random_id
        worker_id = 'test_W_' + random_id
        self.redirect('/portal?pool={}&token=1a&assignmentId={}&workerId={}&turkSubmitTo=submit'.format(pool, assignment_id, worker_id))

class SubmitHandler(tornado.web.RequestHandler):
    def post(self):
        pool = self.get_argument('pool', None)
        if pool is not None:
            self.redirect('/devportal?confirm_pool={}'.format(pool))
        else:
            self.redirect('/devportal')

class PortalHandler(tornado.web.RequestHandler):
    def get(self):
        pool = self.get_argument('pool', default=None)
        token = self.get_argument('token', default=None)
        assignment_id = self.get_argument('assignmentId', default=None)
        hit_id = self.get_argument('hitID', default=None)
        turk_submit_to = self.get_argument('turkSubmitTo', default=None)
        worker_id = self.get_argument('workerId', default=None)

        if pool is None:
            return self.render(
                "error.html",
                error_msg="This HIT was not configured correctly (missing pool name)"
            )
        elif token is None:
                return self.render(
                    "error.html",
                    error_msg="This HIT was not configured correctly (missing secret token)"
                )
        elif assignment_id is None:
            return self.render(
                "error.html",
                error_msg="No assigment id specified"
            )
        elif assignment_id == "ASSIGNMENT_ID_NOT_AVAILABLE":
            pool_data = DB.get_pool_data(pool)
            if pool_data is None:
                return self.render(
                    "error.html",
                    error_msg="An internal error occured. Please do not accept this task."
                )

            return self.render(
                "portal.html",
                accepted=False,
                assignment_id=None,
                worker_id=None,
                turk_submit_to=None,
                completion_code=CompletionCode.active,
                CompletionCode=CompletionCode,
                token="",
                token_response="",
                pool_data = pool_data,
                pool=pool,
            )
        elif turk_submit_to is None:
            return self.render(
                "error.html",
                error_msg="No turk_submit_to specified"
            )
        elif worker_id is None:
            return self.render(
                "error.html",
                error_msg="No worker ID was submitted with the job. This error should never occur"
            )
        else:
            pool_data = DB.get_pool_data(pool)
            if pool_data is None:
                return self.render(
                    "error.html",
                    error_msg="An internal error occured. Please return this task."
                )
            DB.assign_worker(assignment_id, worker_id, pool, token)
            if worker_id in WORKER_TO_DASHBOARD_WS:
                handler = WORKER_TO_DASHBOARD_WS[worker_id]
                if handler is not None:
                    handler.send_assignment_list()
            return self.render(
                "portal.html",
                accepted=True,
                completion_code=DB.get_completion_code(assignment_id),
                CompletionCode=CompletionCode,
                assignment_id=assignment_id,
                worker_id=worker_id,
                turk_submit_to=turk_submit_to,
                token=token,
                token_response=DB.get_token_response(token=token),
                pool_data = pool_data,
                pool=pool,
            )

class DashboardHandler(tornado.web.RequestHandler):
    def get(self):
        worker_id = self.get_argument('workerId', default=None)
        preview = self.get_argument('preview', default=None)

        if worker_id is None and preview is None:
            return self.render(
                "error.html",
                error_msg="You are not logged in as a worker. Please click a link from Mechanical Turk to go to the correct page"
            )

        return self.render(
            "dashboard.html",
            worker_id=worker_id
        )


class DashboardWSHandler(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        """
        Called when a client opens the websocket
        """
        self.worker_id = self.get_argument('workerId', default=None)
        self.active = False

        if self.worker_id is None:
            serverlog.warning("[dashboard] Rejecting connection without a worker id")
            self.close()
            return
        serverlog.debug("[dashboard] [wId: {}] open called {}".format(self.worker_id, self))

        global WORKER_TO_DASHBOARD_WS
        worker_connection = WORKER_TO_DASHBOARD_WS.setdefault(self.worker_id, None)
        if worker_connection is None:
            serverlog.debug("[dashboard] [wId: {}] setting as active {}".format(self.worker_id, self))
            self.active = True
            WORKER_TO_DASHBOARD_WS[self.worker_id] = self
        else:
            serverlog.debug("[dashboard] [wId: {}] prompting regarding second dashboard {}".format(self.worker_id, self))
            self.prompt_replace()

    def on_close(self):
        """
        Called when the websocket closes
        """
        serverlog.debug("[dashboard] [wId: {}] closed called".format(self.worker_id))

        if self.worker_id is None:
            return

        global WORKER_TO_DASHBOARD_WS
        if self.worker_id in WORKER_TO_DASHBOARD_WS and WORKER_TO_DASHBOARD_WS[self.worker_id] is self:
            WORKER_TO_DASHBOARD_WS[self.worker_id] = None
        else:
            serverlog.debug("[dashboard] [wId: {}] Warning, can't remove self".format(self.worker_id))

    ### Websocket-related

    def on_message(self, message):
        """
        Called when a message is received over the websocket
        """

        obj = json.loads(message)
        msg_type = obj["type"]
        data = obj["data"]

        if msg_type in [ "subscribe", "unsubscribe", "publish"]:
            serverlog.error("This is not a websocket bridge")
            return
        elif msg_type == "get_task":
            self.send_task()
        elif msg_type == "submit_task":
            self.server_side_submit(data)
        elif msg_type == "replace":
            self.replace_other()
        else:
            raise Exception("Invalid websocket message type: " + msg_type)

    def ws_send(self, type, data):
        """
        Convenience method for sending data over the websocket
        """
        self.write_message(json.dumps({"type": type, "data": data}))

    def prompt_replace(self):
        self.ws_send("prompt_replace", {})

    def prompt_returned(self):
        if not self.active:
            return
        self.ws_send("prompt_returned", {})

    def prompt_assignment_completed(self):
        if not self.active:
            return
        self.ws_send("prompt_assignment_completed", {})

    def replace_other(self):
        serverlog.debug("replacing other {}".format(self))
        global WORKER_TO_DASHBOARD_WS
        worker_connection = WORKER_TO_DASHBOARD_WS.setdefault(self.worker_id, None)
        serverlog.debug("candidate for replacement: {}".format(worker_connection))
        if worker_connection is not None:
            worker_connection.close()

        WORKER_TO_DASHBOARD_WS[self.worker_id] = self
        self.active = True
        self.send_assignment_list()

    def send_assignment_list(self):
        if not self.active:
            return

        assignment_id = DB.get_assignment(self.worker_id)

        all_tasks = DB.get_all_tasks(self.worker_id)

        uncompleted_tasks = [task for task in all_tasks if not task['completed']]

        def get_progress(assignment):
            done = 0
            total = 0
            for task in all_tasks:
                if task['assignment_id'] != assignment:
                    continue
                if task['completed']:
                    done += 1
                total += 1
            return (done, total)

        def get_assignment(assignment):
            assignment_data = DB.get_assignment_pool_data(assignment)
            return {
                'name': assignment_data["name"] + (
                    " ({}/{} done)".format(*get_progress(assignment)) if assignment == assignment_id else ""
                    ),
                'description': assignment_data["description"],
                'completed': False
                }

        assignments = []
        if assignment_id is not None:
            assignments.append(get_assignment(assignment_id))
            processed = {assignment_id}
        else:
            processed = set()

        for task in uncompleted_tasks:
            if task['assignment_id'] not in processed:
                assignments.append(get_assignment(task['assignment_id']))
            processed.add(task['assignment_id'])


        self.ws_send("assignment_list", {
            'loaded': True,
            'assignments': assignments
        })

    def send_task(self):
        if not self.active:
            return

        self.send_assignment_list()
        task = DB.get_task(self.worker_id)
        if task is None:
            return

        self.ws_send("task", task)

    def server_side_submit(self, data):
        if not self.active:
            return

        if not hasattr(self, 'worker_id'):
            serverlog.error("Missing worker id during submit!")
            self.worker_id = None

        serverlog.info("[wId: {}] Marking task as completed on server".format(self.worker_id))
        task_id = data['task_id']
        if 'code' in data:
            code = data['code']
            try:
                if isinstance(code, str):
                    code = CompletionCode[code]
                elif isinstance(code, int):
                    code = CompletionCode(code)
            except:
                code = CompletionCode.generic_error
        else:
            code = CompletionCode.completed

        if 'task_id' in data:
            del data['task_id']
        if 'code' in data:
            del data['code']
        if 'submissionTimestamp' not in data:
            data['submissionTimestamp'] = datetime.now(timezone.utc).astimezone().isoformat()

        completed_assignment_id = DB.complete_task(task_id, data, code)
        if completed_assignment_id:
            serverlog.info("[wId: {}; aId: {}] Completing this task has also completed the assignment".format(self.worker_id, completed_assignment_id))
            self.prompt_assignment_completed()

        if code == CompletionCode.completed and completed_assignment_id is not None and completed_assignment_id in ASSIGNMENT_TO_PORTAL_WS:
            token_response = DB.get_token_response(completed_assignment_id)
            if token_response is not None:
                for connection in ASSIGNMENT_TO_PORTAL_WS[completed_assignment_id]:
                    connection.send_submit(token_response)
            else:
                serverlog.error("[wId: {}] Cannot auto-submit due to bad token response".format(self.worker_id))

class PortalWSHandler(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        """
        Called when a client opens the websocket
        """
        self.assignment_id = self.get_argument('assignmentId', default=None)

        if self.assignment_id is None:
            serverlog.info("[portal] Rejecting connection without an assignment id")
            self.close()
            return
        serverlog.debug("[portal] [aId: {}] open called".format(self.assignment_id))

        global ASSIGNMENT_TO_PORTAL_WS
        assignment_connections = ASSIGNMENT_TO_PORTAL_WS.setdefault(self.assignment_id, [])
        assignment_connections.append(self)


    def on_close(self):
        """
        Called when the websocket closes
        """
        if not hasattr(self, 'assignment_id'):
            serverlog.debug("[portal] Websocket being closed without having ever been opened!")
        serverlog.debug("[portal] [aId: {}] closed called".format(self.assignment_id))
        if self.assignment_id is None:
            return
        global ASSIGNMENT_TO_PORTAL_WS
        assignment_connections = ASSIGNMENT_TO_PORTAL_WS.setdefault(self.assignment_id, [])
        if self not in assignment_connections:
            serverlog.debug("[portal] [aId: {}] Warning, can't remove self".format(self.assignment_id))
            return
        assignment_connections.remove(self)


    def send_submit(self, token_response):
        self.write_message(json.dumps({"type": "submit", "token_response": token_response}))


def make_app():
    return tornado.web.Application([
        (r"/", RootHandler),
        (r"/portal", PortalHandler),
        (r"/portal/ws", PortalWSHandler),
        (r"/devportal", DevPortalHandler),
        (r"/submit/mturk/externalSubmit", SubmitHandler),
        (r"/dashboard", DashboardHandler),
        (r"/ws", DashboardWSHandler),
    ],
        template_path=os.path.join(os.path.dirname(__file__), "templates"),
        static_path=os.path.join(os.path.dirname(__file__), "static"),
        debug=True,
    )

if __name__ == "__main__":
    tornado.options.parse_command_line()

    DB = AssignmentDatabase(options.db)

    app = make_app()

    if os.path.isfile(options.cert) and os.path.isfile(options.key):
        ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_ctx.load_cert_chain(options.cert, options.key)
        serverlog.info("Using https")
    else:
        ssl_ctx = None
        serverlog.info("Not using https")

    load_sqs_credentials()

    messaging_thread = Thread(target=receive_messages, name="sqs-msgs", daemon=True)
    messaging_thread.start()

    http_server = tornado.httpserver.HTTPServer(app, ssl_options=ssl_ctx)
    http_server.listen(options.port)

    tornado.ioloop.IOLoop.current().start()
