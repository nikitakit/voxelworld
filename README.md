# voxelworld

This repository has code associated with interpreting spatial descriptors.

The dataset and all code necessary to replicate our results will be distributed separately.

The code in this repository mostly relates to the data collection and visualization required for our work. It includes an in-browser renderer for voxel scenes (which we used for both data collection and for visualizing model behavior during development). The remainder of this README assumes that you are interested in understanding our data collection pipeline.

## Architecture

Our code follows a microservices architecture. It is designed such that individual pieces can be restarted independently of each other, for a faster development workflow. Modules written in Python communicate with each other via LCM, using protobufs for the message format.

We also have a dev dashboard that runs inside the browser: it uses the LCMProto Websocket Bridge to also connect to LCM channels.

Data collection using Amazon Mechanical Turk requires hosting our own server, the code for which is in the `mturk_server` folder. This server is designed to run independently of the LCM system, since it will typically be hosted on a dedicated machine.

Summary of the code in this repository:
* `lcmproto/`: part of our microservices RPC system
* `learning/`: our model for the 3D spatial descriptors task
* `lrpc/`: part of our microservices RPC system
* `mturk_server/`: serves interactive tasks to MTurk workers
* `mturk_submit/`: scripts for creating MTurk HITs via the API
* `proto/`: protobuf types definitions, used throughout the system
* `task_tools/`: tools for creating tasks and processing responses to them
* `toy2d/`: (standalone) a simplified 2D version of our task. No guarantees of code here being in a working state.
* `util/`: Misc. utilities
* `voxel-ui/`: The in-browser renderer for voxel scenes
* `world-service/`: Used for reading/writing Minecraft save files. Not very relevant at the moment because we chose to use synthetic scenes rather than Minecraft save files downloaded from the Internet.

## Dependencies

Python 3.5 or later is required to run this code. Most python dependencies are specified in `requirements.txt`.

Additionally, you will need :
* The protobuf compiler, `protoc`
* LCM (https://github.com/lcm-proj/lcm), with Python 3 bindings enabled
* mceditlib (https://github.com/mcedit/mcedit2/tree/4652e537aec5e11a57ca196586aacdb2de32448c). Run `setup_mceditlib.py` from that repository to install it.
* tmux and teamocil

To install javascript dependencies, run `npm install` inside `voxel-ui/`.

If you want to run just the `mturk_server` on a dedicated machine, you do not strictly need protobuf, LCM, or mceditlib. The server stores all of its data in a sqlite database. However, note that all three are needed to insert tasks into the database, and that protobuf is required to extract tasks from the database (you can choose to work around this by copying the database between machines).

## Running the dev system

To start the data collection/inspection side of our codebase:
* Run `gen_types.sh`, which will compile our protobuf files
* Inside the `mturk_server` folder, create a file `aws_env.json`, and set its contents to the empty object, `{}`. To switch to running the mturk server in production, follow the instructions further down in this document.
* In `world_service/world_service.py`, replace the `PATHS` to your locations for Minecraft maps (make it empty if only using synthetic data).
* Start tmux, and run `run_nodes.sh` inside tmux. This will start a large number of microservices.

The following ports need to be available on your machine to run the code:
* 8080: Used for the development dashboard
* 8081: Used to host a development copy of the MTurk server
* 8000: Used by the LCM Websocket Bridge (part of our microservices message-passing setup)
* You also need to have a network interface that enables UDP multicast. See https://lcm-proj.github.io/multicast_setup.html

## Using the dev dashboard

The dev dashboard is, by default, hosted at http://localhost:8080

### Controls tab

In the Controls tab, you can request Minecraft Maps from the world service. Query the list of available worlds, select one, and hit "(Re)load world". You can then click into the world view on the left portion of the page, and use WASD/arrows/shift/space to navigate the scene. As long as dynamic chunk rendering is not disabled, the visible section of the world will update as you move around the map.

### Snapshots & Prep tab

The Snapshots & Prep tab helps you convert pieces of the world into tasks for crowd workers. Most tasks require that the camera start at a certain position in the world -- we call these saved camera positions Snapshots. You can save the current camera position with the "Snapshot!" button, and list existing positions with the "Search folder" button. Snapshots are stored on disk inside the `task_tools/snapshots` folder.

Task prep lets you convert snapshots to tasks for workers. Task data comes in two varieties: "base" and "processed". The base task is a compact representation that requires microservices to be available to fill in missing information (like the voxel data for the world), while the processed version is self-contained. The "Get Base" button creates a base task for a given game name (using the selected snapshot in the "Snapshots" section when a snapshot is required). "Process & Activate" processes the base task, and activates it in the dev dashboard. The "Process & Save" button instead saves to disk, using the name specified as a path, relative to `task_tools/tasks`.

The Responses section shows the last response from a task activated in the dev console. You can also enter a task name under the "Task" heading and hit "Load responses" to see responses from workers.

You can visualize tasks from our dataset by copying `tasks` and `task_responses` from the dataset distribution into `task_tools`. You can also generate your own synthetic tasks by running `task_tools/generate_whereis_tasks_synthetic.py`.

### Task tab

This is a preview of what workers will see on their screen. It only shows something when you activate a task, either through the "Snapshots & Prep" tab or via LCM.

## Using the mturk server

While the dev dashboard is used to test out tasks for workers, the `mturk_server` is what actually serves them in production. It is a standalone system backed by a sqlite database.

A group of tasks is known as a "pool" -- for a sample pool description, see `mturk_server/sample_pool.json`. The portal html fields are used in the iframe embedded inside the mturk page, while the description fields are used within the self-hosted interactive console. For each HIT a worker accepts, `count` tasks are allocated to them (this allows adjusting the granularity of HITs). Task templates are either a JSON object, or a string task name relative to `task_tools/tasks` (omitting the `.pb` extension).

A pool description is imported into the sqlite database using `python manage_tasks.py --create=path_to_file.json`. You can then go to http://localhost:8081/devportal, enter the pool name, and test the workflow locally to make sure it is as expected. You can run `python manage_tasks.py --extract=pool_name` to copy responses from the sqlite database into `task_tools/task_responses`. The responses will then be available via the task service. They can be retrieved using the "Load responses" button on the dev dashboard, or mass converted into CSV form. To get a CSV file, run `python responses_to_csv.py --path=pool_name`.

### Using the MTurk server in production

For actually collecting data, the mturk server needs to be run on a public server. This involves a few additional considerations.

The mturk server manages its own state for HITs, and only funnels the bare minimum information through Amazon. To work properly, it needs to know when HITs expire or are returned. This means that HITs need to be set up to send notifications via Amazon SQS, and the mturk server needs to be able to read these notifications.

See [Amazon's documentation](http://docs.aws.amazon.com/AWSMechTurk/latest/AWSMturkAPI/ApiReference_NotificationReceptorAPI_SQSTransportArticle.html) for how to set up an SQS queue and grant Mechanical Turk permissions to use that queue. Once a queue is set up, you also need to issue credentials that allow the mturk server read/write access to the queue. The credentials and queue name are stored in the `mturk_server/aws_env.json` file (see `mturk_server/aws_env_sample.json` for a template).

The `run_nodes.sh` starts one section that runs `npm run-script dashboard`. This watches the `voxel-ui` folder for changes, and writes a debug build to `mturk_server/static/dashboard-bundle.js`. The debug build is excessively large because it is unminified and contains source maps. For production, you should kill the debug script and run `npm run-script dashboard-min` instead. This will result in a significantly smaller `dashboard-bundle.js` that has to be transmitted to workers.

To create HITs, we have a script at `mturk_submit/hit_creation.py`. You should modify the `portal_url` and `queue_url` to match your usecase. Also be sure to specify the correct `pool` name. The file `mturk_submit/mturkconfig.json` is used to specify credentials and settings for the mturk library. You can use this file to toggle between the mturk sandbox and real mturk.

Note the use of `SECRET` in both `mturk_server/main.py` and `mturk_submit/hit_creation.py`. The two values must match, and should not be known to workers. The `SECRET` is used to make sure that workers actually have to complete the task before they can get paid.

Finally, the command for running the `mturk_server` in production is `python main.py --port=NUMBER_HERE --cert=PATH_TO_CERT --key=PATH_TO_KEY`. The port number, certificate, and private key are specified via command line arguments.

## Dataset Collection and Visualization

To collect the dataset, we followed roughly the following procedure (be sure that `run_nodes.sh` is active throughout):
* Run `task_tools/generate_whereis_tasks_synthetic.py` to generate some scenes for annotation. The "arguments" to the script are hardcoded at the top of the file (much of the code in this repo was written in an interactive style using [hydrogen](https://github.com/nteract/hydrogen), which is why some scripts still use constants at the top of the file instead of command-line arguments). There is an argument `REQUEST_FEEDBACK` which will send tasks as it generates them to the dev console for approval. In general, many of the scripts in the repo have options to send data to the dev console for visualization.
* Collect crowdsourced responses. `task_tools/responses_to_csv.py` can be used to dump the responses to CSV format. (Note that before running this script, responses need to be extracted from the sqlite database used by the mturk_server using `mturk_server/manag_tasks.py --extract=[pool_name]`).
* Run `task_tools/generate_whereis_guess_tasks.py` to convert some of these tasks to the test-set evaluation of choosing between Misty's true location and five distractors. Then crowdsourced more responses.
* The script `learning/responses_to_tfrecords.py` can convert responses to a data format usable for machine learning. It is a more general version of the `make_tfrecords.py` file shipped with the raw dataset distribution, in that it talks over RPC and can be configured via command-line options. Our final dataset conversion was done as per the raw dataset distribution, but a roughly equivalent command would be `python responses_to_tfrecords.py --path=whereis_synth3,whereis_synth4 --out=whereis.tfrecords --test_filename=dev_test_split.txt`.
* Code related to training various models on our data is in `learning/`
* The utility functions in `task_tools/whereis_inspection.py` are what was used to visualize model behavior during development. (These are used by `inspect_one()` in the model file.) These utilities allow sending Minecraft scenes with attached heatmaps over to the browser, which can then render the heatmaps in the same 3D renderer as the scenes.
