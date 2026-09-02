==============
User Interface
==============


Introduction
============

RD-Agent will generate some logs during the R&D process. These logs are very useful for debugging and understanding the R&D process. However, just viewing the terminal log is not intuitive enough. RD-Agent provides a web app as UI to visualize the R&D process. You can easily view the R&D process and understand the R&D process better.

Streamlit UI
============

Start Web App
-------------

In `RD-Agent/` folder, run:

.. code-block:: bash

    rdagent ui --port <port> --log-dir <log_dir like "log/"> [--debug]

This will start a web app on `http://localhost:<port>`.

**NOTE**: The log_dir parameter is not required. You can manually enter the log_path in the web app. If you set the log_dir parameter, you can easily select a different log_path in the web app.

--debug is optional, it will show a "Single Step Run" button in sidebar and saved objects info in the web app.

Use Web App
-----------

1. Open the sidebar.

.. TODO: update these

2. Select the scenario you want to show. There are some pre-defined scenarios:
    - Qlib Model
    - Qlib Factor
    - Data Mining
    - Model from Paper
    - Kaggle

3. Click the `Config⚙️` button and input the log path (if you set the log_dir parameter, you can select a log_path in the dropdown list).

4. Click the buttons below Config⚙️ to show the scenario execution process. Buttons are:
    - All Loops: Show complete scenario execution process.
    - Next Loop: Show one success **R&D Loop**.
    - One Evolving: Show one **evolving** step of **development** part.
    - refresh logs: clear shown logs.


Flask Web UI
============

RD-Agent also provides a separate frontend in ``web/`` backed by the Flask log
server started with ``rdagent server_ui``. This UI provides real-time trace,
upload, process-control, and user-interaction APIs.

Build and start
---------------

Install the frontend dependencies and build the static assets:

.. code-block:: bash

    cd web
    npm install
    npm run build:flask
    cd ..

The generated assets are served from ``./git_ignore_folder/static`` by default.
Set ``UI_STATIC_PATH`` before starting the server to use another directory.

Start the server locally:

.. code-block:: bash

    rdagent server_ui --port 19899

Then open ``http://127.0.0.1:19899``. The server listens on localhost by
default, so its process-control, upload, and trace APIs are not exposed to
other machines.

Remote access and authentication
--------------------------------

To access the Flask Web UI remotely, explicitly select a non-local address and
configure a long, random authentication token:

.. code-block:: bash

    export UI_SERVER_AUTH_TOKEN='<a-long-random-token>'
    rdagent server_ui --port 19899 --host 0.0.0.0

The server refuses to bind to a non-local address unless
``UI_SERVER_AUTH_TOKEN`` is set. Open the following URL once to establish an
authenticated browser session:

.. code-block:: text

    http://<server-host>:19899/?token=<a-long-random-token>

The server redirects to ``/`` after storing the token in an HTTP-only,
same-site cookie. API clients can supply the same token without using a cookie:

.. code-block:: text

    Authorization: Bearer <a-long-random-token>

Put remotely accessible deployments behind an HTTPS reverse proxy. Avoid
recording the initial token-bearing URL in proxy logs or sharing it through an
untrusted channel. When the server is started through the CLI, ``--host``
controls the listening address; ``UI_SERVER_HOST`` is the corresponding default
when invoking the backend entry point directly.

CORS is disabled by default. If a browser frontend is hosted on another origin,
configure an explicit JSON allowlist:

.. code-block:: bash

    export UI_CORS_ALLOWED_ORIGINS='["https://ui.example.com"]'

Configuration
-------------

The Flask Web UI supports the following environment variables:

.. list-table::
   :header-rows: 1
   :widths: 30 25 70

   * - Environment variable
     - Default
     - Description
   * - ``UI_STATIC_PATH``
     - ``./git_ignore_folder/static``
     - Directory containing the built Web UI assets.
   * - ``UI_TRACE_FOLDER``
     - ``./git_ignore_folder/traces``
     - Directory containing generated trace data and process logs.
   * - ``UI_UPLOAD_FOLDER``
     - ``./git_ignore_folder/uploads``
     - Isolated directory for uploaded inputs. Mount, back up, and clean it
       separately from the trace directory.
   * - ``UI_SERVER_HOST``
     - ``127.0.0.1``
     - Default host used by the backend entry point. Use ``server_ui --host``
       when starting the server through the CLI.
   * - ``UI_SERVER_AUTH_TOKEN``
     - empty
     - Bearer/cookie authentication token. Required for non-localhost bindings.
   * - ``UI_CORS_ALLOWED_ORIGINS``
     - ``[]``
     - JSON list of allowed browser origins. CORS is disabled when empty.
   * - ``UI_MAX_UPLOAD_MB``
     - ``20``
     - Maximum size in MiB of the complete HTTP request, including all files and
       form data.
   * - ``UI_LOAD_LEGACY_PICKLE_TRACES``
     - ``false``
     - Whether to deserialize persisted pickle traces at startup. Enable only
       for a fully trusted trace directory.

Upload and trace safety
-----------------------

Uploaded input files are stored outside ``UI_TRACE_FOLDER`` so they cannot be
discovered and deserialized as persisted traces. Uploads ending in ``.dill``,
``.pickle``, ``.pkl``, ``.py``, ``.pyc``, or ``.pyo`` are rejected. Workflows
using these formats as uploaded inputs must convert them to a non-executable
data format or provide them through another trusted mechanism.

Legacy pickle trace loading is disabled by default because deserializing a
pickle can execute code. After a restart, a historical trace may still appear
in the history list while its saved messages remain unloaded. To browse trusted
historical traces, opt in explicitly:

.. code-block:: bash

    export UI_LOAD_LEGACY_PICKLE_TRACES=true
    rdagent server_ui --port 19899

Only enable this setting when every file under ``UI_TRACE_FOLDER`` is trusted
and the directory is not writable by untrusted users or services.

Data-science trace share links no longer accept a URL-controlled ``log_folder``.
A link can preserve the selected trace, but the recipient must configure or
select the corresponding log folder in the UI.
