==============
User Interface
==============


Introduction
============

RD-Agent will generate some logs during the R&D process. These logs are very useful for debugging and understanding the R&D process. However, just viewing the terminal log is not intuitive enough. RD-Agent provides a web app as UI to visualize the R&D process. You can easily view the R&D process and understand the R&D process better.

A Quick Demo
============

Start Web App
-------------

In `RD-Agent/` folder, run:

.. code-block:: bash

    streamlit run rdagent/log/ui/app.py --server.port <port> -- --log_dir <log_dir>

This will start a web app on `http://localhost:<port>`.

**NOTE**: The log_dir parameter is not required. You can manually enter the log_path in the web app. If you set the log_dir parameter, you can easily select a different log_path in the web app.

Use Web App
-----------

1. Open the sidebar.

.. TODO: update these

2. Select the scenario you want to show. There are some pre-defined scenarios:
    - Qlib Model
    - Qlib Factor
    - Data Mining
    - Model from Paper

3. Click the `Config⚙️` button and input the log path (if you set the log_dir parameter, you can select a log_path in the dropdown list).

4. Click the buttons below Config⚙️ to show the scenario execution process. Buttons are:
    - All Loops: Show complete scenario execution process.
    - Next Loop: Show one success **R&D Loop**.
    - One Evolving: Show one **evolving** step of **development** part.
    - refresh logs: clear shown logs.
