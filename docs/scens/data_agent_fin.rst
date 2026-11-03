.. _data_agent_fin:

=====================
Finance Data Agent
=====================


**🤖 Automated Quantitative Trading & Iterative Factors Evolution**
-------------------------------------------------------------------

📖 Background
~~~~~~~~~~~~~~
In the dynamic world of quantitative trading, **factors** serve as the strategic tools that enable traders to exploit market inefficiencies. 
These factors—ranging from simple metrics like price-to-earnings ratios to complex models like discounted cash flows—are the key to predicting stock prices with a high degree of accuracy.

By leveraging these factors, quantitative traders can develop sophisticated strategies that not only identify market patterns but also significantly enhance trading efficiency and precision. 
The ability to systematically analyze and apply these factors is what separates ordinary trading from truly strategic market outmaneuvering.
And this is where the **Finance Model Agent** comes into play.

🎥 `Demo <https://rdagent.azurewebsites.net/factor_loop>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

    <div style="display: flex; justify-content: center; align-items: center;">
      <video width="600" controls>
        <source src="https://rdagent.azurewebsites.net/media/65bb598f1372c1857ccbf09b2acf5d55830911625048c03102291098.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </div>


🌟 Introduction
~~~~~~~~~~~~~~~~
In this scenario, our agent illustrates the iterative process of hypothesis generation, knowledge construction, and decision-making. 

It highlights how financial factors evolve through continuous feedback and refinement. 

Here's an enhanced outline of the steps:

**Step 1 : Hypothesis Generation 🔍**

- Generate and propose initial hypotheses based on previous experiment analysis and domain expertise, with thorough reasoning and financial justification.

**Step 2 : Factor Creation ✨**

- Based on the hypothesis, divide the tasks.
- Each task involves developing, defining, and implementing a new financial factor, including its name, description, formulation, and variables.

**Step 3 : Factor Implementation 👨‍💻**

- Implement the factor code based on the description, evolving it as a developer would.
- Quantitatively validate the newly created factors.

**Step 4 : Backtesting with Qlib 📉**

- Integrate the full dataset into the factor implementation code and prepare the factor library.
- Conduct backtesting using the Alpha158 plus newly developed factors and LGBModel in Qlib to evaluate the new factors' effectiveness and performance.

+----------------+------------+----------------+----------------------------------------------------+
| Dataset        | Model      | Factors        | Data Split                                         |
+================+============+================+====================================================+
| CSI300         | LGBModel   | Alpha158 Plus  | +-----------+--------------------------+           |
|                |            |                | | Train     | 2008-01-01 to 2014-12-31 |           |
|                |            |                | +-----------+--------------------------+           |
|                |            |                | | Valid     | 2015-01-01 to 2016-12-31 |           |
|                |            |                | +-----------+--------------------------+           |
|                |            |                | | Test      | 2017-01-01 to 2020-08-01 |           |
|                |            |                | +-----------+--------------------------+           |
+----------------+------------+----------------+----------------------------------------------------+


**Step 5 : Feedback Analysis 🔍**

- Analyze backtest results to assess performance.
- Incorporate feedback to refine hypotheses and improve the model.

**Step 6 :Hypothesis Refinement ♻️**

- Refine hypotheses based on feedback from backtesting.
- Repeat the process to continuously improve the model.

⚡ Quick Start
~~~~~~~~~~~~~~~~~

Please refer to the installation part in :doc:`../installation_and_configuration` to prepare your system dependency.

You can try our demo by running the following command:

- 🐍 Create a Conda Environment

  - Create a new conda environment with Python (3.10 and 3.11 are well tested in our CI):

    .. code-block:: sh

          conda create -n rdagent python=3.10

  - Activate the environment:

    .. code-block:: sh

        conda activate rdagent

- 📦 Install the RDAgent
  
  - You can install the RDAgent package from PyPI:

    .. code-block:: sh

        pip install rdagent

- 🚀 Run the Application
    
  - You can directly run the application by using the following command:
    
    .. code-block:: sh

        rdagent fin_factor


Use saved Adanos reports as optional extra data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This offline recipe adds a separate ``adanos_reddit.h5`` input to already
prepared Full and Debug research folders. It does not start an agent, download
prices, or change Qlib's market. `Adanos <https://adanos.org/>`_ is an optional
commercial API with a limited free plan. Check the `API documentation
<https://api.adanos.com/docs>`_ for access, history limits, and data terms.
RD-Agent does not require an Adanos account to run this recipe or its tests.

Prepare price data and matching instrument identities before using extra data.
The default Qlib example uses CN instruments and historical experiment periods.
Do not map US stock reports onto similarly named CN listings. Use exact
instrument identifiers from your own matching US-market dataset, with prices
and research periods that overlap your observations. The example map below
assumes that your prepared US provider names its instrument ``AAPL``. Verify
that identity yourself. The recipe does not infer identities or create prices.

Keep saved JSON outside the research data folders. For separately authorized
acquisition, the endpoint is
``GET https://api.adanos.com/reddit/stocks/v1/stock/{ticker}``, with inclusive
UTC ``from`` and ``to`` dates and an ``X-API-Key`` header read from
``ADANOS_API_KEY``. The following command is acquisition documentation only.
It is not part of the offline recipe or tests. All dates in this subsection
are illustrative, not claims about provider coverage or plan access.

.. code-block:: sh

    curl --fail --get 'https://api.adanos.com/reddit/stocks/v1/stock/AAPL' \
      --header "X-API-Key: ${ADANOS_API_KEY:?set outside the recipe}" \
      --data-urlencode 'from=2026-05-04' --data-urlencode 'to=2026-05-06' \
      --output aapl-report.json

Only use a saved successful stock report. Authentication failures, unsupported
tickers, historical-limit errors, rate limits, validation errors, and service
errors are not neutral observations. Do not turn error responses into zeros.
Do not include keys or redistribute actual exports with a contribution.
Record the request window and the actual snapshot observation time separately.
An aggregate's UTC day label is not its publication timestamp. A later-fetched
aggregate is not proven available at an earlier market close. A lag alone does
not resolve revisions or availability bias. This recipe makes no point-in-time,
alpha, or performance claim.

Normalize one or more saved report objects with an explicit, one-to-one
ticker-to-instrument map. The frame has a sorted ``datetime, instrument``
MultiIndex and three numeric columns. ``$adanos_sentiment`` is a score in
[-1, 1], not an expected return or probability. ``$adanos_mentions`` counts
attributed mentions, including inherited thread-context mentions.
``$adanos_buzz`` is an activity score in [0, 100], not a return magnitude.
Missing sentiment and buzz stay NaN. Missing dates stay absent, without
calendar filling, forward filling, or zero imputation. Overlapping reports
with duplicate instrument/date observations fail rather than select a revision.

.. code-block:: python
    :name: adanos-offline-recipe

    import json
    import math
    import os
    from datetime import date
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import pandas as pd


    def to_adanos_frame(reports: list[dict], instrument_map: dict[str, str]) -> pd.DataFrame:
        if not isinstance(reports, list) or not isinstance(instrument_map, dict):
            raise ValueError("Use a list of report objects and a ticker-to-instrument dictionary")
        if any(
            not isinstance(value, str) or not value or value.strip() != value
            for pair in instrument_map.items()
            for value in pair
        ) or len(set(instrument_map.values())) != len(instrument_map):
            raise ValueError("Instrument identities must be explicit, nonempty, and one-to-one")
        fields = [
            ("sentiment_score", "$adanos_sentiment", -1, 1),
            ("mentions", "$adanos_mentions", 0, math.inf),
            ("buzz_score", "$adanos_buzz", 0, 100),
        ]
        rows = []
        for report in reports:
            if not isinstance(report, dict) or "detail" in report or "error" in report:
                raise ValueError("Expected a successful saved stock report, not an error response")
            ticker = report.get("ticker")
            if not isinstance(ticker, str) or ticker not in instrument_map:
                raise ValueError("Every report ticker needs an explicit instrument mapping")
            trend = report.get("daily_trend")
            if trend is None:
                trend = []
            if not isinstance(trend, list) or type(report.get("found", True)) is not bool:
                raise ValueError("Invalid daily_trend or v1 found flag")
            if report.get("found") is False and trend:
                raise ValueError("A no-data report cannot contain daily observations")
            for daily in trend:
                if not isinstance(daily, dict):
                    raise ValueError("Expected daily observation objects")
                day = daily.get("date")
                if not isinstance(day, str) or date.fromisoformat(day).isoformat() != day:
                    raise ValueError("Expected an ISO UTC day label in YYYY-MM-DD format")
                values = []
                for field, _, lower, upper in fields:
                    value = daily.get(field)
                    if field == "mentions" and (type(value) is not int or value < 0):
                        raise ValueError("Daily mentions must be a nonnegative integer")
                    if value is not None and (
                        type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper
                    ):
                        raise ValueError(f"Invalid {field}")
                    values.append(float("nan") if value is None else float(value))
                rows.append((pd.Timestamp(day), instrument_map[ticker], *values))
        columns = [column for _, column, _, _ in fields]
        frame = pd.DataFrame(rows, columns=["datetime", "instrument", *columns])
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame = frame.set_index(["datetime", "instrument"]).astype("float64").sort_index()
        if frame.index.has_duplicates:
            raise ValueError("Duplicate instrument/date observations")
        return frame


    def write_adanos_extra(frame: pd.DataFrame, folder: Path) -> Path:
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError("Prepare the research folder first")
        destination = folder / "adanos_reddit.h5"
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        with TemporaryDirectory(prefix=".adanos-", dir=folder) as staging:
            staged = Path(staging) / destination.name
            frame.to_hdf(staged, key="data", mode="w")
            os.link(staged, destination)
        return destination

The writer requires a local filesystem with hard-link support. It stages a
complete HDF and publishes it without replacing an existing file, including
symlinks or a destination created by another process during the write.
It never writes ``daily_pv.h5`` or ``result.h5``. On a rerun, choose a fresh
research folder rather than overwrite a previous snapshot.

For an offline demonstration, save this synthetic object as
``aapl-report.json`` outside the research folders. These values are invented.

.. code-block:: json
    :name: adanos-offline-report

    {
      "ticker": "AAPL",
      "found": true,
      "daily_trend": [
        {"date": "2026-05-06", "mentions": 4, "sentiment_score": 0.5, "buzz_score": 20.0},
        {"date": "2026-05-04", "mentions": 0, "sentiment_score": null, "buzz_score": null}
      ]
    }

Save your verified mapping as ``instrument-map.json``. For this synthetic
example, its content is ``{"AAPL": "AAPL"}``. With the definitions above in the
same Python session, read the saved files and add the extra input to both
already-prepared folders. Replace the folder paths with your own locations.
The second write can fail independently. An existing first output stays intact.

.. code-block:: python
    :name: adanos-offline-usage

    reports = [json.loads(Path("aapl-report.json").read_text(encoding="utf-8"))]
    instrument_map = json.loads(Path("instrument-map.json").read_text(encoding="utf-8"))
    frame = to_adanos_frame(reports, instrument_map)
    for folder in (Path("research/full"), Path("research/debug")):
        write_adanos_extra(frame, folder)
    extra = pd.read_hdf("research/debug/adanos_reddit.h5", key="data")
    pd.testing.assert_frame_equal(extra, frame)

For multiple reports, load each JSON file into the ``reports`` list. An absent,
null, or empty ``daily_trend`` yields no rows, including a supported ticker with
``found=false``. It does not create a neutral observation. The v1 ``found`` flag
is compatibility metadata, not an instrument identity.

Set ``FACTOR_CoSTEER_data_folder`` and ``FACTOR_CoSTEER_data_folder_debug`` to
your prepared Full and Debug folders for later, separately configured research.
Both must exist before folder discovery. Do not call
``generate_data_folder_from_qlib()`` to prepare this example. That path generates
the default price dataset and can replace existing folder contents.
The native folder description reads HDF files, and factor execution links
the selected folder's files into its workspace. Keep the extra file in both
folders and describe its columns, provenance, missing values, observation time,
and market alignment in your own folder documentation. Leave existing prices
and ``result.h5`` untouched. Reading the HDF above needs no model or agent run.


🛠️ Usage of modules
~~~~~~~~~~~~~~~~~~~~~

.. _Env Config: 

- **Env Config**

The following environment variables can be set in the `.env` file to customize the application's behavior:

.. autopydantic_settings:: rdagent.app.qlib_rd_loop.conf.FactorBasePropSetting
    :settings-show-field-summary: False
    :exclude-members: Config

.. autopydantic_settings:: rdagent.components.coder.factor_coder.config.FactorCoSTEERSettings
    :settings-show-field-summary: False
    :members: coder_use_cache, data_folder, data_folder_debug, file_based_execution_timeout, select_method, max_loop, knowledge_base_path, new_knowledge_base_path
    :exclude-members: Config, fail_task_trial_limit, v1_query_former_trace_limit, v1_query_similar_success_limit, v2_query_component_limit, v2_query_error_limit, v2_query_former_trace_limit, v2_error_summary, v2_knowledge_sampler
    :no-index:
