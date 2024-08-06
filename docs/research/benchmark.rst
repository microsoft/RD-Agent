==============================
Benchmark
==============================

Introduction
=============


Benchmarking the capabilities of the R&D is a very important research problem of the research area.

Currently we are continuously exploring how to benchmark them.

The current benchmarks are listed in this page


Development Capability Benchmarking
===================================


Benchmark is used to evaluate the effectiveness of factors with fixed data.

It mainly includes the following steps:

1. :ref:`read and prepare the eval_data <data>`

2. :ref:`declare the method to be tested and pass the arguments <config>`

3. :ref:`declare the eval method and pass the arguments <config>`

4. :ref:`run the eval <run>` 

5. :ref:`save and show the result <show>` 

Configuration 
-------------
.. _config:

.. autopydantic_settings:: rdagent.components.benchmark.conf.BenchmarkSettings

Example
++++++++
.. _example:

The default value for ``bench_test_round`` is 10, and it will take about 2 hours to run 10 rounds.
To modify it from ``10`` to ``2`` you can adjust this by adding environment variables in the .env file as shown below.

.. code-block:: Properties

      BENCHMARK_BENCH_TEST_ROUND=1

Data Format
-------------
.. _data:

The sample data in ``bench_data_path`` is a dictionary where each key represents a factor name. 

The value associated with each key is factor data containing the following information:

- **description**: A textual description of the factor.
- **formulation**: A LaTeX formula representing the model's formulation.
- **variables**: A dictionary of variables involved in the factor.
- **Category**: The category or classification of the factor.
- **Difficulty**: The difficulty level of implementing or understanding the factor.
- **gt_code**: A piece of code associated with the factor.

Here is the example of this data format:

.. literalinclude:: ../../rdagent/components/benchmark/example.json
   :language: json

Run Benchmark
-------------
.. _run:

Start benchmark after finishing the :doc:`../installation_and_configuration`.

.. code-block:: Properties

      python rdagent/app/quant_factor_benchmark/eval.py



Once completed, a pkl file will be generated, and its path will be printed on the last line of the console.

Show Result
-------------
.. _show:

The ``analysis.py`` script is used to read data from pkl and convert it to an image.
Modify the python code in ``rdagent/app/quant_factor_benchmark/analysis.py`` to specify the path to the pkl file and the output path for the png file.

.. code-block:: Properties

      python rdagent/app/quant_factor_benchmark/analysis.py

A png file will be saved to the designated path as shown below.

.. image:: ../_static/benchmark.png


Related Paper
-------------

- `Towards Data-Centric Automatic R&D <https://arxiv.org/abs/2404.11276>`_:
  We have developed a comprehensive benchmark called RD2Bench to assess data and model R&D capabilities. This benchmark includes a series of tasks that outline the features or structures of models. These tasks are used to evaluate the ability of LLM-Agents to implement them.

.. code-block:: bibtex

    @misc{chen2024datacentric,
        title={Towards Data-Centric Automatic R&D},
        author={Haotian Chen and Xinjie Shen and Zeqi Ye and Wenjun Feng and Haoxue Wang and Xiao Yang and Xu Yang and Weiqing Liu and Jiang Bian},
        year={2024},
        eprint={2404.11276},
        archivePrefix={arXiv},
        primaryClass={cs.AI}
    }

.. image:: https://github.com/user-attachments/assets/494f55d3-de9e-4e73-ba3d-a787e8f9e841
