==============================
Benchmark
==============================

Introduction
=============

Benchmark is used to evaluate the effectiveness of code with fixed data.

It mainly includes the following steps:

1.read and prepare the eval_data

2.declare the method to be tested and pass the arguments

3.declare the eval method and pass the arguments

4.run the eval

5.save and show the result


Run Benchmark
=============

Start benchmark after finishing the `installation_and_configuration <installation_and_configuration.rst>`_.

.. code-block:: Properties

      python rdagent/app/quant_factor_benchmark/eval.py

By default, it will take about 2 hours to run 10 rounds, but you can adjust this by adding environment variables in the .env file as shown below.

.. code-block:: Properties

      BENCHMARK_BENCH_TEST_ROUND=1

Once completed, a pkl file will be generated, and its path will be printed on the last line of the console.

Show Result
=============
The ``analysis.py`` script is used to read data from pkl and convert it to an image.
Modify the python code in ``rdagent/app/quant_factor_benchmark/analysis.py`` to specify the path to the pkl file and the output path for the png file.

.. code-block:: Properties

      python rdagent/app/quant_factor_benchmark/analysis.py

A png file will be saved to the designated path as shown below.

.. image:: docs/_static/benchmark.png




