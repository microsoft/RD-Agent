===============================
Framework Design & Components
===============================

Framework & Components
=========================

.. NOTE: This depends on the correctness of `c-v` of github.

.. image:: _static/Framework-RDAgent.png
    :alt: Components & Feature Level

The image above shows the overall framework of RDAgent.

In a data mining expert's daily research and development process, they propose a hypothesis (e.g., a model structure like RNN can capture patterns in time-series data), design experiments (e.g., finance data contains time-series and we can verify the hypothesis in this scenario), implement the experiment as code (e.g., Pytorch model structure), and then execute the code to get feedback (e.g., metrics, loss curve, etc.). The experts learn from the feedback and improve in the next iteration.

We have established a basic method framework that continuously proposes hypotheses, verifies them, and gets feedback from the real world. This is the first scientific research automation framework that supports linking with real-world verification.


.. image:: https://github.com/user-attachments/assets/60cc2712-c32a-4492-a137-8aec59cdc66e
    :alt: Class Level Figure

The figure above shows the main classes and how they fit into the workflow for those interested in the detailed code.


.. Detailed Design
.. ===============
