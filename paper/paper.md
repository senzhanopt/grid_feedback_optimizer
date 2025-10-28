---
title: 'Grid-feedback-optimizer: A Python package for feedback-based optimization of power grid operation'
tags:
  - Python
  - feedback optimization
  - real-time optimization
  - voltage control
  - congestion management
  - power grid operation

authors:
  - name: Sen Zhan
    orcid: 0000-0003-0405-5001
    affiliation: 1
affiliations:
 - name: Eindhoven University of Technology, The Netherlands
   index: 1

date: 26 October 2025
bibliography: paper.bib
---

# Summary


The increasing integration of distributed energy resources (DERs), such as photovoltaics, electric vehicles, and battery storage systems, into electrical distribution networks has led to more frequent voltage and congestion issues. Ensuring secure and efficient operation of distribution grids requires real-time control of inverter-interfaced DERs, achieved through dynamic optimization of their active and reactive power setpoints.

Online feedback optimization has recently emerged as a promising framework for this task. It iteratively drives the physical power grid toward an optimal operating point by embedding optimization algorithms directly within the feedback loop. This approach can function effectively under imprecise system models, limited or outdated data, and is robust to disturbances and measurement errors, as it relies on real-time feedback from the physical system rather than accurate forecasts or full observability.

`grid_feedback_optimizer` is an open-source Python package implementing the principles of online feedback optimization for power distribution networks. It links iterative optimization algorithms, such as projected gradient descent and primal–dual methods, with a nonlinear power flow solver, enabling closed-loop optimal control of grid-connected DERs. The package accepts network descriptions in JSON or Excel formats and computes the resulting optimal operating point. Its modular and extensible architecture facilitates customization for various research and engineering applications.

The projected gradient descent algorithm is formulated as a convex conic program, solved using the CLARABEL solver. This method typically requires less parameter tuning than the primal–dual algorithm, albeit with higher computational cost per iteration. The package can be applied to solving static optimal power flow problems, demonstrating and validating online feedback optimization algorithms, benchmarking real-time control strategies for distribution grids, supporting educational activities in power systems and optimization, managing microgrids and virtual power plants under distribution system operator coordination, and prototyping or evaluating local voltage and congestion management algorithms prior to SCADA/DMS deployment by DSOs.



# Statement of need

# Implementation

Primal dual: @DallAnese2018

Gradient projection: @Haberle2021

![Flowchart of the feedback optimization calculation process. \label{fig:flowchart}](Flowchart.pdf)

Compared to xxx.

Example result in ...


# Acknowledgements

This work is part of the NO-GIZMOS project (MOOI52109) which received funding from the Topsector Energie MOOI subsidy program of the Netherlands Ministry of Economic Affairs and Climate Policy, executed by the Netherlands Enterprise Agency (RVO).

# Conflicts of interest
The authors declare no conflicts of interest.


# References