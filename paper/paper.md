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
# Statement of need


The increasing integration of distributed energy resources (DERs), such as photovoltaics, electric vehicles, and battery storage systems, into electrical distribution networks has led to more frequent voltage and congestion issues. Ensuring secure and efficient operation of distribution grids requires real-time control of fast-responding inverter-interfaced DERs, achieved through dynamic optimization of their active and reactive power setpoints.

Online feedback optimization has recently emerged as a promising framework for this task [@DallAnese2018;@Haberle2021]. It iteratively drives the physical power grid toward an optimal operating point by embedding optimization algorithms directly within the feedback loop [@Hauswirth2021]. This approach can function effectively under imprecise system models, limited or outdated data, and is robust to disturbances and measurement errors [@robust], as it relies on real-time feedback from the physical system rather than accurate forecasts or full observability .

`grid_feedback_optimizer` is an open-source Python package implementing the principles of online feedback optimization for power distribution networks. It links iterative optimization algorithms, such as projected gradient descent [@Haberle2021] and primal–dual [@DallAnese2018], with a nonlinear power flow solver, enabling closed-loop optimal control of grid-connected DERs. The package accepts network descriptions in JSON or Excel formats and computes the resulting optimal operating point. Its modular and extensible architecture facilitates customization for various research and engineering applications.

The projected gradient descent algorithm is formulated as a convex conic program, solved by default using the `CLARABEL` solver [@Clarabel_2024]. This method typically requires less parameter tuning than the primal–dual algorithm, albeit with higher computational cost per iteration. To solve power flow programs efficiently, we use `power-grid-model` [@Xiang2023;@Xiang_PowerGridModel_power-grid-model]. `grid_feedback_optimizer` can be applied to solving static optimal power flow problems, demonstrating and validating online feedback optimization algorithms, benchmarking real-time control strategies for distribution grids, supporting educational activities in power systems and optimization, managing micro-grids and virtual power plants under distribution system operator coordination, and prototyping or evaluating voltage and congestion management algorithms prior to SCADA/DMS deployment by DSOs.

(\autoref{fig:flowchart}) shows the flowchart of the calculation process. The optimization algorithm engine takes power flow engine outputs, which are the network state, as feedback and generate updated setpoints for DERs. The updated setpoints are then fed into the power flow engine to extract new network state. Upon convergence, a steady state optimum is achieved, where distribution grid operational constraints are respected.

![Flowchart of the feedback optimization calculation process. \label{fig:flowchart}](Flowchart.pdf)

There are some existing packages on the market that solve optimal power flow problems. `Pandapower` [@pandapower.2018] uses `PYPOWER` [@matpower] as an optimization engine, supporting linearized DC and full AC optimal power flow problems. It uses the interior point method as the internal solver. `Pandapower` also integrates `PowerModels.jl`, which mainly uses `Ipopt [@WachterBiegler2006]` as the underlying solver but can be extended to other solvers.  `PyPSA` is another open-source optimal power flow solver, which allows solving a multi-period optimal power flow program. It comes by default with `HiGHS` [@huangfu2018parallelizing] and can call other solvers. In summary, all existing libraries require formulate a full AC OPF problem and call nonlinear programming solvers to solve them. Differently, `grid_feedback_optimizer` makes use of power flow calculations as input and uses simple first-order projected gradient descent or primal-dual algorithms. Particularly, when using the primal-dual algorithm, and when the DER includes only quadratic or box reactive power constraints, the projection problems can be solved analytically. The overall algorithm then does not require any optimization solver. By using power flow calculations as feedback, the algorithm also compensates for any modeling inaccuracy, leading to solutions that respect the operational constraints, which are not expected for convex relaxation algorithms.

The library comes with some examples results in the `notebooks` folder, including static results for a 97-bus low voltage system, and time-varying simulation results.



# Acknowledgements

This work is part of the NO-GIZMOS project (MOOI52109) which received funding from the Topsector Energie MOOI subsidy program of the Netherlands Ministry of Economic Affairs and Climate Policy, executed by the Netherlands Enterprise Agency (RVO).

# Conflicts of interest
The authors declare no conflicts of interest.


# References