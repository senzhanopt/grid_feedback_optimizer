---
title: 'grid-feedback-optimizer: A Python package for feedback-based optimization of power grids'
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


The increasing integration of distributed energy resources (DERs)--such as photovoltaics, electric vehicles, and battery storage systems--into electrical distribution networks has led to more frequent voltage and congestion issues. Ensuring secure and efficient operation of modern distribution grids requires real-time control of fast-responding inverter-interfaced DERs, achieved through dynamic optimization of their active and reactive power setpoints.

Online feedback optimization has recently emerged as a promising framework for this purpose [@DallAnese2018;@Haberle2021]. It iteratively drives the physical system toward an optimal operating point by embedding optimization algorithms directly within the feedback loop [@Hauswirth2021]. This approach performs reliably even under imprecise system models, measurement errors, outdated data, and disturbances [@robust], as it relies on real-time measurements rather than perfect forecasts or full observability.

`grid_feedback_optimizer` is an open-source Python package implementing the principles of online feedback optimization for power distribution networks. It couples iterative optimization algorithms--such as projected gradient descent [@Haberle2021] and primal–dual methods [@DallAnese2018]--with a nonlinear power flow solver, enabling closed-loop optimal control of grid-connected DERs. The package supports network data in `JSON` or `Excel` formats and features a modular, extensible architecture suitable for both research and practical applications.

# Statement of need

Several open-source packages are available for solving optimal power flow (OPF) problems. `Pandapower` [@pandapower.2018] employes `PYPOWER` [@matpower] as its optimization engine, supporting both linearized DC and full AC OPF problems through an interior-point solver. It also integrates `PowerModels.jl`, which mainly uses `Ipopt [@WachterBiegler2006]` as the internal solver but can be extended to other solvers.  `PyPSA` is another open-source OPF framework that allows multi-period formulations and uses `HiGHS` [@huangfu2018parallelizing] as its default solver, with the option to interface with additional solvers.

In summary, all existing libraries require formulating a full AC OPF problem and depend on nonlinear programming solvers to compute optimal operating points. In contrast, `grid_feedback_optimizer` adopts a fundamentally different approach. It leverages power flow calculations as feedback and employs simple first-order algorithms--projected gradient descent or primal–dual method--to iteratively update DER setpoints. Notably, when using the primal–dual algorithm and when DERs are subject only to quadratic or box-type reactive power constraints, the projection steps can be easily solved analytically. As a result, the overall optimization process can operate efficiently entirely without external solvers. By integrating power flow calculations directly into the feedback loop, `grid_feedback_optimizer` inherently compensates for modeling inaccuracies and ensures that grid operational constraints are satisfied. This feedback-based approach provides robustness not typically achievable with convex relaxation methods.

The package can be applied to a wide range of use cases, including solving static OPF problems, demonstrating and validating online feedback optimization algorithms, benchmarking real-time control strategies for distribution grids, supporting educational activities in power systems and optimization, managing microgrids and virtual power plants under distribution system operator (DSO) coordination, and prototyping or evaluating voltage and congestion management algorithms prior to deployment by DSOs.


# Implementation

\autoref{fig:flowchart} illustrates the feedback optimization calculation process. The optimization algorithm engine receives the network state computed by the power flow engine as feedback and uses it to generate updated setpoints for DERs. These new setpoints are then fed back into the power flow engine to compute the corresponding network state. This iterative process continues until convergence to a steady-state optimum, where all operational constraints of the distribution grid are satisfied.

![Flowchart of the feedback optimization calculation process. \label{fig:flowchart}](Flowchart.pdf)

The projected gradient descent algorithm is formulated as a convex conic program and is, by default, solved using the `CLARABEL` solver [@Clarabel_2024]. The primal-dual method dualizes network constraints and enforces individual DER constraints through projection, which is also solved, by default, by `CLARABEL` when analytical solutions are not available. Compared to projected gradient descent, the primal–dual algorithm typically requires more parameter tuning--particularly the selection of multiple step sizes--but offers a lower computational cost per iteration. Power flow programs are efficiently solved using `power-grid-model` [@Xiang2023;@pgm]. 


The library also includes several example notebooks demonstrating its capabilities, featuring both static results for a 97-bus low-voltage network and dynamic, time-varying simulations.



# Acknowledgements

This work is part of the NO-GIZMOS project (MOOI52109) which received funding from the Topsector Energie MOOI subsidy program of the Netherlands Ministry of Economic Affairs and Climate Policy, executed by the Netherlands Enterprise Agency (RVO).

# Conflicts of interest
The authors declare no conflicts of interest.


# References