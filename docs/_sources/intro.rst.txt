.. _introduction:

************
Introduction
************

.. This specification document defines the programming model and the application
.. programming interface (API) of Intel® SHMEM.
.. This :ref:`Introduction<introduction>` section first describes :ref:`"What is
.. Intel® SHMEM" <what_is_ishmem>`, explaining how it differs from
.. the original OpenSHMEM standard, which is described in the :ref:`"What is
.. OpenSHMEM" <what_is_openshmem>` section.

.. _what_is_ishmem:

================================
What is Intel® SHMEM?
================================

Intel® SHMEM is a C++ software library interface that enables OpenSHMEM
communication for applications using Intel® Data Center GPU Max Series devices
with device kernels implemented in SYCL.
Intel® SHMEM includes both host-initiated operations as in OpenSHMEM, and new
device-initiated operations callable directly from GPU kernels.
When available, Intel® SHMEM optimizes performance for GPUs connected by Intel®
:math:`\text{X}^e` Link fabric.
For multi-node communication, Intel® SHMEM enables GPU-initiated operations by
passing requests to the host CPU, which transfers the data through the network
fabric to another GPU's memory.

The original OpenSHMEM specification, described in the :ref:`"What is
OpenSHMEM?" section <what_is_openshmem>`, establishes a programming and memory
model that is most suitable for *host-driven* (CPU) communication.
The Intel® SHMEM APIs are based on the original OpenSHMEM
interfaces, but are augmented to support *device-driven* communication across
a network of compute accelerators, as exposed by the |sycl_spec_link|.
Supporting OpenSHMEM communication between SYCL host and device memories
requires subtle modifications to the original OpenSHMEM interfaces, memory
model, and execution model.
The Intel® SHMEM APIs are defined in the context of this adjusted
memory and execution model, which is described in sections :ref:`Memory
Model <memory_model>` and :ref:`Execution Model <execution_model>`.

.. |sycl_spec_link| raw:: html

   <a href="https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html" target="_blank">SYCL programming model</a>

.. _what_is_openshmem:

==================
What is OpenSHMEM?
==================

OpenSHMEM is a Partitioned Global Address Space (PGAS) library interface
specification.
OpenSHMEM aims to provide a standard API for SHMEM libraries to aid portability
and facilitate uniform predictable results of OpenSHMEM programs by explicitly
stating the behavior and semantics of the OpenSHMEM library calls.
Through the different versions, OpenSHMEM will continue to address the
requirements of the PGAS community.
As of this specification, many existing vendors support OpenSHMEM-compliant
implementations and new vendors are developing OpenSHMEM library
implementations to help the users write portable OpenSHMEM code.
This ensures that programs can run on multiple platforms without having to deal
with subtle vendor-specific implementation differences.

The OpenSHMEM [#]_ effort is driven by the OpenSHMEM specification committee
with continuous input from the OpenSHMEM user community.
To see all of the contributors and participants for the OpenSHMEM API, please
see: `http://www.openshmem.org/site/Contributors <http://www.openshmem.org/site/Contributors>`_.
In addition to the specification, the effort includes a reference OpenSHMEM
implementation, validation and verification suites, tools, a mailing list and
website infrastructure to support specification activities.  For more
information please refer to: `http://www.openshmem.org <http://www.openshmem.org/>`_.

.. [#] The OpenSHMEM specification is owned by Open Source Software Solutions Inc., a nonprofit organization, under an agreement with HPE [#f1]_.
.. [#f1] Other names and brands may be claimed as the property of others.


