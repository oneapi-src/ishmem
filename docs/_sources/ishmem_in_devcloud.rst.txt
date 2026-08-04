.. _ishmem_in_devcloud:

============================================
Using Intel® SHMEM in Intel® Tiber™ AI Cloud
============================================

Intel® SHMEM is now enabled in Intel® Tiber™ AI Cloud for oneAPI. 
To get access or sign in, please refer to the `Getting Started Guide: <https://devcloud.intel.com/oneapi/get_started/>`_.
When requesting hardware to launch an instance, please choose the
Intel® Max Series GPU from the hardware catalog list. This will ensure the
current supported GPUs for Intel® SHMEM are added to the requested instance.
Intel® SHMEM has been tested and validated on instances with Ubuntu 22.04 LTS
operating system (labeled as ubuntu-2204-jammy-v20230122).

After logging into the allocated instance, the Intel® oneAPI DPC++/C++ Compiler needs to be
added to the environment::

    source /opt/intel/oneapi/setvars.sh

By default, `cmake` is not installed in the instance which is required
for building Intel® SHMEM. It can be added with the following command::

    sudo apt install cmake

We also recommend to install a specific version of GNU `autoconf` that
works well with Sandia [#f1]_ OpenSHMEM installation. During testing, we occasionally found
issues with the default `autoconf` version of 2.71 . Users can follow the 
instructions below to download and install `autoconf` 2.69 in such a case::

    wget https://ftp.gnu.org/gnu/autoconf/autoconf-2.69.tar.gz
    tar -zxvf autoconf-2.69.tar.gz
    cd autoconf-2.69
    ./configure
    make
    sudo make install

With the above, users can directly follow the installation instructions
provided in `README.md <https://github.com/oneapi-src/ishmem/blob/main/README.md>`_
to install Intel® SHMEM with OpenSHMEM back-end.


.. [#f1] Other names and brands may be claimed as the property of others.
