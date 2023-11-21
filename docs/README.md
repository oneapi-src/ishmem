## Intel® SHMEM Documentation

The Intel® SHMEM documentation uses reStructuredText (rST) lightweight markup
with the Sphinx documentation generator.

To add a new topic, create a new .rst file in the source folder and add it to
the TOC (index.rst).

To get started with Sphinx, Read the Docs, and rST syntax please refer to:
    https://docs.readthedocs.io/en/latest/index.html
    http://docutils.sourceforge.net/docs/user/rst/quickref.html

### Requirements to generate the documentation:

1.  Python 3.x and the `pip` package manager.
2.  Only Linux operating systems are currently supported.
3.  Install Sphnix to build the documentation and the Read the Docs theme:
```
    pip install sphinx
    pip install sphinx_rtd_theme
```
4.  Generate documentation using the steps below.

### To generate and view the documentation locally:

1.  Go to the `docs/source` directory.
2.  Optionally run `make clean` to remove any pre-existing files.
3.  Run `make html`.
4.  Open the `build/html/index.html` file with your favorite browser.
