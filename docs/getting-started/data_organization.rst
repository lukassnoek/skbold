Data organization
-----------------

Skbold is quite flexible in how to work with data organized in different
ways. However, to optimally use the `Mvp` classes (especially the `MvpBetween`
class), it's best to organize your data hierarchically. For example,
suppose you have a `.feat`-directory with first-level pattern estimates of
two two runs ('run-1', 'run-2') for 3 subjects ('sub-01', 'sub-02', 'sub-03').
A nice organization would be as follows:

.. code-block:: text

  project_dir
   ├── sub-01
   │   ├── run-1
   │   │   └── some_task.feat
   │   └── run-2
   │       └── some_task.feat
   ├── sub-02
   │   ├── run-1
   │   │   └── some_task.feat
   │   └── run-2
   │       └── some_task.feat
   └── sub-03
       ├── run-1
       │   └── some_task.feat
       └── run-2
           └── some_task.feat

This organization makes it easy for `Mvp` objects to find, load in, and
organize patterns from these `.feat`-directories, as will become more
clear in the examples-section on `Mvp` objects.
