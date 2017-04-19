# Copyright (C) 2017 Charlie Scherer <charlie@tildesrc.com>. All Rights Reserved.
# This file is licensed under GPLv3.  Please see COPYING for more information.
"""mldb
Support routines for managing machine learning datasets that minimize memory
footprint even for huge datasets, facilitates concurrency and robust
persistence.

The main class here is DataSet which stores DataPoints.  Each of these classes
stores data in an lmdb database providing the following benefits:

  *  numpy.ndarrays stored in a DataPoint will be memory mapped
     rather than being copied into memory, keeping the memory footprint much
     smaller than the size of the dataset.  Other data is stored pickled in the
     database and unpickled on request.

  *  All transactions are protected by locks in the database
     meaning it's safe to access the dataset from multiple processes,
     facilitating parallel processing.

  *  Changes are written to the memory map after each operation,
     and will persist immediately.

DataSet and DataPoint implement MutableSequence and MutableMapping,
respectively, meaning they act mostly like lists and dicts.
"""

import persistent
import util
from data import DataPoint, DataSet, FieldConstraint, load_rectangular_dump, PARTITIONS
from persistent import finalize
from util import DelayedInterrupt

