# mldb
mldb is a wrapper around [lmdb](https://lmdb.readthedocs.io/) designed with machine learning or other number crunching in mind.   lmdb databases are memory mapped which allows very fast reading of large amounts of data with a minimal memory footprint. mldb takes advantage of this by leaving the data of numpy.ndarray's in the memory map rather than copying it.  This means you can process numeric data many times larger than the amount of RAM available with very little overhead.  lmdb's locking mechanism means it is safe to read and write to a DataSet from multiple processes.

The interface quacks like a list of dicts and there is no need to explicitly save so you don't need know about the database to use it.

## EXAMPLES
```python
dataset = mldb.DataSet('my_dataset')
for i in range(100000):
    dataset.append({'input': numpy.random.rand(1000)})
for point in dataset:
    point['output'] = point['input'] ** 2
```

You can optionally add dtype and shape constraints to keep your data homogenous
```python
input_constraint = mldb.FieldConstraint(dtype='float', shape=(1000,))
dataset.add_constraint('input', input_constraint)
dataset.auto_constraint('output', shape_mask=(True,)) # gives same constraint automatically
```

DataSet also has an .annotations member which quacks like a dict:
```python
algorithm = MyAlgorithmClass()
dataset.annotations['algorithm'] = algorithm # pickles the algorithm
dataset.annotations['parameter'] = 7
````
