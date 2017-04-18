import copy
import cStringIO
import pickle
import collections
import warnings

import numpy
import lmdb
import fallocate

import persistent
from persistent import ManageTransaction, ValueSerializer

PARTITIONS = ('training', 'testing', 'cross_validation')

NumPyArrayRef = collections.namedtuple(
    'NumPyArrayRef',
    ('shape', 'dtype', 'strides', 'order')
)

class DataPoint(persistent.Dict): # pylint: disable=too-many-ancestors,too-many-public-methods
    """
    A dict-like persistent value that memory maps ndarrays.
    During __init__ a partition is chosen from PARTITIONS
    with the following probabilities:
    0.8 -> training
    0.1 -> testing
    0.1 -> cross_validation
    """
    def __init__(self, dataset, prefix):
        self.dataset = dataset
        def pad_length(pos, itemsize):
            return (itemsize - pos) % itemsize
        def serializer(value):
            """
            Serializes by:
              * ndarrays: pickling a NumPyArrayRef and dumping the array's data
              * non-ndarrays: pickling
            """
            if isinstance(value, numpy.ndarray):
                ref = NumPyArrayRef(
                    shape=value.shape,
                    dtype=value.dtype,
                    strides=value.strides,
                    order='C' if value.flags.carray else 'F'
                )
                header = pickle.dumps(ref)
                header += b"\0" * pad_length(len(header), ref.dtype.itemsize)
                return header + value.tobytes()
            else:
                return pickle.dumps(value)
        def deserializer(value):
            """
            Deserializes by unpickling, if the result is a NumPyArray a ndarray
            is creating whose data is the remainder of the buffer.
            """
            fub = cStringIO.StringIO(value)
            item = pickle.load(fub)
            if isinstance(item, NumPyArrayRef):
                fub.seek(pad_length(fub.tell(), item.dtype.itemsize), 1)
                item = numpy.ndarray(
                    item.shape,
                    dtype=item.dtype,
                    buffer=buffer(value, fub.tell()),
                    strides=item.strides,
                    order=item.order
                )
            return item
        persistent.Dict.__init__(
            self,
            self.dataset.manager,
            prefix,
            value_serializer=ValueSerializer(
                serializer=serializer,
                deserializer=deserializer,
                finalizer=None
            ),
            key_serializer=ValueSerializer(
                serializer=lambda x: x,
                deserializer=lambda x: x,
                finalizer=None
            )
        )
        if 'partition' not in self:
            random_sample = numpy.random.rand()
            if random_sample < 0.8:
                self['partition'] = 'training'
            elif random_sample < 0.9:
                self['partition'] = 'testing'
            else:
                self['partition'] = 'cross_validation'

    def __setitem__(self, key, value):
        if key in self.dataset.constraints:
            self.dataset.constraints[key].check(value)
        persistent.Dict.__setitem__(self, key, value)

    def __eq__(self, other):
        if not isinstance(other, DataPoint):
            return False
        else:
            return other.dataset == self.dataset and other.prefix == self.prefix

class DataSet(persistent.List): # pylint: disable=too-many-ancestors,too-many-public-methods
    """
    A persistent list-like value that stores DataPoints.
    In addition to list-like methods, DataSet also has
      * .add_constraint and .auto_constrain for contraining ndarray fields
      * a persistent dict-like .annotations for storing DataSet-wide data
      * .dump_rectangular_field to facilitate cooperation with other languages.
    **kwargs are passed to TransactionManager, see the doc-string for that class,
    especially regarding the quick_write keyword.
    """
    def __init__(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs
        self.manager = persistent.TransactionManager(path, **kwargs)
        def serializer(value): # pylint: disable=missing-docstring
            prefix = self.next_prefix()
            datapoint = DataPoint(self, prefix)
            datapoint.update(value)
            return prefix
        def deserializer(prefix): # pylint: disable=missing-docstring
            return DataPoint(self, prefix)
        def finalizer(value): # pylint: disable=missing-docstring
            deserializer(value).clear()
        main_prefix = bytearray([1])
        persistent.List.__init__(
            self,
            self.manager,
            main_prefix,
            value_serializer=ValueSerializer(
                serializer=serializer,
                deserializer=deserializer,
                finalizer=finalizer
            )
        )

        constraint_prefix = bytearray([2])
        self.constraints = persistent.Dict(
            self.manager,
            constraint_prefix,
            value_serializer=ValueSerializer(
                serializer=pickle.dumps,
                deserializer=pickle.loads,
                finalizer=None
            ),
            key_serializer=ValueSerializer(
                serializer=lambda x: x,
                deserializer=lambda x: x,
                finalizer=None
            )
        )

        annotation_prefix = bytearray([3])
        self.annotations = persistent.Dict(
            self.manager,
            annotation_prefix,
            value_serializer=ValueSerializer(
                serializer=pickle.dumps,
                deserializer=pickle.loads,
                finalizer=None
            ),
            key_serializer=ValueSerializer(
                serializer=lambda x: x,
                deserializer=lambda x: x,
                finalizer=None
            )
        )

        self.next_prefix = persistent.PrefixManager(self.manager)
        self.next_prefix.set_range(annotation_prefix)

    def __getstate__(self):
        return {
            "path": self.path,
            "kwargs": self.kwargs
        }

    def __setstate__(self, state):
        self.__init__(**state)

    def close(self):
        self.manager.close()

    def clear_field(self, field):
        for point in self:
            if field in point:
                del point[field]

    def partition_iterator(self, partition):
        for point in self:
            if point["partition"] == partition:
                yield point

    def rename_field(self, old_name, new_name, overwrite=False):
        def rename(obj):
            if old_name in obj and (overwrite or new_name not in obj):
                obj[new_name] = obj[old_name]
                del obj[old_name]
        rename(self.constraints)
        rename(self.annotations)
        for point in self:
            rename(point)

    @ManageTransaction(buffers=True, write=True)
    def add_constraint(self, field, constraint):
        """
        Add a constraint to current and future values of an ndarray field.
        """
        self.constraints[field] = copy.deepcopy(constraint)
        self.check_constraint(field)

    @ManageTransaction(buffers=True, write=True)
    def auto_constrain(self, field, shape_mask=None):
        """
        Attempt to automatically guess a constraint for a given field and add
        it.  This is done by using the first example of this field as a
        template.  If a shape_mask is given, values where it is True indicate
        which dimensions of the example's shape to use in the constraint.
        """
        for point in self:
            if field in point:
                value = point[field]
                if shape_mask == None:
                    shape = None
                else:
                    if len(shape_mask) != len(value.shape):
                        raise ConstraintError(
                            "Shape mask doesn't match ndim of field value"
                        )
                    shape = list(value.shape)
                    for i in range(len(shape_mask)):
                        if not shape_mask[i]:
                            shape[i] = None
                self.add_constraint(field, FieldConstraint(value.dtype, shape))
                return
        raise KeyError("%s not found in any data point" % (field,))

    def check_constraint(self, field): # pylint: disable=missing-docstring
        for point in self:
            if field in point:
                if not self.constraints[field].check(point[field]):
                    raise ConstraintError(
                        "Constraint failed on field %s" % (field,)
                    )

    def field_is_rectangular(self, field):
        """
        Rectangular means the field is constrained to 1 or 2 dimensions and has a
        constrained last dimension.
        """
        if field not in self.constraints:
            return False

        constraint = self.constraints[field]

        if constraint.ndim not in (1, 2):
            return False

        if constraint.shape[-1] == None:
            return False

        return True

    def dump_rectangular_field(
            self,
            field,
            target,
            allowed_partitions=PARTITIONS,
            order='C'):
        """
        Dump a rectangular field to the file-like target and return a memmap
        together with kwargs for memmap'ing the target in the future.  This
        results in a memmap of dimension 2 with last dimension equal to the the
        last dimension of the field.
        """
        if not self.field_is_rectangular(field):
            raise ConstraintError("%s is not rectangular" % (field,))

        shape = [0, 0]
        for point in self:
            if point["partition"] in allowed_partitions and field in point:
                value = point[field]
                value = value.reshape(-1, value.shape[-1])
                shape[0] += value.shape[0]
        shape[1] = value.shape[-1]
        shape = tuple(shape)
        dtype = value.dtype
        start = target.tell()

        fallocate.posix_fallocate(
            target.fileno(),
            start,
            shape[0] * shape[1] * dtype.itemsize
        )

        mdata_kwargs = {
            "offset": start,
            "dtype": dtype,
            "shape": shape,
            "order": order
        }
        mdata = numpy.memmap(target, mode="r+", **mdata_kwargs)

        offset = 0
        for point in self:
            if point["partition"] in allowed_partitions and field in point:
                value = point[field]
                value = value.reshape(-1, value.shape[-1])
                new_offset = offset + value.shape[0]
                mdata[offset:new_offset, :] = value
                offset = new_offset

        target.flush()

        return mdata, mdata_kwargs

class FieldConstraint(object): # pylint: disable=too-few-public-methods,
    """
    A class to constrain ndarray field values.  A constraint includes at least
    a dtype which all field values must share.  Additionally, a shape
    constraint can be provided.  The length of the shape constraint must match
    the dimension of the field values.  Additionally, each non-None entry in
    the shape constraints must match exactly the entry in the field value's
    shape.  For example, (None, 10) constrains to dimension 2 with second
    dimension 10.
    """
    def __init__(self, dtype, shape=None):
        self.dtype = numpy.dtype(dtype)
        self.shape = shape

    def check(self, value): # pylint: disable=missing-docstring
        if not issubclass(type(value), numpy.ndarray):
            return False
        if value.dtype != self.dtype:
            return False
        if self.shape == None:
            return True
        else:
            if value.ndim != len(self.shape):
                return False
            for v_dim, s_dim in zip(value.shape, self.shape):
                if s_dim != None and v_dim != s_dim:
                    return False
            return True

    @property
    def ndim(self): # pylint: disable=missing-docstring
        if self.shape is None:
            return None
        else:
            return len(self.shape)

class ConstraintError(ValueError): # pylint: disable=missing-docstring
    pass

def load_rectangular_dump(dump_spec, **kwargs):
    """
    Load a dumped field.  dump_spec = (name, kwargs) where name is the
    filename of the dump and kwargs are the kwargs returned by
    dump_rectangular_field.  This argument is structured this way to facilitate
    storing it in the dataset's annotations.
    """
    dump_spec = copy.deepcopy(dump_spec)
    dump_spec[1].update(kwargs)
    return numpy.memmap(dump_spec[0], **dump_spec[1])

