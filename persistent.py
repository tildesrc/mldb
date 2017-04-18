import abc
import collections
import pickle
import ctypes
import os
import cStringIO

import lmdb

default_map_size = 1024**4

class ManageTransaction(object): # pylint: disable=too-few-public-methods
    """
    Decorator that begins a transaction before a function and commits it after.
    Signature is alias of lmdb.Environment.begin
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    def __call__(self, func):
        def wrapper(obj, *args, **kwargs): # pylint: disable=missing-docstring
            with obj.transaction(*self.args, **self.kwargs):
                return func(obj, *args, **kwargs)
        return wrapper

manager_states = {}

def unique_pid():
    pid = os.getpid()
# See man proc
    return (pid, int(open("/proc/%d/stat" % (pid,)).read().split(" ")[21]))

def finalize():
    for (pid, path), state in manager_states.items():
        if pid == unique_pid():
            state["env"].close()

class TransactionManager(object):
    """
    Manages transactions to lmdb allowing functions which interact with the
    database to call one another and share a transaction.  Functions that begin
    the transaction in read mode can not call functions that need a writeable
    transaction.
    quick_write causes {'sync': False, 'map_async': True, 'writemap': True}
    to be added to kwargs (defaults to True).  This will speed up serial
    writes considerably but the database is not guaranteed to remain
    consistent after a system crash.
    """
    def __init__(self, path, quick_write=True, **kwargs):
        self.path = os.path.realpath(path)
        self.pid = unique_pid()
        self.kwargs = {'map_size': default_map_size}
        if quick_write:
            self.kwargs['sync'] = False
            self.kwargs['map_async'] = True
            self.kwargs['writemap'] = True
        self.kwargs.update(kwargs)
        global manager_states
        if (self.pid, self.path) not in manager_states:
            manager_states[(self.pid, self.path)] = {
                "env": lmdb.Environment(self.path, **self.kwargs),
                "ref_count": 0,
                "txn": None
            }
        self.state = manager_states[(self.pid, self.path)]

    def __getattr__(self, name):
        if name == "state":
            return {}
        if name in self.state:
            return self.state[name]
 
    def __setattr__(self, name, value):
        if name in self.state:
            self.state[name] = value
        else:
            self.__dict__[name] = value

    def begin(self, *args, **kwargs):
        """
        Alias for lmdb.Environment.begin that updates an internal reference
        count.
        """
        if self.pid != unique_pid(): # A fork occured
            self.__init__(self.path)
 
        if self.txn == None:
            self.txn = self.env.begin(*args, **kwargs)
            assert self.ref_count == 0

        self.ref_count += 1
        return self.txn

    def commit(self):
        """
        Alias for lmdb.Environment.commit that updates an internal
        reference count.
        """
        self.ref_count -= 1
        assert self.ref_count >= 0
        if self.ref_count == 0:
            self.txn.commit()
            self.txn = None

    def close(self):
        self.env.close()

    @property
    def active():
        return self.ref_count == 0

class TransactionContext(object): # pylint: disable=too-few-public-methods
    """
    Context manager that manages a transaction to lmdb.
    """
    def __init__(self, obj, *args, **kwargs):
        """
        Alias for lmdb.Environment.begin
        """
        self.manager = obj.manager
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        return self.manager.begin(*self.args, **self.kwargs)

    def __exit__(self, exception_type, value, traceback):
        self.manager.commit()


class ValueSerializer(object): # pylint: disable=too-few-public-methods
    """
    Serialization details for a value:
     * serializer transforms an object before it is written to the database
     * deserializer transformed an objected read from the database
     * finalizer is called on an object that is about to be deleted from the
       database, except in cases like pop or popitem where the item will
       continue to be referenced by another variable.
    Defaults are:
      serializer=pickle.dumps
      deserializer=pickle.dumps
      finalizer=lambda x: None (a no-op)
    """
    def __init__(self, serializer=None, deserializer=None, finalizer=None):
        self.serializer = serializer
        self.deserializer = deserializer
        self.finalizer = finalizer
        if self.serializer == None:
            self.serializer = pickle.dumps
        if self.deserializer == None:
            self.deserializer = pickle.loads
        if self.finalizer == None:
            self.finalizer = lambda x: None

class CTypeSerializer(ValueSerializer): # pylint: disable=too-few-public-methods
    """A ValueSerializer automatically generated for a given ctype."""
    def __init__(self, ctype):
        self.ctype = ctype
        def serializer(value): # pylint: disable=missing-docstring
            return buffer(ctype(value))
        def deserializer(value): # pylint: disable=missing-docstring
            return ctype.from_buffer_copy(value).value
        def finalizer(_): # pylint: disable=missing-docstring
            pass
        ValueSerializer.__init__(
            self,
            serializer=serializer,
            deserializer=deserializer,
            finalizer=finalizer
        )

class Value(object):
    """Abstract class for a value that is stored in the database."""
    __metaclass__ = abc.ABCMeta
    def __init__(
            self,
            manager,
            prefix,
            value_serializer=None):
        self.manager = manager
        self.prefix = bytearray(prefix)
        #print "%02x"*len(self.prefix) % tuple(self.prefix)
        self.value_serializer = value_serializer
        if self.value_serializer == None:
            self.value_serializer = ValueSerializer()

    def key_prefix(self, key):
        """returns initial segment of a key the of length as self's prefix"""
        return key[:len(self.prefix)]

    def key_suffix(self, key):
        """returns the rest of a key after the key_prefix"""
        return key[len(self.prefix):]

    def finalize_value(self, value):
        """Finalizes the raw (non-deserialized) value from the database."""
        self.value_serializer.finalizer(value)

    def serialize_value(self, value): # pylint: disable=missing-docstring
        return self.value_serializer.serializer(value)

    def deserialize_value(self, value): # pylint: disable=missing-docstring
        return self.value_serializer.deserializer(value)

    def transaction(self, *args, **kwargs):
        """
        Transaction context that uses self's TransactionManager.  Signature is
        alias of lmdb.Environment.begin.
        """
        return TransactionContext(self, *args, **kwargs)

    @abc.abstractmethod
    def clear(self):
        """Completely removes this value from the database."""
        pass

class Collection(Value): # pylint: disable=abstract-method
    """ Abstract base class for database backed collections."""
    __metaclass__ = abc.ABCMeta
    def __init__(
            self,
            manager,
            prefix,
            value_serializer=None,
            key_serializer=None):
        self.key_serializer = key_serializer
        if self.key_serializer == None:
            self.key_serializer = ValueSerializer()

        Value.__init__(
            self,
            manager,
            prefix,
            value_serializer
        )

    def serialize_key(self, key): # pylint: disable=missing-docstring
        return self.prefix + self.key_serializer.serializer(key)

    def deserialize_key(self, key): # pylint: disable=missing-docstring
        return self.key_serializer.deserializer(self.key_suffix(key))

    def finalize_key(self, key): # pylint: disable=missing-docstring
        self.key_serializer.finalizer(key)

class Scalar(Value):
    """
    A generic database backed value.  Here "scalar" means that although the
    complete value of the object is stored in the database, none of the
    structure is integrated into the database.  Constrast this with
    persistent.List whose members have their own database entries and
    therefore some of its structure is integrated into the databases's
    structure.
    """
    @ManageTransaction(buffers=True, write=False)
    def get(self): # pylint: disable=missing-docstring
        return self.deserialize_value(self.manager.txn.get(self.prefix))

    @ManageTransaction(buffers=True, write=True)
    def set(self, value): # pylint: disable=missing-docstring
        old_value = self.manager.txn.replace(
            self.prefix,
            self.serialize_value(value)
        )
        if old_value is not None:
            self.finalize_value(old_value)

    @property
    @ManageTransaction(buffers=True, write=False)
    def is_unset(self): # pylint: disable=missing-docstring
        return self.manager.txn.get(self.prefix) == None

    @ManageTransaction(buffers=True, write=True)
    def clear(self): # pylint: disable=missing-docstring
        self.manager.txn.delete(self.prefix)

class CType(Scalar):
    """
    A persistent.Scalar that stores a ctype.  ValueSerializer is generated
    automatically.
    """
    def __init__(self, manager, prefix, ctype):
        self.ctype = ctype
        Scalar.__init__(
            self,
            manager,
            prefix,
            CTypeSerializer(ctype)
        )

def deleting_iterator(indices):
    """
    Iterate over indices but adjust them assuming that the already
    iterated indices have been removed from the list.  E.g.
    [1, 2, 4] -> [1, 1, 2]
    Since removing 1 moves the item at index 2 to index 1, etc.
    """
    j = 0
    last_i = None
    for i in indices:
        if last_i is not None and last_i < i:
            j += 1
        yield i-j
        last_i = i

class List(Collection, collections.MutableSequence): # pylint: disable=too-many-ancestors
    """
    A database backed MutableSequence (list-like) class.
    """
    def __init__(
            self,
            manager,
            prefix,
            value_serializer=None):
        self.length = CType(manager, prefix, ctypes.c_ulong)
        Collection.__init__(
            self,
            manager,
            prefix,
            value_serializer,
            key_serializer=CTypeSerializer(ctypes.c_ulong)
        )

    @ManageTransaction(buffers=True, write=False)
    def __len__(self):
        if self.length.is_unset:
            return 0
        else:
            return self.length.get()

    @ManageTransaction(buffers=True, write=True)
    def __increment_length(self): # pylint: disable=missing-docstring
        if self.length.is_unset:
            self.length.set(1)
        else:
            self.length.set(self.length.get()+1)

    @ManageTransaction(buffers=True, write=True)
    def __decrement_length(self): # pylint: disable=missing-docstring
        assert not self.length.is_unset
        self.length.set(self.length.get()-1)

    def __increment_key(self, key): # pylint: disable=missing-docstring
        return self.serialize_key(self.deserialize_key(key)+1)

    def __get_cursor(self, index, txn=None):
        """
        Return a cursor pointing to the value at the given index or None if
        that index is out of bounds.
        """
        if txn == None:
            txn = self.manager.txn
        cursor = txn.cursor()
        if index >= 0:
            cursor_valid = cursor.set_key(self.serialize_key(index))
        if index < 0:
            length = len(self)
            if length > 0:
                index = length + index
                cursor_valid = index >= 0 and \
                    cursor.set_key(self.serialize_key(index))
            else:
                cursor_valid = False
        if cursor_valid:
            return cursor
        else:
            return None

    @ManageTransaction(buffers=True, write=False)
    def __getitem__(self, arg):
        if type(arg) == slice:
            return [self.__get_item(i) for i in range(len(self))[arg]]
        else:
            return self.__get_item(arg)

    @ManageTransaction(buffers=True, write=False)
    def __get_item(self, index): # pylint: disable=missing-docstring
        cursor = self.__get_cursor(index)
        if cursor == None:
            raise IndexError("persistent.List index out of range")
        return self.deserialize_value(cursor.value())

    @ManageTransaction(buffers=True, write=True)
    def __setitem__(self, arg, value):
        if type(arg) == slice:
            self.__set_slice(arg, value)
        else:
            self.__set_item(arg, value)

    def __indices_from_slice(self, sli): # pylint: disable=missing-docstring
        return range(len(self))[sli]

    def __set_slice(self, slice_to_set, value): # pylint: disable=missing-docstring
        if slice_to_set.step is not None:
            self.__set_extended_slice(slice_to_set, value)
        else:
            self.__set_standard_slice(slice_to_set, value)

    def __set_extended_slice(self, extended_slice, value):
        """Set an extended slice, i.e., one with a step value."""
        indices = self.__indices_from_slice(extended_slice)
        if len(indices) != len(value):
            raise ValueError(
                "attempt to assign sequence of size %d "
                "to extended slice of size %d" %
                (len(indices), len(value))
            )
        else:
            for i, val in zip(indices, value):
                self.__set_item(i, val)

    def __set_standard_slice(self, standard_slice, value):
        """
        Set a standard slice, i.e., one without a step value.
        Implementation: first delete the slice then insert new elements at its
        beginning.
        """
        indices = self.__indices_from_slice(standard_slice)
        for i in deleting_iterator(indices):
            del self[i]
        try:
            i = indices[0]
        except IndexError:
            i = 0
        for val in value:
            self.insert(i, val)
            i += 1

    @ManageTransaction(buffers=True, write=True)
    def __set_item(self, index, value): # pylint: disable=missing-docstring
        cursor = self.__get_cursor(index)
        if cursor == None:
            raise IndexError("persistent.List assignment index out of range")
        self.finalize_value(cursor.value())
        cursor.replace(cursor.key()[:], self.serialize_value(value))

    @ManageTransaction(buffers=True, write=True)
    def __delitem__(self, arg):
        if type(arg) == slice:
            for i in deleting_iterator(range(len(self))[arg]):
                self.__delete_item(i)
        else:
            self.__delete_item(arg)

    @ManageTransaction(buffers=True, write=True)
    def __delete_item(self, index, finalize=True): # pylint: disable=missing-docstring
        """
        Delete an item.
        Implementation: copy elements from the back of the list to lower
        indices and then delete the last element (which is now a copy of the
        second-to-last).
        """
        cursor = self.__get_cursor(index)
        if cursor == None:
            raise IndexError("persistent.List assignment index out of range")

        if finalize:
            self.finalize_value(cursor.value())

        back_cursor = self.__get_cursor(-1)
        value = back_cursor.value()[:]
        while back_cursor.key() != cursor.key():
            back_cursor.prev()
            key, next_value = copy_buffers(back_cursor.item())
            back_cursor.put(key, value)
            value = next_value
        back_cursor = self.__get_cursor(-1)
        back_cursor.delete()

        self.__decrement_length()

    @ManageTransaction(buffers=True, write=True)
    def pop(self, *args): # pylint: disable=missing-docstring
        if len(args) > 1:
            raise TypeError(
                "pop takes at most 1 argument, got %d" %
                (len(args),)
            )

        if len(self) == 0:
            raise IndexError("pop from empty persistent.List")

        if len(args) == 1:
            index = args[0]
        else:
            index = -1

        cursor = self.__get_cursor(index)
        if cursor == None:
            raise IndexError("pop index out of range")
        value = self.deserialize_value(cursor.value())

        self.__delete_item(index, finalize=False)

        return value

    @ManageTransaction(buffers=True, write=True)
    def insert(self, index, value): # pylint: disable=missing-docstring
        cursor = self.__get_cursor(index)
        append_mode = False
        if cursor == None:
            if index >= 0:
                cursor = self.__get_cursor(-1)
                append_mode = True
            if index < 0:
                cursor = self.__get_cursor(0)

        value = self.serialize_value(value)
        if cursor is None:
            self.manager.txn.put(self.serialize_key(0), value)
        else:
            if not append_mode:
                while True:
                    key, next_value = copy_buffers(cursor.item())
                    cursor.put(key, value)
                    value = next_value
                    if not cursor.next():
                        break
                cursor.put(self.__increment_key(key), value)
            else:
                self.manager.txn.put(self.__increment_key(cursor.key()), value)

        self.__increment_length()

    @ManageTransaction(buffers=True, write=True)
    def clear(self): # pylint: disable=missing-docstring
        cursor = self.manager.txn.cursor()
        if not self.length.is_unset:
            self.length.clear()
        cursor.set_range(self.prefix)
        while self.key_prefix(cursor.key()) == self.prefix:
            self.finalize_value(cursor.value())
            cursor.delete()

class Dict(Collection, collections.MutableMapping): # pylint: disable=too-many-ancestors,too-many-public-methods
    """Database backed MutableMapping (dict-like) class."""
    def __get_cursor(self, key): # pylint: disable=missing-docstring
        """
        Get a cursor pointing to the given key if possible and return the
        cursor with a success boolean.
        """
        cursor = self.manager.txn.cursor()
        return cursor, cursor.set_key(self.serialize_key(key))

    @ManageTransaction(buffers=True, write=False)
    def __getitem__(self, key):
        value = self.manager.txn.get(self.serialize_key(key))
        if value == None:
            raise KeyError(key)
        return self.deserialize_value(value)

    @ManageTransaction(buffers=True, write=True)
    def __setitem__(self, key, value):
        old_value = self.manager.txn.replace(
            self.serialize_key(key),
            self.serialize_value(value)
        )
        if old_value is not None:
            self.finalize_value(old_value)

    @ManageTransaction(buffers=True, write=True)
    def __delitem__(self, key):
        cursor, key_found = self.__get_cursor(key)
        if not key_found:
            raise KeyError(key)
        self.finalize_key(cursor.key())
        self.finalize_value(cursor.value())
        cursor.delete()

    @ManageTransaction(buffers=True, write=True)
    def pop(self, key, *args):
        if len(args) > 1:
            raise TypeError(
                "pop expected at most 2 arguments, got %d" %
                (len(args),)
            )
        cursor, key_found = self.__get_cursor(key)
        if not key_found:
            if len(args) == 1:
                return args[0]
            else:
                raise KeyError(key)
        self.finalize_key(cursor.key())
        value = self.deserialize_value(cursor.value())
        cursor.delete()
        return value

    @ManageTransaction(buffers=True, write=True)
    def popitem(self):
        cursor = self.manager.txn.cursor()
        cursor.set_range(self.prefix)
        if self.key_prefix(cursor.key()) != self.prefix:
            raise KeyError("popitem(): persistent.Dict is empty")
        key = self.deserialize_key(cursor.key())
        value = self.deserialize_value(cursor.value())
        cursor.delete()
        return key, value

    def __get_cursor_at_start(self, txn=None): # pylint: disable=missing-docstring
        cursor = self.manager.txn.cursor()
        return cursor, cursor.set_range(self.prefix)

    def __iter__(self):
        with self.transaction(buffers=True, write=True) as txn:
            cursor, cursor_valid = self.__get_cursor_at_start(txn)
            if not cursor_valid:
                return
            key = cursor.key()

        while self.key_prefix(key) == self.prefix:
            with self.transaction(buffers=True, write=True) as txn:
                yield self.deserialize_key(key)
                cursor = self.manager.txn.cursor()
                assert cursor.set_key(key)
                if not cursor.next():
                    break
                key = cursor.key()

    @ManageTransaction(buffers=True, write=False)
    def __len__(self):
        cursor, cursor_valid = self.__get_cursor_at_start()
        if not cursor_valid:
            return 0
        length = 0
        while self.key_prefix(cursor.key()) == self.prefix:
            length += 1
            cursor.next()
        return length

    @ManageTransaction(buffers=True, write=False)
    def clear(self): # pylint: disable=missing-docstring
        cursor, cursor_valid = self.__get_cursor_at_start()
        if cursor_valid:
            return
        while self.key_prefix(cursor.key()) == self.prefix:
            self.finalize_key(cursor.key())
            self.finalize_value(cursor.value())
            cursor.delete()

class PrefixManager(Scalar):
    """A class that Keeps track of next available prefix in the database."""

    def __init__(self, manager):
        Scalar.__init__(
            self,
            manager,
            bytearray([0]),
            value_serializer=ValueSerializer(
                serializer=lambda x: x,
                deserializer=bytearray
            )
        )

    def __call__(self):
        """Return next available prefix and increment."""
        def increment_counter(counter):
            """
            The counter first runs through all single bytes except 0xff, then
            all pairs of bytes where the first is 0xff and the second isn't
            0xff, then all strings of 4 bytes where the first two are 0xff and
            the last two aren't both 0xff, etc.
            """
            start = i = len(counter) >> 1
            while i < len(counter) and counter[i] == 0xff:
                i += 1
            else:
                counter[i] += 1
                if i + 1 == len(counter) and counter[i] == 0xff:
                    counter.extend([0x00] * len(counter))
                else:
                    while True:
                        i -= 1
                        if i < start:
                            break
                        else:
                            counter[i] = 0
            return counter
        if self.is_unset:
            self.set(bytearray([1]))
        prefix = self.get()
        next_prefix = prefix[:]
        increment_counter(next_prefix)
        self.set(next_prefix)
        next_prefix = self.get()
        return prefix

    def set_range(self, prefix):
        """Increment counter until it is past prefx."""
        while self.is_unset or prefix >= self.get():
            self()

class ValueManager(Dict): # pylint: disable=too-many-ancestors,too-many-public-methods
    """
    A dict-like class that keeps track of other persistent.Values.  Assigning a
    type creates a value of that type stored at the key assigned to.
    Serialization customization of stored types is not yet supported.
    """
    def __init__(self, path):
        self.path = path
        self.manager = TransactionManager(path)
        def serializer(value):
            """
            Serializes other persistent.Values by storing their pickled type and
            prefix.
            """
            if not issubclass(value, Value):
                raise ValueError("expected a subclass of Value")
            prefix = self.next_prefix()
            return buffer(pickle.dumps(value)) + prefix
        def deserializer(buf):
            """
            Deserializes by unpickling an initial segment and taking the rest
            to be the value's prefix.
            """
            fub = cStringIO.StringIO(buf)
            value = pickle.load(fub)
            prefix = buffer(buf, fub.tell())
            return value(self.manager, prefix)
        def finalizer(value): # pylint: disable=missing-docstring
            deserializer(value).clear()
        prefix = bytearray([1])
        Dict.__init__(
            self,
            self.manager,
            prefix,
            value_serializer=ValueSerializer(
                serializer=serializer,
                deserializer=deserializer,
                finalizer=finalizer
            ),
            key_serializer=ValueSerializer(
                serializer=lambda x: x,
                deserializer=lambda x: x,
                finalizer=lambda x: None
            )
        )
        self.next_prefix = PrefixManager(self.manager)
        self.next_prefix.set_range(self.prefix)
