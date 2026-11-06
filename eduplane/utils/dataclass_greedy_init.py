# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from dataclasses import MISSING, dataclass


def dataclass_greedy_init(cls):
    """Dataclass decorator that adds an 'init_from' class method to recursively
    initialize all fields and fully initialize an instance of the class from a
    given subset of fields specified as keyword arguments.

    In order to achieve this, the class is searched for *field init methods*,
    i.e. static methods decoarted with the 'init_field' decorator. A mapping
    from field names to these methods is built and stored as an attribute of
    the class. The 'init_from' method looks up appropriate methods given the
    set fields specified as keyword arguments in the 'init_from' class method.
    It initializes missing fields recursively in a greedy fashion,
    i.e. it initializes the first missing field for which a field init method
    can be found and all arguments to this field init method can be supplied.
    """

    # Collect all field init methods
    init_methods = inspect.getmembers(
        cls,
        lambda f: inspect.isfunction(f) and not inspect.ismethod(f) and hasattr(
            f, "init_field"))
    # Create a mapping from field names to signatures (i.e. required fields)
    # and field init methods.
    cls.init_mapping = {}
    for init_method_name, init_method in init_methods:
        init_field = init_method.init_field
        if init_field not in cls.init_mapping:
            cls.init_mapping[init_field] = []
        cls.init_mapping[init_field].append(
            (inspect.signature(init_method), init_method))

    # Define helper functions to iteratively initialize kwargs using field init
    # methods
    def __greedy_init_fields_step(
            init_fields, kwargs, default_values, verbose_fail, verbose_success):
        missing_fields = [f for f in init_fields if f not in kwargs]
        # Map default values and fields from kwargs to function
        # arguments, kwargs take precedence
        known_fields = dict(default_values, **kwargs)
        for field in missing_fields:
            # Try to call any field init methods
            if field in cls.init_mapping:
                for init_method_sig, init_method in cls.init_mapping[field]:
                    try:
                        mapped_args = {
                            p: known_fields[p]
                            for p in init_method_sig.parameters
                            if p in known_fields
                        }
                        bound_args = init_method_sig.bind(**mapped_args)
                        kwargs[field] = init_method(**bound_args.arguments)
                        if verbose_success:
                            print(
                                f"Initialized field {field} "
                                f"to value {kwargs[field]} "
                                f"from {bound_args.arguments} "
                                f"using {init_method.__name__}")
                        return True
                    except TypeError as err:
                        if verbose_fail:
                            print(
                                f"Could not initialize field {field} "
                                f"from init method {init_method.__name__} "
                                f"with signature {init_method_sig}: {err}")
                        pass
        return False

    def __greedy_init_fields(
            init_fields, kwargs, default_values, verbose_fail, verbose_success):
        """Iteratively and greedily search for field init methods matching any
        missing fields and try to invoke them using the already known field
        values (preferably) or given default values."""
        while __greedy_init_fields_step(init_fields, kwargs, default_values,
                                        verbose_fail, verbose_success):
            pass
        return

    # Add classmethod 'init_from'
    def init_from(cls, *, verbose=False, **kwargs):
        """Initialize dataclass instance from from a set of keyword arguments.
        """
        # Get a list of all fields that can be initialized
        init_fields = [
            f for f in cls.__dataclass_fields__
            if cls.__dataclass_fields__[f].init
        ]
        # Try to initialize missing fields from kwargs only
        if verbose:
            print(f"Greedy initialization of {cls.__name__}: 1st run")
        __greedy_init_fields(
            init_fields,
            kwargs, {},
            verbose_fail=False,
            verbose_success=verbose)
        # Try to initialize missing fields from kwargs and default values.
        # NOTE: Additionally to missing default values, None default values are
        # also excluded as they are usually used to intentionally have
        # uninitialized fields, e.g. to have fields which may be initialized
        # in post-init method. Also, any fields for which init methods exist
        # are excluded as otherwise the default value could be used in another
        # field init method before the actual value of the field is initialized.
        default_values = {
            field: cls.__dataclass_fields__[field].default
            for field in init_fields
            if cls.__dataclass_fields__[field].default not in (
                MISSING, None) and field not in cls.init_mapping
        }
        if verbose:
            print(
                f"Greedy initialization of {cls.__name__}: "
                f"2nd run with default values {default_values}")
        __greedy_init_fields(
            init_fields, kwargs, default_values, verbose, verbose)

        return cls(**kwargs)

    cls.init_from = classmethod(init_from)
    return dataclass(cls)


def init_field(field_name):
    """Decorator to be used in combination with 'dataclass_greedy_init' to
    generate static methods with an additional 'field_name' attribute that
    indicates for which of the dataclass's fields this method should be used
    during initialization."""

    def inner(func):
        func.init_field = field_name
        return staticmethod(func)

    return inner
