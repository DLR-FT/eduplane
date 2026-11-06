# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
import textwrap
from collections import OrderedDict

import numpy
import sympy
import unyt
# Re-exported as part of this module's public API: other modules import these
# math functions from eduplane.utils.smath rather than from sympy directly.
from sympy import (  # noqa: F401
    And, Max, Min, Piecewise, acos, asin, atan, atan2, cos, pi, sin, sqrt, tan)
from sympy.parsing.sympy_parser import parse_expr
from sympy.physics.mechanics import (
    dynamicsymbols, find_dynamicsymbols, init_vprinting)

# sympy.init_printing()
init_vprinting()

# TODO: Make default argument values consistent: name_key="name" -> name_key=None

config = {
    # The default backend to use with sympy.lambdify.
    "default_backend": "numpy",
    # If True, all symbols which are not assumed to be constant will be defined
    # as function of time t.
    "enable_dynamic_symbols": False,
    # Output to be produced by 'display' function ('text' or 'markdown')
    "display_mode": "markdown",
}


def display_markdown(*args, latex_mode="inline"):
    # Imported lazily so that IPython is only required when rich display is
    # actually used (e.g. inside a notebook). This keeps IPython out of the
    # package's runtime dependencies.
    from IPython.display import Markdown
    from IPython.display import display as ipy_display
    md_str = ""
    for arg in args:
        if isinstance(arg, str):
            md_str += arg + ' '
        else:
            md_str += "{} ".format(sympy.latex(arg, mode=latex_mode))
    ipy_display(Markdown(md_str))


def display_text(*args, **kwargs):
    text_str = ""
    for arg in args:
        if isinstance(arg, str):
            text_str += arg + ' '
        else:
            arg_str = sympy.pretty(arg)
            if text_str:
                arg_str = textwrap.indent(arg_str, "  ")
            if '\n' in arg_str:
                text_str += "\n" + arg_str
            else:
                text_str += arg_str + " "
    print(text_str)


def display(*args, **kwargs):
    if config["display_mode"] == "markdown":
        display_markdown(*args, **kwargs)
    elif config["display_mode"] == "text":
        display_text(*args, **kwargs)
    else:
        raise ValueError(
            "Unknown display mode: {}".format(config["display_mode"]))


def get_symbol_name(sid, symbols, name_key=None):
    """Get a name for a symbol given its ID."""
    # If no name key was given, use ID as name
    if name_key is None:
        name = sid
    else:
        name = symbols[sid][name_key]
    return name


def create_symbol(sid, symbols, name_key=None, no_dynamic_symbols=False):
    """Create a symbol from the symbols list."""
    if sid not in symbols:
        raise ValueError(f"Failed to find symbol '{sid}' in list of symbols")
    assumptions = {"real": True}
    if "assumptions" in symbols[sid]:
        assumptions.update(symbols[sid]["assumptions"])
    if (not config["enable_dynamic_symbols"] or no_dynamic_symbols
            or "constant" in assumptions and assumptions["constant"] is True):
        return sympy.Symbol(
            get_symbol_name(sid, symbols, name_key), **assumptions)
    else:
        # Note that we cannot use sympy's find_dynamicsymbols as it cannot
        # handle symbol names containing commas (e.g. V_{K,xy} would be split
        # into multiple symbols)
        return sympy.Function(
            get_symbol_name(sid, symbols, name_key),
            **assumptions)(dynamicsymbols._t)


def get_symbols(expr, symbols, name_id_map=None, name_key=None, exclude=['t']):
    """Retrieve symbol objects of a given expression which are included in the
    list of symbols. Symbol objects may include free symbols and dynamic
    symbols."""
    if name_id_map is None:
        name_id_map = get_name_id_map(symbols, name_key)
    # Extract free and dynamic symbols
    free_symbols = [s for s in expr.free_symbols if s.name not in exclude]
    dyn_symbols = [s for s in find_dynamicsymbols(expr, exclude=exclude)]
    # Return only symbols found in the list of symbols
    result_symbols = [
        s for s in free_symbols + dyn_symbols
        if hasattr(s, "name") and (s.name in symbols or s.name in name_id_map)
    ]
    return result_symbols


def get_dimension(dim_expr):
    """Get unyt dimension from its name or an expression, e.g. 'length/time**2'."""
    try:
        return getattr(unyt.dimensions, dim_expr)
    except AttributeError:
        return parse_expr(dim_expr, local_dict=unyt.dimensions.__dict__)


def create_symbols(symbols, name_key=None, no_dynamic_symbols=False):
    """Create sympy symbols from the list of symbols.
    If a name key is given, the symbols' names are looked up using it.
    Otherwise, the symbols are named by their IDs.

    Tip: To conviently work with the created symbols, inject them into the
         global namespace: `globals().update(symbol_objects)`
    """
    symbol_objects = {
        sid: create_symbol(sid, symbols, name_key, no_dynamic_symbols)
        for sid in symbols
    }
    return symbol_objects


def func2expression(
        func, lhs, symbols, name_key=None, no_dynamic_symbols=False):
    """Generate a symbolic expression from a function."""
    signature = inspect.signature(func)
    symbolic_args = OrderedDict(
        (
            arg,
            create_symbol(
                arg,
                symbols,
                name_key=name_key,
                no_dynamic_symbols=no_dynamic_symbols))
        for arg in signature.parameters)
    func_symbol = create_symbol(
        lhs, symbols, name_key=name_key, no_dynamic_symbols=no_dynamic_symbols)
    symbolic_expr = func(**symbolic_args)
    return func_symbol, symbolic_args, symbolic_expr


def remove_dynamic_symbols(expr, symbols, name_key=None, name_id_map=None):
    if name_id_map is None:
        name_id_map = get_name_id_map(symbols, name_key=name_key)
    name_id_map.update(get_name_id_map(symbols, name_key=None))
    for dyn_symb in find_dynamicsymbols(expr):
        try:
            if dyn_symb.name in name_id_map:
                const_symb = create_symbol(
                    name_id_map[dyn_symb.name],
                    symbols,
                    name_key,
                    no_dynamic_symbols=True)
                expr = expr.subs(dyn_symb, const_symb)
        except AttributeError:
            print(f"Warning: Failed to remove dynamic symbol: {dyn_symb}")
    return expr


def get_name_id_map(symbols, name_key="name"):
    """Get a mapping from symbols' names to IDs from a list of symbols."""
    return {
        (spec[name_key] if name_key in spec else sid): sid
        for sid, spec in symbols.items()
    }


def rename_symbols(
        expr, symbols, name_id_map=None, old_name_key=None, new_name_key=None):
    """Rename all symbols in an expression given an old name to new name mapping.
    Note that by default, a name-ID map is created that maps from both old and new names
    to the IDs to make the operation idempotent.
    """
    # Get a map from old names to IDs
    if name_id_map is None:
        name_id_map = get_name_id_map(symbols, name_key=old_name_key)
    name_id_map.update(get_name_id_map(symbols, name_key=new_name_key))
    # Get symbol IDs
    symbol_objects = get_symbols(expr, symbols, name_id_map)
    symbol_ids = [name_id_map[symb.name] for symb in symbol_objects]
    # Rename symbols using substitution
    for symb, sid in zip(symbol_objects, symbol_ids):
        expr = expr.subs(
            symb, create_symbol(sid, symbols, name_key=new_name_key))
    return expr


def expression2func(
        expr, symbols={}, backend=None, name_id_map=None, name_key=None):
    """Generate an smath function from a symbolic expression.
    As this involves renaming all symbols in the expression to their symbol IDs,
    the renamed expression is returned along with the generated lambda function."""
    if not backend:
        backend = config["default_backend"]
    # Rename all symbols to their IDs (which are valid Python specifiers)
    expr = rename_symbols(
        expr,
        symbols,
        name_id_map=name_id_map,
        old_name_key=name_key,
        new_name_key=None)
    # In order to generate a lambda function from the expression, any dynamic
    # symbols have to be substituted by "regular" free symbols first.
    expr = remove_dynamic_symbols(
        expr, symbols, name_key=None, name_id_map=None)
    # Generate a lambda function from the sympy expression for numeric evaluation
    # with the user-specified math/numerical backend
    func_lambda = sympy.lambdify(list(expr.free_symbols), expr, backend)
    return expr, func_lambda


class smath_func(object):
    """A decorator and factory class to construct smath functions.

    TODO: Consider adding sympy-native symbolic dimensionality check (e.g. using
    ``dimsys_SI.equivalent_dims``).
    """

    def __init__(self, symbols, returns=None, backend=None, name_key="name"):
        """Initialize decorator from a list of symbols.
        Optionally the symbol returned (i.e. the equation's left-hand side)
        and the backend for numerical evaluation may be specified.
        """
        if not backend:
            backend = config["default_backend"]
        self.symbols = symbols
        self.returns = returns
        self.backend = backend
        self.name_key = name_key

    def __call__(self, func):
        """Construct an smath function from a given function."""
        # Determine ID of left-hand-side symbol
        if self.returns:
            lhs = self.returns
        else:
            lhs = func.__name__
        if lhs not in self.symbols:
            raise ValueError(
                f"Could not find left-hand side symbol {lhs} of function {func.__name__} in list of symbols"
            )
        # Get function signature and check if arguments and function name
        # can be found in the list of symbols
        func_sig = inspect.signature(func)
        missing_args = [
            arg for arg in func_sig.parameters if arg not in self.symbols
        ]
        if missing_args:
            raise ValueError(
                f"Missing arguments for function '{func.__name__}' in list of symbols: {missing_args}"
            )
        # Generate a sympy expression using the symbol IDs as symbol names.
        # This one will be used to generate a lambda function for numeric evaluation.
        symbolic_args, symbolic_expr = func2expression(
            func, lhs, self.symbols, name_key=None, no_dynamic_symbols=True)[1:]
        # Generate a lambda function from the sympy expression for numeric evaluation
        # with the user-specified math/numerical backend
        func_lambda = sympy.lambdify(
            symbolic_args.values(), symbolic_expr, self.backend)
        # NOTE: Here, the lambdified function is used directly (rather than
        # wrapping it in an extra Python call layer), so invocation costs *should*
        # be identical to native ``lambdify``.
        # The metadata below is attached as attributes on the same object.
        # (The lambdified function already accepts both positional and keyword arguments.)
        wrapper = func_lambda

        wrapper.func_lambda = func_lambda
        wrapper.func_name = func.__name__
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__

        # Generate another sympy expression with symbols named as in list of symbols.
        # This one will be provided to the user for further use.
        named_func_symbol, __, named_symbolic_expr = func2expression(
            func, lhs, self.symbols, self.name_key)
        # Store the sympy expression and symbolic arguments as function attributes
        wrapper.lhs = named_func_symbol
        wrapper.rhs = named_symbolic_expr
        wrapper.eq = sympy.Eq(named_func_symbol, named_symbolic_expr)

        # Create wrapper function that converts units and checks dimensions
        self.__add_with_units(
            wrapper, func_sig, lhs, symbolic_expr, symbolic_args.values(),
            self.symbols, False)
        self.__add_check_dimensions(wrapper, func_sig)

        # Create and add 'eval_if' function
        self.__add_eval_if(wrapper, func_sig)

        return wrapper

    @staticmethod
    def from_expr(lhs, expr, symbols, backend=None, name_key="name"):
        """Construct an smath function from a sympy expression."""
        if not backend:
            backend = config["default_backend"]
        # Generate a lambda function from the sympy expression for numeric evaluation
        # with the user-specified math/numerical backend.
        # Note that the function and its arguments will be named after symbol IDs.
        name_id_map = get_name_id_map(symbols, name_key)
        expr, func_lambda = expression2func(
            expr, symbols, backend, name_id_map, name_key)
        func_lambda.__name__ = get_symbol_name(lhs, name_id_map, None)
        func_sig = inspect.signature(func_lambda)

        # Define a wrapper function that only accepts kwargs
        # as there is no obvious order for args
        @functools.wraps(func_lambda)
        def wrapper(**kwargs):
            return func_lambda(**kwargs)

        # Rename all symbols in the given expression as specified in the list of symbols.
        # This one will be provided to the user for further use.
        named_func_symbol = create_symbol(lhs, symbols, name_key=name_key)
        named_symbolic_expr = rename_symbols(
            expr, symbols, old_name_key=None, new_name_key=name_key)

        # Store the sympy expression as function attributes
        wrapper.lhs = named_func_symbol
        wrapper.rhs = named_symbolic_expr
        wrapper.eq = sympy.Eq(named_func_symbol, named_symbolic_expr)

        # Create wrapper function that converts units and checks dimensions
        smath_func.__add_with_units(
            wrapper,
            func_sig,
            lhs,
            expr,
            get_symbols(expr, symbols, name_id_map, name_key),
            symbols,
            kwonly=True)
        smath_func.__add_check_dimensions(wrapper, func_sig)

        # Create and add 'eval_if' function
        smath_func.__add_eval_if(wrapper, func_sig)

        return wrapper

    @staticmethod
    def __add_with_units(
            func, func_sig, lhs, rhs, rhs_symbolic_args, symbols, kwonly):
        # Create wrapper function that converts units and checks dimensions
        arg_dimensions = {
            arg: get_dimension(symbols[arg]["dimension"])
            for arg in func_sig.parameters
        }
        # Note that we have to use numpy as math/numerical backend to work with unyt
        result_dimension = get_dimension(symbols[lhs]["dimension"])
        numpy_ufunc = sympy.lambdify(rhs_symbolic_args, rhs, "numpy")
        # Workaround for issue about checking dimensionless results with the 'returns' decorator
        # See: https://github.com/yt-project/unyt/issues/146
        if result_dimension != unyt.dimensions.dimensionless:
            numpy_ufunc_dim_check = unyt.returns(result_dimension)(numpy_ufunc)
        else:
            numpy_ufunc_dim_check = numpy_ufunc
        numpy_ufunc_dim_check = unyt.accepts(
            **arg_dimensions)(numpy_ufunc_dim_check)

        def __process_bound_args(bound_args):
            for arg in bound_args.arguments.keys():
                desired_unit = symbols[arg]["unit"]
                # Try to convert argument to desired unit or apply desired unit
                # if the given value is not a quantity with unit
                try:
                    bound_args.arguments[arg].convert_to_units(desired_unit)
                except AttributeError:
                    bound_args.arguments[arg] = unyt.unyt_quantity(
                        bound_args.arguments[arg], desired_unit)

        if kwonly:

            @functools.wraps(func)
            def with_units(**kwargs):
                # Note that we have to use 'bind_partial' here because
                # we have added default arguments dynamically above.
                bound_args = func_sig.bind(**kwargs)
                __process_bound_args(bound_args)
                return numpy_ufunc_dim_check(**bound_args.arguments)
        else:

            @functools.wraps(func)
            def with_units(*args, **kwargs):
                # Note that we have to use 'bind_partial' here because
                # we have added default arguments dynamically above.
                bound_args = func_sig.bind(*args, **kwargs)
                __process_bound_args(bound_args)
                return numpy_ufunc_dim_check(**bound_args.arguments)

        func.with_units = with_units

    @staticmethod
    def __add_check_dimensions(func, func_sig):

        def check_dimensions(raise_except=False, **kwargs):
            default_args = {p: 1 for p in func_sig.parameters}
            args = {**default_args, **kwargs}
            try:
                func.with_units(**args)
            except unyt.exceptions.UnitConversionError as err:
                print(f"Unit conversion error in {func.__name__}: {err}")
                if raise_except:
                    raise err
                else:
                    return False
            except unyt.exceptions.UnitOperationError as err:
                print(f"Unit operation error in {func.__name__}: {err}")
                if raise_except:
                    raise err
                else:
                    return False
            except RuntimeWarning:
                pass
            except Exception as err:
                print(
                    f"Failed to check dimensions and units of {func.__name__}: {err}"
                )
                if raise_except:
                    raise err
                else:
                    return False
            return True

        func.check_dimensions = check_dimensions

    @staticmethod
    def __add_eval_if(func, func_sig):

        def eval_if(dict_or_df={}, mask=True, fill_value=numpy.nan, **kwargs):
            """Evaluate smath function taking the function's arguments from
            a dictionary, a pandas.DataFrame or from additional keyword
            arguments.

            For dataframes, a mask (boolean array) may be given to constrain
            the rows for which to evaluate the function.
            For dictionaries, a single boolean value may be given to indicate
            wether or not to evaluate the function. This allows for a
            consistent syntax for passing both dicts and dataframes.
            """
            func_inputs = [
                (dict_or_df[p] if p in dict_or_df else kwargs[p])
                for p in func_sig.parameters
            ]
            if isinstance(mask, (bool, numpy.bool_)):
                if mask:
                    return func.func_lambda(*func_inputs)
                else:
                    return fill_value
            else:
                mask = numpy.asarray(mask)
                func_inputs_masked = [
                    numpy.broadcast_to(input, mask.shape)[mask]
                    for input in func_inputs
                ]
                result = numpy.full(mask.shape, fill_value, dtype=float)
                result[mask] = func.func_lambda(*func_inputs_masked)
                return result

        func.eval_if = eval_if


def parse_units(symbols):
    """Parse unit expressions in the given list of symbols and replace with unyt.Unit objects.
    This may provide a runtime performance benefit as units are parsed and instantiated only once."""
    for spec in symbols.values():
        if "unit" in spec:
            spec["unit"] = unyt.Unit(spec["unit"])


def get_definitions(symbols, name_key="name", name_prefix="", backend=None):
    """Parse symbol definitions from a list of symbols and create corresponding
    smath functions."""
    definitions = {}
    # Definition expressions are required to use symbol IDs, so we have to
    # provide accordingly named symbols for the expression parser to use.
    symbol_objects = create_symbols(
        symbols, name_key=None, no_dynamic_symbols=False)
    # Parse definition expression for symbols in the list of symbols.
    for symb_id, symb_spec in symbols.items():
        if "definition" in symb_spec:
            # Parse expression based on symbol IDs or names
            expr = parse_expr(
                symb_spec["definition"], local_dict=symbol_objects)
            # Without dynamic symbols enabled, all symbols are treated as
            # constants and all derivatives in definitions will be one.
            if not config["enable_dynamic_symbols"] and expr == 1:
                continue
            # Create and store smath func object
            smath_f = smath_func.from_expr(
                symb_id, expr, symbols, backend, name_key)
            definitions[name_prefix + symb_id] = smath_f
    return definitions


def is_zero(expr):
    """Return ``True`` if ``expr`` is symbolically (or numerically) zero.

    Uses ``sympy.simplify`` first and falls back to ``Expr.equals`` (which
    samples numerically) for identities that a full simplification cannot
    close.
    """
    if sympy.simplify(expr) == 0:
        return True
    try:
        return bool(expr.equals(0))
    except Exception:
        return False


def solve_equiv(eq, symbol, func):
    """Solve ``eq`` for ``symbol`` and check it equals ``func``'s right-hand side.

    Returns ``True`` if any solution of ``eq`` for ``symbol`` simplifies to the
    symbolic right-hand side ``func.eq.rhs`` (where ``func`` is an
    ``smath_func`` decorated function). Used for "solve the source equation
    for the target symbol and compare" proofs.
    """
    sols = sympy.solve(eq, symbol)
    if not sols:
        return False
    return any(is_zero(s - func.eq.rhs) for s in sols)


def positive_branch(piecewise):
    """Return the positive-sign branch expression of a two-branch ``Piecewise``.

    Given a ``sympy.Piecewise`` of the form ``((+expr, cond), (-expr, True))``
    (i.e. a positive and a negative sign branch, with the negative branch as
    the ``True`` fallback), returns ``+expr``.

    This avoids relying on branch ordering via ``.args[0][0]``: it selects the
    branch whose condition is not ``sympy.true`` (the explicit condition).
    """
    if not isinstance(piecewise, sympy.Piecewise):
        raise ValueError(
            f"Expected a Piecewise, got {type(piecewise).__name__}")
    explicit = [e for e, c in piecewise.args if c is not sympy.true]
    if len(explicit) != 1:
        raise ValueError(
            "Expected a two-branch Piecewise with a single explicit condition; "
            f"got conditions {[c for _, c in piecewise.args]}")
    return explicit[0]
