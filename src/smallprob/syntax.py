"""Source-based decoration of ordinary, deliberately restricted Python models."""

from __future__ import annotations

import ast
from contextvars import ContextVar
import functools
import inspect
import operator
import re
import textwrap
import uuid

from .core import Categorical, Expr, Graph, HalfNormal, ModelError, Node, Normal, StudentT, UnsupportedModelError, as_expr, lookup, _contained_dependencies, _normal_components, _student_t_components, _snapshot


_active_builder: ContextVar = ContextVar("smallprob_active_builder", default=None)
_SITE = re.compile(r"^([A-Za-z_][A-Za-z_0-9]*)(?:\[(0|[1-9][0-9]*)\])?$")
_DISTRIBUTIONS = {"Normal": Normal, "HalfNormal": HalfNormal, "StudentT": StudentT, "Categorical": Categorical}


class _Builder:
    def __init__(self):
        self.nodes = []
        self.groups = {}
        self.series = set()
        self.names = set()
        self.collections = {}

    def add(self, distribution, name, series):
        if not isinstance(distribution, (Normal, HalfNormal, StudentT, Categorical)):
            raise UnsupportedModelError("draw() currently supports Normal, HalfNormal, StudentT and Categorical distributions.")
        if not isinstance(name, str) or not (match := _SITE.fullmatch(name)):
            raise ModelError("draw() names must be Python identifiers or indexed names such as 'y[0]'.")
        group, explicit_index = match.groups()
        if isinstance(distribution, (Normal, StudentT)):
            distribution_type = type(distribution)
            components = _normal_components if isinstance(distribution, Normal) else _student_t_components
            scalar_parameters, parameters = components(distribution)
            if parameters is not None:
                if series or explicit_index is not None:
                    raise UnsupportedModelError(
                        f"A vector {distribution_type.__name__} declaration must be assigned to its own series name; "
                        "nested vectors in draw lists, comprehensions, append or indexed names are unsupported."
                    )
                if group in self.groups:
                    raise UnsupportedModelError(f"Random series {group!r} cannot be redeclared; use a new variable name.")
                self.register_series(group)
                values = [self.add(distribution_type(*item), group, True) for item in parameters]
                return _register_series(group, values)
            distribution = distribution_type(*scalar_parameters)
        is_series = series or explicit_index is not None
        if group in self.groups and ((group in self.series) != is_series):
            raise ModelError(f"Name {group!r} cannot be both a scalar and a series.")
        if is_series:
            self.series.add(group)
            members = self.groups.setdefault(group, [])
            index = len(members) if explicit_index is None else int(explicit_index)
            if index != len(members):
                raise ModelError(f"Series {group!r} must be declared in order starting at index 0; next index is {len(members)}.")
            site = f"{group}[{index}]"
        else:
            site = group
            members = self.groups.setdefault(group, [])
        if site in self.names:
            raise ModelError(
                f"Random variable {site!r} was declared twice. "
                "Use a list comprehension, list.append(draw(...)), or explicit indexed names for repeated draws."
            )
        self.names.add(site)
        members.append(site)
        self.nodes.append(Node(site, distribution))
        return Expr("ref", (site,))

    def graph(self):
        return Graph(self.nodes, self.groups, self.series)

    def register_series(self, name):
        if name in self.groups and name not in self.series:
            raise ModelError(f"Name {name!r} cannot be both a scalar and a series.")
        self.groups.setdefault(name, [])
        self.series.add(name)


def _register_series(name, value):
    builder = _active_builder.get()
    if name is None:
        # An explicit f-string name can only be resolved after iteration. Do
        # not let that escape hatch bypass the same index stability guarantees
        # as an implicitly named list.
        if not value:
            raise UnsupportedModelError(
                "Cannot infer the group of an empty series with dynamic explicit names. "
                "Omit name= and let the assignment name the series."
            )
        groups = []
        for item in value:
            match = _SITE.fullmatch(item.args[0]) if isinstance(item, Expr) and item.op == "ref" else None
            if match is None or match.group(2) is None:
                raise UnsupportedModelError("A named draw list must contain one indexed random series.")
            groups.append(match.group(1))
        if len(set(groups)) != 1:
            raise UnsupportedModelError(
                "Explicit names in one draw list must belong to the same indexed series. "
                "Omit name= to infer consistent names."
            )
        name = groups[0]
    builder.register_series(name)
    if name in builder.collections:
        raise UnsupportedModelError(f"Random series {name!r} cannot be reassigned; use a new variable name.")
    if not isinstance(value, list):
        # Tuples are immutable, but rebinding them can still change which
        # symbolic site a positional index refers to.
        builder.collections[name] = value
        return value
    tracked = _SeriesList(name, value)
    builder.collections[name] = tracked
    return tracked


class _SeriesList(list):
    """A named series can grow by draws, but its site indices are permanent."""

    def __init__(self, name, values):
        self.name = name
        super().__init__()
        for value in values:
            self.append(value)

    # Arithmetic means elementwise random-variable arithmetic, never Python
    # list concatenation or repetition. The parameter lowering step resolves
    # its shape and broadcasts fixed scalars without evaluating random values.
    __array_priority__ = 1000

    def __add__(self, other):
        return as_expr(self) + other

    def __radd__(self, other):
        return other + as_expr(self)

    def __sub__(self, other):
        return as_expr(self) - other

    def __rsub__(self, other):
        return other - as_expr(self)

    def __mul__(self, other):
        return as_expr(self) * other

    def __rmul__(self, other):
        return other * as_expr(self)

    def __truediv__(self, other):
        return as_expr(self) / other

    def __rtruediv__(self, other):
        return other / as_expr(self)

    def __pow__(self, other):
        return as_expr(self) ** other

    def __rpow__(self, other):
        return other ** as_expr(self)

    def __neg__(self):
        return -as_expr(self)

    def __pos__(self):
        return as_expr(self)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return as_expr(self).__array_ufunc__(ufunc, method, *inputs, **kwargs)

    def append(self, value):
        expected = f"{self.name}[{len(self)}]"
        if not isinstance(value, Expr) or value.op != "ref" or value.args[0] != expected:
            raise UnsupportedModelError(
                f"Random series {self.name!r} can only append its next declared draw, {expected!r}. "
                "Keep deterministic values in a separate list."
            )
        super().append(value)

    def _reject_mutation(self, *args, **kwargs):
        raise UnsupportedModelError(
            f"Cannot reorder, replace or remove items in random series {self.name!r}: "
            "its indices are observation names. Create a separate derived list instead."
        )

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    __iadd__ = _reject_mutation
    __imul__ = _reject_mutation
    reverse = _reject_mutation
    sort = _reject_mutation
    clear = _reject_mutation
    pop = _reject_mutation
    remove = _reject_mutation
    insert = _reject_mutation
    extend = _reject_mutation


def _append_series(collection, item):
    if not isinstance(collection, _SeriesList):
        raise UnsupportedModelError(
            "Append draws only to a model series initialised with [], a draw list, "
            "or a draw comprehension; supplied input lists cannot become random series."
        )
    collection.append(item)


def _bind_name(name, value):
    builder = _active_builder.get()
    if name in builder.collections and value is not builder.collections[name]:
        raise UnsupportedModelError(
            f"Random series {name!r} cannot be rebound: its indices are observation names. "
            "Assign a reordered or transformed list to a different variable."
        )
    if name in builder.groups and name not in builder.series:
        if not isinstance(value, Expr) or value.op != "ref" or value.args[0] != name:
            raise UnsupportedModelError(
                f"Random variable {name!r} cannot be rebound; use a new name for its transformed value."
            )
    return value


def _guard_list_mutator(method):
    @functools.wraps(method)
    def guarded(collection, *args, **kwargs):
        if isinstance(collection, _SeriesList):
            collection._reject_mutation()
        return method(collection, *args, **kwargs)
    return guarded


_COMPARISONS = {
    "Eq": operator.eq, "NotEq": operator.ne, "Lt": operator.lt,
    "LtE": operator.le, "Gt": operator.gt, "GtE": operator.ge,
    "Is": operator.is_, "IsNot": operator.is_not,
    "In": lambda left, right: left in right,
    "NotIn": lambda left, right: left not in right,
}


def _compare_chain(operations, *operands):
    """Preserve Python's chained-comparison short circuit and reject symbols.

    Identity and membership comparisons do not necessarily call Expr methods;
    checking here prevents them silently deciding a branch from object identity.
    """
    left = operands[0]()
    for index, (operation, operand) in enumerate(zip(operations, operands[1:])):
        right = operand()
        if isinstance(left, Expr) or isinstance(right, Expr) or _contained_dependencies(left) or _contained_dependencies(right):
            raise UnsupportedModelError(
                "Comparisons involving random variables are not supported while building a model. "
                "Use fixed inputs for Python control flow."
            )
        result = _COMPARISONS[operation](left, right)
        if index == len(operations) - 1 or not result:
            return result
        left = right
    return result


def draw(distribution, *, name=None, _series=False):
    """Declare a random variable inside a source-readable ``@model`` function.

    Names are normally inferred from an assignment, list comprehension or
    append. ``name=`` is useful when that inference is ambiguous.
    """
    builder = _active_builder.get()
    if builder is None:
        raise ModelError("draw() must be called inside a @model function; use .simulate() to generate data.")
    if name is None:
        raise UnsupportedModelError("Cannot infer this draw's name. Assign it directly or supply draw(..., name='variable').")
    return builder.add(distribution, name, _series)


def _is_draw(node):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "draw"


def _is_distribution(node):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _DISTRIBUTIONS


def _is_declaration(node):
    return _is_draw(node) or _is_distribution(node)


def _explicit_name(node):
    return next((keyword.value for keyword in node.keywords if keyword.arg == "name"), None) if _is_draw(node) else None


def _checked_constructor(constructor, name):
    # Global and closure bindings can change after decoration, including while
    # executing model helpers. Check the actual callable at each declaration.
    if constructor is not _DISTRIBUTIONS[name]:
        raise UnsupportedModelError(
            f"The shorthand constructor {name!r} is shadowed or has been rebound. "
            f"Import {name} directly from smallprob without rebinding it, "
            "or use an explicit draw(...) with your distribution factory."
        )
    return constructor


def _bound_names(definition):
    """Conservative scope check: a local binding makes shorthand ambiguous.

    Comprehension and lambda bindings are included deliberately. Keeping
    constructor names unshadowed throughout a model is a small, clear rule.
    """
    names = set()
    for node in ast.walk(definition):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not definition:
            names.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
    return names


class _Transformer(ast.NodeTransformer):
    def __init__(self, draw_helper, lookup_helper, series_helper, append_helper, bind_helper, mutator_helper, compare_helper, constructor_helper, append_names, bound_names, environment):
        self.draw_helper = draw_helper
        self.lookup_helper = lookup_helper
        self.series_helper = series_helper
        self.append_helper = append_helper
        self.bind_helper = bind_helper
        self.mutator_helper = mutator_helper
        self.compare_helper = compare_helper
        self.constructor_helper = constructor_helper
        self.append_names = append_names
        self.bound_names = bound_names
        self.environment = environment
        self.context = None
        self.distribution_depth = 0

    def _register(self, name, expression):
        return ast.copy_location(
            ast.Call(func=ast.Name(id=self.series_helper, ctx=ast.Load()),
                     args=[ast.Constant(name), expression], keywords=[]), expression
        )

    def _visit_with(self, node, context=None):
        old = self.context
        self.context = context
        try:
            return self.visit(node)
        finally:
            self.context = old

    def _value(self, value, name):
        if _is_declaration(value):
            return self._visit_with(value, (name, False))
        if isinstance(value, ast.ListComp) and _is_declaration(value.elt):
            explicit = _explicit_name(value.elt)
            series_name = name if explicit is None else (
                explicit.value if isinstance(explicit, ast.Constant) and isinstance(explicit.value, str)
                and explicit.value.isidentifier() else None
            )
            value.elt = self._visit_with(value.elt, (name, True))
            value.generators = [self._visit_with(generator) for generator in value.generators]
            return self._register(series_name, value)
        if isinstance(value, (ast.List, ast.Tuple)) and value.elts and all(_is_declaration(element) for element in value.elts):
            explicit_names = [_explicit_name(element) for element in value.elts]
            series_name = name
            if any(explicit is not None for explicit in explicit_names):
                # Explicit names may describe unrelated sites. Only auto-track
                # a list when every site belongs to one identifiable series.
                groups = []
                for explicit in explicit_names:
                    if explicit is None:
                        groups.append(name)
                    elif isinstance(explicit, ast.Constant) and isinstance(explicit.value, str) and (match := _SITE.fullmatch(explicit.value)):
                        groups.append(match.group(1))
                    else:
                        groups.append(None)
                series_name = groups[0] if groups[0] is not None and all(group == groups[0] for group in groups) else None
            value.elts = [self._visit_with(element, (name, True)) for element in value.elts]
            return self._register(series_name, value)
        if isinstance(value, ast.List) and not value.elts and name in self.append_names:
            return self._register(name, value)
        return self._visit_with(value)

    def visit_Assign(self, node):
        if len(node.targets) == 1:
            node.value = self._assigned_value(node.targets[0], node.value)
        else:
            node.value = self._visit_with(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    node.value = self._bind(target.id, node.value)
        node.targets = [self._visit_with(target) for target in node.targets]
        return node

    def _assigned_value(self, target, value):
        if isinstance(target, ast.Name):
            return self._bind(target.id, self._value(value, target.id))
        if (
            isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List))
            and len(target.elts) == len(value.elts)
        ):
            value.elts = [self._assigned_value(item, expression) for item, expression in zip(target.elts, value.elts)]
            return value
        return self._visit_with(value)

    def _bind(self, name, value):
        return ast.copy_location(ast.Call(
            func=ast.Name(id=self.bind_helper, ctx=ast.Load()),
            args=[ast.Constant(name), value], keywords=[]), value)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            node.value = self._assigned_value(node.target, node.value)
        node.target = self._visit_with(node.target)
        return node

    def visit_Call(self, node):
        if _is_distribution(node) and self.distribution_depth == 0:
            name = node.func.id
            if name in self.bound_names:
                raise UnsupportedModelError(
                    f"The shorthand constructor {name!r} is shadowed by a local binding. "
                    "Keep constructor names unshadowed; use another name for inputs, loops and local variables."
                )
            _checked_constructor(self.environment.get(name), name)
            if self.context is None:
                raise UnsupportedModelError(
                    "Cannot infer a shorthand declaration's name in this expression. "
                    "Assign it directly, use a declaration list/comprehension or append, "
                    "or supply draw(..., name='variable') explicitly."
                )
            node.func = ast.copy_location(ast.Call(
                func=ast.Name(id=self.constructor_helper, ctx=ast.Load()),
                args=[node.func, ast.Constant(name)], keywords=[]), node.func)
            wrapped = ast.copy_location(ast.Call(
                func=ast.Name(id="draw", ctx=ast.Load()), args=[node], keywords=[]), node)
            return self.visit_Call(wrapped)
        if _is_draw(node):
            context = self.context
            explicit_name = any(keyword.arg == "name" for keyword in node.keywords)
            if any(keyword.arg == "_series" or keyword.arg is None for keyword in node.keywords):
                raise UnsupportedModelError("draw() does not support **kwargs or the reserved _series argument.")
            if not explicit_name and context is None:
                raise UnsupportedModelError(
                    "Cannot infer a draw's name in this expression. "
                    "Use 'x = draw(...)', 'x = [draw(...) for ...]', 'x.append(draw(...))', "
                    "or give draw(..., name='x') explicitly."
                )
            self.distribution_depth += 1
            try:
                node.args = [self._visit_with(argument) for argument in node.args]
            finally:
                self.distribution_depth -= 1
            node.keywords = [ast.keyword(arg=keyword.arg, value=self._visit_with(keyword.value)) for keyword in node.keywords]
            if not explicit_name:
                node.keywords.append(ast.keyword(arg="name", value=ast.Constant(context[0])))
            if context is not None and context[1]:
                node.keywords.append(ast.keyword(arg="_series", value=ast.Constant(True)))
            node.func = ast.Name(id=self.draw_helper, ctx=ast.Load())
            return node
        if (
            isinstance(node.func, ast.Attribute) and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name) and len(node.args) == 1
            and _is_declaration(node.args[0]) and not node.keywords
        ):
            node.args = [self._visit_with(node.func.value), self._visit_with(node.args[0], (node.func.value.id, True))]
            node.func = ast.Name(id=self.append_helper, ctx=ast.Load())
            return node
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == "draw":
            raise UnsupportedModelError("Use direct draw(...) calls; aliases or reassignment of draw are not supported.")
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if (
            isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name) and node.value.id == "list"
            and node.attr in {"reverse", "sort", "clear", "pop", "remove", "insert", "extend", "append",
                              "__setitem__", "__delitem__", "__iadd__", "__imul__"}
        ):
            return ast.copy_location(ast.Call(
                func=ast.Name(id=self.mutator_helper, ctx=ast.Load()), args=[node], keywords=[]), node)
        return node

    def visit_Subscript(self, node):
        node.value = self._visit_with(node.value)
        node.slice = self._visit_with(node.slice)
        if isinstance(node.ctx, ast.Load):
            index = node.slice
            return ast.copy_location(
                ast.Call(func=ast.Name(id=self.lookup_helper, ctx=ast.Load()), args=[node.value, index], keywords=[]), node
            )
        return node

    def visit_Slice(self, node):
        return ast.copy_location(ast.Call(
            func=ast.Name(id="slice", ctx=ast.Load()),
            args=[self._visit_with(part) if part is not None else ast.Constant(None)
                  for part in (node.lower, node.upper, node.step)], keywords=[]), node)

    def visit_Compare(self, node):
        operands = [node.left, *node.comparators]
        if any(isinstance(child, ast.NamedExpr) for operand in operands for child in ast.walk(operand)):
            raise UnsupportedModelError("Assignment expressions inside model comparisons are not supported.")
        thunks = [ast.Lambda(
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=self._visit_with(operand),
        ) for operand in operands]
        return ast.copy_location(ast.Call(
            func=ast.Name(id=self.compare_helper, ctx=ast.Load()),
            args=[ast.Tuple(elts=[ast.Constant(type(op).__name__) for op in node.ops], ctx=ast.Load()), *thunks],
            keywords=[],
        ), node)

    def visit_FunctionDef(self, node):
        raise UnsupportedModelError("Nested functions inside @model are not supported; keep draws in the model body.")

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        raise UnsupportedModelError("Classes inside @model are not supported.")

    def visit_Yield(self, node):
        raise UnsupportedModelError("A @model function must build a finite graph, not yield a generator.")

    visit_YieldFrom = visit_Yield

    def visit_Global(self, node):
        raise UnsupportedModelError("Model definitions cannot assign global variables.")

    def visit_Nonlocal(self, node):
        raise UnsupportedModelError("Model definitions cannot assign nonlocal variables.")

    def visit_GeneratorExp(self, node):
        if any(_is_declaration(child) for child in ast.walk(node)):
            raise UnsupportedModelError(
                "Generator expressions containing random declarations may execute after model construction. "
                "Use an eager list comprehension instead."
            )
        return self.generic_visit(node)


def model(function):
    """Compile a source-readable function into a callable model factory.

    Direct Normal, HalfNormal, StudentT and Categorical calls in assignment, eager list
    or tuple, comprehension and append positions declare random variables.
    Explicit draw(...) remains available for reusable distribution objects.
    Put definitions in a Python file (or a notebook that preserves source).
    ``exec`` strings and some interactive shells do not expose source to Python.
    """
    if not inspect.isfunction(function) or inspect.iscoroutinefunction(function):
        raise UnsupportedModelError("@model expects an ordinary synchronous Python function.")
    try:
        source_lines, source_start = inspect.getsourcelines(function)
        source = textwrap.dedent("".join(source_lines))
    except (OSError, TypeError) as exc:
        raise UnsupportedModelError(
            "@model needs readable Python source to infer names and symbolic indexing. "
            "Save the function in a .py file; exec strings and some interactive definitions are unsupported."
        ) from exc
    module = ast.parse(source)
    definitions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == function.__name__]
    if len(definitions) != 1:
        raise UnsupportedModelError("Could not identify a single source function for @model.")
    definition = definitions[0]
    all_arguments = [*definition.args.posonlyargs, *definition.args.args, *definition.args.kwonlyargs]
    if any(argument.arg == "draw" for argument in all_arguments):
        raise UnsupportedModelError("The name 'draw' is reserved for random variable declarations, not model inputs.")
    definition.decorator_list = []
    definition.returns = None
    definition.args.defaults = []
    definition.args.kw_defaults = [None for _ in definition.args.kwonlyargs]
    for argument in [*definition.args.posonlyargs, *definition.args.args, *definition.args.kwonlyargs]:
        argument.annotation = None
    if definition.args.vararg:
        definition.args.vararg.annotation = None
    if definition.args.kwarg:
        definition.args.kwarg.annotation = None
    suffix = uuid.uuid4().hex
    draw_helper = f"__smallprob_draw_{suffix}"
    lookup_helper = f"__smallprob_lookup_{suffix}"
    series_helper = f"__smallprob_series_{suffix}"
    append_helper = f"__smallprob_append_{suffix}"
    bind_helper = f"__smallprob_bind_{suffix}"
    mutator_helper = f"__smallprob_mutator_{suffix}"
    compare_helper = f"__smallprob_compare_{suffix}"
    constructor_helper = f"__smallprob_constructor_{suffix}"
    environment = dict(function.__globals__)
    environment.update(inspect.getclosurevars(function).nonlocals)
    append_names = {
        node.func.value.id for node in ast.walk(definition)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append" and isinstance(node.func.value, ast.Name)
        and len(node.args) == 1 and _is_declaration(node.args[0]) and not node.keywords
    }
    transformer = _Transformer(draw_helper, lookup_helper, series_helper, append_helper, bind_helper,
                               mutator_helper, compare_helper, constructor_helper, append_names,
                               _bound_names(definition), environment)
    definition.body = [transformer.visit(statement) for statement in definition.body]
    compiled_module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    ast.increment_lineno(compiled_module, source_start - 1)
    helpers = {draw_helper: draw, lookup_helper: lookup, series_helper: _register_series,
               append_helper: _append_series, bind_helper: _bind_name, compare_helper: _compare_chain,
               constructor_helper: _checked_constructor}
    helpers[mutator_helper] = _guard_list_mutator
    environment.update(helpers)
    exec(compile(compiled_module, inspect.getsourcefile(function) or "<smallprob-model>", "exec"), environment)
    compiled = environment[function.__name__]
    compiled.__defaults__ = function.__defaults__
    compiled.__kwdefaults__ = function.__kwdefaults__

    @functools.wraps(function)
    def factory(*args, **kwargs):
        # Preserve ordinary late-bound global/closure inputs without evaluating
        # defaults or decorators again. Helpers use collision-resistant names.
        environment.update(function.__globals__)
        environment.update(inspect.getclosurevars(function).nonlocals)
        environment.update(helpers)
        if _active_builder.get() is not None:
            raise UnsupportedModelError("Calling one @model from another is not supported in this prototype.")
        from .arrays import infer_backend
        bound = inspect.signature(function).bind(*args, **kwargs)
        bound.apply_defaults()
        selected_backend = infer_backend(bound.arguments)
        inputs = _snapshot(dict(bound.arguments))
        # Model-local deterministic list edits must not change either caller
        # data or the original fixed-input metadata used by equation displays.
        bound.arguments.update(_snapshot(inputs))
        builder = _Builder()
        token = _active_builder.set(builder)
        try:
            compiled(*bound.args, **bound.kwargs)
        finally:
            _active_builder.reset(token)
        from .api import BoundModel
        return BoundModel(builder.graph(), function.__name__, backend=selected_backend,
                          source=source, inputs=inputs)

    factory.source = source
    return factory


def build_graph(function, *args, **kwargs):
    """Build a graph through the same frontend used by ``@model``."""
    return model(function)(*args, **kwargs).graph
