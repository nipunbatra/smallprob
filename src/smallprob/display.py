"""Notebook mathematics and dependency diagrams, with optional Matplotlib.

Labels are presentation metadata. They never rename graph sites or alter a
model. The graph is authoritative; source is used only for a conservative
subset of equations so fixed inputs can keep names such as ``x_i``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from html import escape
from io import BytesIO
from pathlib import Path
import re

import numpy as np

from .core import Expr, ModelError


_GREEK = {"mu", "sigma", "alpha", "beta", "gamma", "theta", "tau", "nu", "pi", "eta"}
_FAMILIES = {"Normal", "HalfNormal", "StudentT", "Categorical"}


def _label(name, labels):
    if name in labels:
        return labels[name].strip("$")
    if name in _GREEK:
        return "\\" + name
    match = re.fullmatch(r"(.+)\[(\d+)\]", name)
    if match:
        return _label(match[1], labels) + "_{" + str(int(match[2]) + 1) + "}"
    if len(name) == 1 and name.isalpha():
        return name
    return r"\mathit{" + name.replace("_", r"\_") + "}"


def _number(value):
    if isinstance(value, (int, float, np.number)):
        return f"{value:g}"
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value)
        if array.ndim == 1:
            return r"\left[" + ", ".join(_number(x) for x in array) + r"\right]"
        if array.ndim == 2:
            return r"\left[" + "; ".join(", ".join(_number(x) for x in row) for row in array) + r"\right]"
    raise ModelError(f"Cannot display fixed value of type {type(value).__name__} as mathematics.")


def _binary(op, left, right):
    if op == "add":
        return left + " + " + right
    if op == "sub":
        return left + " - " + r"\left(" + right + r"\right)"
    if op == "mul":
        return r"\left(" + left + r"\right)\," + r"\left(" + right + r"\right)"
    if op == "div":
        return r"\frac{" + left + "}{" + right + "}"
    if op == "pow":
        return r"\left(" + left + r"\right)^{" + right + "}"
    if op == "index":
        return r"\left(" + left + r"\right)_{" + right + "}"
    raise ModelError(f"Cannot render expression operation {op!r}.")


def _expr(expression, labels):
    if expression.op == "ref":
        return _label(expression.args[0], labels)
    if expression.op == "const":
        raw = expression.args[0]
        if isinstance(raw, (list, tuple)) and any(isinstance(item, Expr) for item in raw):
            return r"\left[" + ", ".join(_expr(item, labels) if isinstance(item, Expr) else _number(item) for item in raw) + r"\right]"
        return _number(raw)
    if expression.op == "neg":
        return r"-\left(" + _expr(expression.args[0], labels) + r"\right)"
    return _binary(expression.op, *(_expr(item, labels) for item in expression.args))


def _ast_expression(node, labels, names, indexed, aliases):
    if isinstance(node, ast.Name):
        if node.id in aliases:
            return aliases[node.id]
        if node.id not in names:
            raise ValueError("Unknown intermediate variable")
        return _label(node.id, labels) + ("_{i}" if node.id in indexed else "")
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return _number(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        result = _ast_expression(node.operand, labels, names, indexed, aliases)
        return ("-" if isinstance(node.op, ast.USub) else "") + r"\left(" + result + r"\right)"
    if isinstance(node, ast.BinOp):
        operation = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.Pow: "pow"}.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported arithmetic")
        left = _ast_expression(node.left, labels, names, indexed, aliases)
        right = _ast_expression(node.right, labels, names, indexed, aliases)
        # Keep the common mathematical linear-regression expression uncluttered.
        if operation == "mul" and isinstance(node.left, (ast.Name, ast.Constant)) and isinstance(node.right, (ast.Name, ast.Constant)):
            return left + r"\," + right
        return _binary(operation, left, right)
    # Arbitrary Python indexing, function calls, branches and local intermediates
    # deliberately fall back to actual graph expressions.
    raise ValueError("Unsupported source expression")


def _distribution(family, parameters):
    if family == "Normal":
        mean, sd = parameters
        return r"\mathcal{N}\!\left(" + mean + r",\;\left(" + sd + r"\right)^2\right)"
    if family == "HalfNormal":
        return r"\mathrm{HalfNormal}\!\left(" + parameters[0] + r"\right)"
    if family == "StudentT":
        df, loc, scale = parameters
        return r"t_{" + df + r"}\!\left(" + loc + r",\;" + scale + r"\right)"
    if family == "Categorical":
        return r"\mathrm{Categorical}\!\left(" + parameters[0] + r"\right)"
    raise ModelError(f"No display notation for distribution {family!r}.")


def _graph_distribution(distribution, labels):
    family = type(distribution).__name__
    fields = {"Normal": ("mean", "sd"), "HalfNormal": ("sd",),
              "StudentT": ("df", "loc", "scale"), "Categorical": ("probs",)}[family]
    return _distribution(family, [_expr(getattr(distribution, field), labels) for field in fields])


def _source_equations(model, labels):
    """Use only unambiguous top-level declarations in the original source."""
    source = getattr(model, "source", None)
    inputs = getattr(model, "inputs", {})
    if not source:
        return {}
    try:
        definition = next(item for item in ast.parse(source).body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)))
    except (SyntaxError, StopIteration):
        return {}
    # Source is a presentation aid, not another evaluator. Unknown calls could
    # mutate or transform an input through an alias; only the small declarative
    # subset is safe to substitute for the authoritative graph equations.
    allowed_calls = _FAMILIES | {"draw", "range"}
    if any(isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in allowed_calls)
           for statement in definition.body for node in ast.walk(statement)):
        return {}
    if any(isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
           for statement in definition.body for node in ast.walk(statement)):
        return {}
    known = set(inputs) | set(model.graph.groups)
    # The source shortcut describes original inputs, so it is inappropriate if
    # a function reassigns or mutates one before constructing a distribution.
    # Likewise, a separate deterministic intermediate needs its own equation;
    # fall back to the actual graph rather than guessing its value or meaning.
    unsafe_inputs = {node.id for node in ast.walk(definition) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in inputs}
    for node in ast.walk(definition):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in inputs:
            unsafe_inputs.add(node.value.id)
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store) and isinstance(node.value, ast.Name) and node.value.id in inputs:
            unsafe_inputs.add(node.value.id)
    known -= unsafe_inputs
    result = {}
    for statement in definition.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            continue
        name = statement.targets[0].id
        if name not in model.graph.groups:
            continue
        value = statement.value
        aliases = {}
        indexed = set()
        size = len(model.graph.groups[name])
        if isinstance(value, ast.ListComp):
            if len(value.generators) != 1 or value.generators[0].ifs or value.generators[0].is_async:
                continue
            generator = value.generators[0]
            if not isinstance(generator.target, ast.Name):
                continue
            if isinstance(generator.iter, ast.Name) and generator.iter.id in inputs and generator.iter.id not in unsafe_inputs:
                aliases[generator.target.id] = _label(generator.iter.id, labels) + "_{i}"
            elif not (isinstance(generator.iter, ast.Call) and isinstance(generator.iter.func, ast.Name) and generator.iter.func.id == "range"):
                continue
            value = value.elt
        elif name in model.graph.series:
            for input_name, raw in inputs.items():
                shape = np.shape(raw)
                if len(shape) == 1 and shape[0] == size:
                    indexed.add(input_name)
                elif len(shape) == 1 and shape[0] == 1:
                    aliases[input_name] = _label(input_name, labels) + "_{1}"
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "draw" and len(value.args) == 1:
            value = value.args[0]
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name) or value.func.id not in _FAMILIES:
            continue
        family = value.func.id
        fields, defaults = {
            "Normal": (("mean", "sd"), (0, 1)),
            "HalfNormal": (("sd",), (1,)),
            "StudentT": (("df", "loc", "scale"), (None, 0, 1)),
            "Categorical": (("probs",), (None,)),
        }[family]
        parameters = dict(zip(fields, (ast.Constant(default) for default in defaults)))
        parameters.update(zip(fields, value.args))
        if any(keyword.arg not in fields for keyword in value.keywords):
            continue
        parameters.update((keyword.arg, keyword.value) for keyword in value.keywords)
        try:
            result[name] = _distribution(family, [_ast_expression(parameters[field], labels, known, indexed, aliases) for field in fields])
        except (ValueError, ModelError):
            pass
    return result


@dataclass(frozen=True)
class DisplayNode:
    key: str
    label: str
    family: str
    equation: str
    sites: tuple[str, ...]
    parents: tuple[str, ...]
    size: int | None
    observed: bool


def _specification(model, labels, observed):
    labels = {} if labels is None else dict(labels)
    available = set(model.graph.groups) | {node.name for node in model.graph.nodes} | set(getattr(model, "inputs", {}))
    if set(labels) - available:
        raise ModelError(f"Unknown display labels: {sorted(set(labels) - available)}.")
    if any(not isinstance(value, str) or not value.strip("$ ") for value in labels.values()):
        raise ModelError("Display labels must be nonempty strings containing plain names or TeX.")
    if observed is None:
        observed = set()
    elif isinstance(observed, str):
        observed = {observed}
    else:
        observed = set(observed)
    site_names = {node.name for node in model.graph.nodes}
    if observed - (site_names | set(model.graph.groups)):
        raise ModelError(f"Unknown observed variables: {sorted(observed - (site_names | set(model.graph.groups)))}.")
    observed_sites = set()
    for name in observed:
        observed_sites.update(model.graph.groups.get(name, [name]))
    source = _source_equations(model, labels)
    by_name = {node.name: node for node in model.graph.nodes}
    all_dependencies = {node.name: node.distribution.dependencies() for node in model.graph.nodes}
    specifications = []
    for group, names in model.graph.groups.items():
        if not names:
            continue
        members = [by_name[name] for name in names]
        dependencies = [node.distribution.dependencies() for node in members]
        family = type(members[0].distribution).__name__
        same_family = all(type(node.distribution).__name__ == family for node in members)
        same_parents = all(deps == dependencies[0] for deps in dependencies)
        inside_edges = any(set(names) & deps for deps in dependencies)
        same_shading = all((name in observed_sites) == (names[0] in observed_sites) for name in names)
        graph_equations = [_graph_distribution(node.distribution, labels) for node in members]
        can_group = group in model.graph.series and same_family and same_parents and not inside_edges and same_shading
        can_group = can_group and (group in source or len(set(graph_equations)) == 1)
        # A single i-indexed node cannot stand in for distinct parents of
        # individually displayed children: eta_i -> y_1 would hide which eta
        # is used. Keep such parent sites explicit, preserving each edge.
        member_names = set(names)
        distinguished_by_child = any(
            bool(deps & member_names) and deps & member_names != member_names
            for child, deps in all_dependencies.items() if child not in member_names
        )
        can_group = can_group and not distinguished_by_child
        if can_group:
            label = _label(group, labels) + "_{i}"
            equation = label + r"\sim " + source.get(group, graph_equations[0])
            specifications.append(DisplayNode(group, label, family, equation, tuple(names), tuple(sorted(dependencies[0])), len(names), names[0] in observed_sites))
        else:
            for node, graph_equation in zip(members, graph_equations):
                label = _label(node.name, labels)
                equation = label + r"\sim " + (source.get(group, graph_equation) if group not in model.graph.series else graph_equation)
                specifications.append(DisplayNode(node.name, label, type(node.distribution).__name__, equation, (node.name,), tuple(sorted(node.distribution.dependencies())), None, node.name in observed_sites))
    mapping = {site: node.key for node in specifications for site in node.sites}
    edges = sorted({(mapping[parent], node.key) for node in specifications for parent in node.parents})
    return specifications, edges, labels


def _hierarchical_layout(model, nodes):
    """Recognise aligned local-parent/observation pairs without changing edges.

    Every local variable has its own child. Shared roots may also parent each
    child. This only sets drawing positions; equations and sites stay explicit.
    """
    by_key = {node.key: node for node in nodes}
    series = [tuple(names) for group, names in model.graph.groups.items()
              if group in model.graph.series and len(names) > 1]
    if len(series) != 2:
        return None
    for local, children in (series, series[::-1]):
        if len(local) != len(children) or not 2 <= len(local) <= 12:
            continue
        if any(name not in by_key for name in local + children):
            continue
        if any(by_key[name].parents for name in local):
            continue
        shared = set(by_key[children[0]].parents) - {local[0]}
        if not shared or any(name not in by_key or by_key[name].parents or by_key[name].size is not None for name in shared):
            continue
        if any(set(by_key[child].parents) != shared | {parent} for parent, child in zip(local, children)):
            continue
        if set(by_key) != shared | set(local) | set(children):
            continue
        return tuple(node.key for node in nodes if node.key in shared), local, children
    return None


def _factorization(nodes, labels):
    factors = []
    for node in nodes:
        parent_labels = [_label(parent, labels) for parent in node.parents]
        term = r"p\!\left(" + node.label
        if parent_labels:
            term += r"\mid " + ",".join(parent_labels)
        term += r"\right)"
        if node.size is not None:
            term = r"\prod_{i=1}^{" + str(node.size) + "}" + term
        factors.append(term)
    return r"p(\mathrm{all\ random\ variables}\mid\mathrm{fixed\ inputs}) = " + r"\,".join(factors)


@dataclass(frozen=True)
class HMMDisplay:
    states: tuple[str, ...]
    emissions: tuple[str, ...]
    equations: tuple[str, ...]
    category_count: int
    state_labels: tuple[str, ...] | None
    factorization: str

    @property
    def shown_indices(self):
        """Time positions drawn explicitly; the intervening chain is elided."""
        count = len(self.states)
        return tuple(range(count)) if count <= 6 else (0, 1, 2, count - 1)


def _hmm_display(model, labels):
    """Recognise a fixed-parameter categorical chain from the actual graph.

    No source pattern is assumed: every transition, emission and fixed table
    must match. Unrecognised or time-varying models keep their full equations.
    State names are presentation metadata; they never change category indices.
    """
    graph = model.graph
    if any(key in labels for key in {node.name for node in graph.nodes} - set(graph.groups)):
        return None
    if len(graph.groups) != 2:
        return None
    by_name = {node.name: node for node in graph.nodes}
    groups = [(group, names) for group, names in graph.groups.items() if names]
    if len(groups) != 2:
        return None
    state_options = [(group, names) for group, names in groups if all(type(by_name[name].distribution).__name__ == "Categorical" for name in names)]
    emission_options = [(group, names) for group, names in groups if all(type(by_name[name].distribution).__name__ == "Normal" for name in names)]
    if len(state_options) != 1 or len(emission_options) != 1:
        return None
    state_group, states = state_options[0]
    emission_group, emissions = emission_options[0]
    count = len(states)
    if count < 2 or count != len(emissions):
        return None

    def fixed_lookup(expression, parent):
        if expression.op != "index":
            raise ValueError("not an indexed parameter")
        base, index = expression.args
        if base.dependencies() or index.op != "ref" or index.args != (parent,):
            raise ValueError("different lookup dependency")
        return np.asarray(base.eval({}), dtype=float)

    def emission_scale(expression, parent):
        if expression.dependencies():
            return fixed_lookup(expression, parent)
        return np.asarray(expression.eval({}), dtype=float)

    try:
        initial_expr = by_name[states[0]].distribution.probs
        if initial_expr.dependencies():
            return None
        initial = np.asarray(initial_expr.eval({}), dtype=float)
        transition = fixed_lookup(by_name[states[1]].distribution.probs, states[0])
        means = fixed_lookup(by_name[emissions[0]].distribution.mean, states[0])
        scales = emission_scale(by_name[emissions[0]].distribution.sd, states[0])
        categories = len(initial)
        if initial.shape != (categories,) or transition.shape != (categories, categories) or means.shape != (categories,) or categories < 1:
            return None
        if scales.shape not in ((), (categories,)):
            return None
        for t, state in enumerate(states[1:], start=1):
            if not np.array_equal(fixed_lookup(by_name[state].distribution.probs, states[t - 1]), transition):
                return None
        for t, emission in enumerate(emissions):
            distribution = by_name[emission].distribution
            if not np.array_equal(fixed_lookup(distribution.mean, states[t]), means):
                return None
            if not np.array_equal(emission_scale(distribution.sd, states[t]), scales):
                return None
    except (ValueError, TypeError, ModelError):
        return None
    z, y = _label(state_group, labels), _label(emission_group, labels)
    sigma = r"\sigma_{" + z + r"_t}" if scales.ndim else r"\sigma"
    equations = [
        z + r"_1\sim \mathrm{Categorical}(\pi)",
        z + r"_t\mid " + z + r"_{t-1}\sim \mathrm{Categorical}(A_{" + z + r"_{t-1},:})",
        y + r"_t\mid " + z + r"_t\sim \mathcal{N}(\mu_{" + z + r"_t},\;" + sigma + r"^2)",
        r"\pi=" + _number(initial),
    ]
    equations.extend("A_{" + str(index) + ",:}=" + _number(row) for index, row in enumerate(transition))
    equations.append(r"\mu=" + _number(means))
    equations.append(r"\sigma=" + _number(scales.item() if scales.ndim == 0 else scales))
    state_labels = getattr(model, "hmm_state_labels", None)
    if state_labels is not None:
        if len(state_labels) != categories or any(not isinstance(value, str) for value in state_labels):
            state_labels = None
        else:
            state_labels = tuple(state_labels)
    horizon = str(count)
    factorization = (r"p(" + z + "_{1:" + horizon + "}," + y + "_{1:" + horizon + "}) = p(" + z + r"_1)\," +
                     r"\prod_{t=2}^{" + horizon + "}p(" + z + r"_t\mid " + z + r"_{t-1})\," +
                     r"\prod_{t=1}^{" + horizon + "}p(" + y + r"_t\mid " + z + "_t)")
    return HMMDisplay(tuple(states), tuple(emissions), tuple(equations), categories, state_labels, factorization)


class ModelMath:
    """Rich LaTeX equations. Display with the final expression in a notebook."""

    def __init__(self, model, *, labels=None):
        self.model = model
        self.nodes, self.edges, self.labels = _specification(model, labels, None)
        self.equations = tuple(node.equation + (r",\quad i=1,\ldots," + str(node.size) if node.size is not None else "") for node in self.nodes)
        hmm = _hmm_display(model, self.labels)
        if hmm is not None:
            self.equations = hmm.equations + (r"t=1,\ldots," + str(len(hmm.states)) + r";\quad \mathrm{transition:}\ t\geq 2;\quad \mathrm{states:}\ 0,\ldots," + str(hmm.category_count - 1),)
        self.factorization = hmm.factorization if hmm is not None else _factorization(self.nodes, self.labels)

    @property
    def latex(self):
        equations = " \\\\ \n".join(self.equations)
        return r"\begin{gathered}" + equations + r"\end{gathered}"

    def _repr_latex_(self):
        return "$$" + self.latex + "$$"

    def __str__(self):
        return "\n".join(self.equations) + "\n" + self.factorization

    def __repr__(self):
        return f"ModelMath({self.model.name!r}, equations={len(self.equations)})"


class ModelDiagram:
    """A model diagram with equations, optional observed shading and vector glyphs.

    ``observed`` controls presentation only; it never conditions a model.
    SVG and PNG export require the optional ``smallprob[viz]`` extra.
    """

    def __init__(self, model, *, labels=None, observed=None):
        self.model = model
        self.nodes, self.edges, self.labels = _specification(model, labels, observed)
        self.hmm = _hmm_display(model, self.labels)
        self.hierarchy = _hierarchical_layout(model, self.nodes)
        self.factorization = self.hmm.factorization if self.hmm is not None else _factorization(self.nodes, self.labels)

    def _figure(self):
        try:
            import matplotlib
            from matplotlib.figure import Figure
            from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
        except ImportError as exc:
            raise ModelError("Diagram rendering needs Matplotlib: install 'smallprob[viz]'.") from exc
        if self.hmm is None and len(self.nodes) > 30:
            raise ModelError("This expanded graph has more than 30 nodes. Use math() for all site equations; compact diagrams support simple independent series.")
        palette = {"bg": "#fbfaf6", "ink": "#153e42", "muted": "#6c8182", "line": "#9ab3b1", "observed": "#d8ebe4", "plate": "#edf3ed", "accent": "#bf7959"}
        with matplotlib.rc_context({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "svg.fonttype": "path", "text.usetex": False}):
            if self.hmm:
                height = max(5.8, 2.8 + .4 * len(self.hmm.equations))
            elif self.hierarchy:
                height = max(5.8, 2.5 + .36 * len(self.nodes))
            else:
                height = max(5.2, 2.5 + .55 * len(self.nodes))
            figure = Figure(figsize=(12.8, height), facecolor=palette["bg"], layout=None)
            figure.text(.045, .925, self.model.name.replace("_", " "), fontsize=22, color=palette["ink"], weight="bold")
            figure.text(.046, .873, "GENERATION  /  CONDITIONAL DEPENDENCIES", fontsize=8.6, color=palette["muted"])
            graph_ax = figure.add_axes((.035, .19, .42, .63), facecolor=palette["bg"])
            graph_ax.set_xlim(0, 1); graph_ax.set_ylim(0, 1); graph_ax.axis("off")
            graph_ax.set_aspect("equal", adjustable="box")
            equation_ax = figure.add_axes((.50, .20, .47, .63), facecolor=palette["bg"])
            equation_ax.set_xlim(0, 1); equation_ax.set_ylim(0, 1); equation_ax.axis("off")
            levels = {}
            for node in self.nodes:
                parents = [left for left, right in self.edges if right == node.key]
                levels[node.key] = max((levels[parent] + 1 for parent in parents), default=0)
            max_level = max(levels.values(), default=0)
            positions = {}
            for level in range(max_level + 1):
                row = [node for node in self.nodes if levels[node.key] == level]
                for index, node in enumerate(row):
                    positions[node.key] = ((index + 1) / (len(row) + 1), .77 - .52 * level / max(max_level, 1))
            radius = min(.085, .33 / max((sum(value == level for value in levels.values()) for level in set(levels.values())), default=1))
            drawn_nodes = self.nodes
            drawn_edges = self.edges
            if self.hierarchy is not None:
                shared, local, children = self.hierarchy
                radius = min(.070, .38 / len(local))
                for index, name in enumerate(shared):
                    positions[name] = ((index + 1) / (len(shared) + 1), .87)
                for index, (parent, child) in enumerate(zip(local, children)):
                    x = .075 + .85 * index / (len(local) - 1)
                    positions[parent], positions[child] = (x, .52), (x, .17)
                graph_ax.text(.5, .99, "shared population parameters", fontsize=8.5, ha="center", color=palette["muted"])
                graph_ax.text(.5, .02, "each local effect has its own observation", fontsize=8.5, ha="center", color=palette["muted"])
            if self.hmm is not None:
                count = len(self.hmm.states)
                indices = self.hmm.shown_indices
                compressed = len(indices) < count
                columns = len(indices) + int(compressed)
                radius = min(.075, .36 / columns)
                shown = set()
                for column, index in enumerate(indices):
                    if compressed and column == len(indices) - 1:
                        column += 1
                    state, emission = self.hmm.states[index], self.hmm.emissions[index]
                    x = .10 + .80 * column / (columns - 1)
                    positions[state], positions[emission] = (x, .65), (x, .28)
                    shown.update((state, emission))
                drawn_nodes = [node for node in self.nodes if node.key in shown]
                drawn_edges = [(left, right) for left, right in self.edges if left in shown and right in shown]
                graph_ax.text(.5, .85, "hidden state sequence", fontsize=9, ha="center", color=palette["muted"])
                if compressed:
                    gap_x = .10 + .80 * (columns - 2) / (columns - 1)
                    for y in (.65, .28):
                        graph_ax.text(gap_x, y, "⋯", fontsize=24, ha="center", va="center", color=palette["muted"])
                    # The dashed continuations terminate at an ellipsis, not at
                    # a nonadjacent state: the omitted chain is not a new edge.
                    left_x = positions[self.hmm.states[indices[-2]]][0]
                    right_x = positions[self.hmm.states[indices[-1]]][0]
                    for start_x, end_x in ((left_x + radius * 1.12, gap_x - .055), (gap_x + .055, right_x - radius * 1.17)):
                        graph_ax.add_patch(FancyArrowPatch((start_x, .65), (end_x, .65), arrowstyle="-|>", mutation_scale=11, linestyle=(0, (3, 3)), linewidth=1.25, color=palette["line"], zorder=1))
                    graph_ax.text(.5, .10, f"{count - len(indices)} intervening time steps omitted", fontsize=8.5, ha="center", color=palette["muted"])
                else:
                    graph_ax.text(.5, .10, "one emission per time step", fontsize=9, ha="center", color=palette["muted"])
            for left, right in drawn_edges:
                start, end = np.asarray(positions[left]), np.asarray(positions[right])
                unit = (end - start) / np.linalg.norm(end - start)
                graph_ax.add_patch(FancyArrowPatch(start + radius * 1.12 * unit, end - radius * 1.17 * unit, arrowstyle="-|>", mutation_scale=12, linewidth=1.25, color=palette["line"], zorder=1))
            for node in drawn_nodes:
                x, y = positions[node.key]
                if node.size is not None:
                    graph_ax.add_patch(FancyBboxPatch((x - radius - .045, y - radius - .09), 2 * radius + .09, 2 * radius + .17, boxstyle="round,pad=0.018,rounding_size=0.025", linewidth=1, edgecolor=palette["line"], facecolor=palette["plate"], zorder=0))
                    graph_ax.text(x, y - radius - .065, f"$i=1,\\ldots,{node.size}$", color=palette["muted"], fontsize=9, ha="center", va="center")
                graph_ax.add_patch(Circle((x, y), radius, facecolor=palette["observed"] if node.observed else palette["bg"], edgecolor=palette["ink"], linewidth=1.5, zorder=3))
                node_font = min(17, 17 * radius / .065) if self.hierarchy else (17 if len(node.label) < 20 else 12)
                graph_ax.text(x, y + radius * .22, "$" + node.label + "$", fontsize=node_font, color=palette["ink"], ha="center", va="center", zorder=4)
                self._glyph(graph_ax, node.family, x, y - radius * .48, radius * .72, radius * .40, palette["accent"], self.hmm.category_count if self.hmm else None)
            if self.hmm is not None:
                spacing = min(.119, .87 / len(self.hmm.equations))
                for index, equation in enumerate(self.hmm.equations):
                    y = .97 - index * spacing
                    equation_ax.text(0, y, "$" + equation + "$", fontsize=13 if len(equation) < 110 else 10, color=palette["ink"], va="top")
                equation_ax.text(0, .025, f"t = 1, …, {len(self.hmm.states)}; transitions start at t = 2. States: 0, …, {self.hmm.category_count - 1}.", fontsize=8.2, color=palette["muted"], va="top")
            else:
                count = max(1, len(self.nodes))
                spacing = min(.19, .9 / count)
                for index, node in enumerate(self.nodes):
                    y = .94 - index * spacing
                    equation_ax.text(0, y, "$" + node.equation + "$", fontsize=11 if self.hierarchy else (14 if len(node.equation) < 180 else 11), color=palette["ink"], va="top")
                    if node.size is not None:
                        equation_ax.text(.015, y - spacing * .40, f"{node.size} conditionally independent observations" if node.observed else f"{node.size} conditionally independent variables", fontsize=8.5, color=palette["muted"], va="top")
            if self.hmm is not None and self.hmm.state_labels:
                mapping = "   ·   ".join(f"{index} = {name}" for index, name in enumerate(self.hmm.state_labels))
                figure.text(.048, .153, "STATE NAMES   " + mapping, fontsize=8.6, color=palette["ink"], parse_math=False)
            figure.text(.048, .118, "○  latent variable       ●  marked observed       rounded box  =  repetition", fontsize=8.3, color=palette["muted"])
            figure.text(.048, .079, "Normal: second mathematical parameter is variance.  Student t and HalfNormal: scale parameter.", fontsize=8.3, color=palette["muted"])
            figure.text(.048, .044, "Distribution glyphs show family shapes, not fitted densities.  Observed shading is a display choice.", fontsize=8.1, color=palette["muted"])
            return figure

    @staticmethod
    def _glyph(ax, family, x, y, width, height, color, categories=None):
        if family == "Categorical":
            count = 3 if categories is None else categories
            ax.bar(x + np.linspace(-.7, .7, count) * width, np.resize([.4, 1, .65, .8], count) * height, width=width * min(.3, 1.2 / count), bottom=y, color=color, linewidth=0, zorder=5)
            return
        grid = np.linspace(-3.5, 3.5, 61)
        if family == "HalfNormal":
            grid = np.linspace(0, 3.5, 61)
            density = np.exp(-.5 * grid ** 2)
            horizontal = x - width + 2 * width * grid / 3.5
        else:
            density = (1 + grid ** 2 / 3) ** -2 if family == "StudentT" else np.exp(-.5 * grid ** 2)
            horizontal = x + width * grid / 3.5
        ax.plot(horizontal, y + height * density, color=color, linewidth=1.4, zorder=5)

    def _bytes(self, format):
        figure = self._figure()
        buffer = BytesIO()
        try:
            figure.savefig(buffer, format=format, dpi=180, facecolor=figure.get_facecolor())
        except ValueError as exc:
            raise ModelError(f"Cannot render mathematical labels: {exc}") from exc
        finally:
            figure.clear()
        return buffer.getvalue()

    def _repr_svg_(self):
        svg = self._bytes("svg").decode("utf-8")
        # Retain a searchable accessible summary even when math glyphs are paths.
        equations = self.hmm.equations if self.hmm else (node.equation for node in self.nodes)
        summary = self.model.name + ": " + "; ".join(equations)
        if self.hmm:
            summary += f"; {len(self.hmm.states)} time steps."
            if self.hmm.state_labels:
                summary += "; " + ", ".join(f"state {index} = {name}" for index, name in enumerate(self.hmm.state_labels))
        summary = escape(summary)
        index = svg.index(">", svg.index("<svg")) + 1
        return svg[:index] + "<title>" + summary + "</title>" + svg[index:]

    def save(self, path):
        path = Path(path)
        format = path.suffix.lower().lstrip(".")
        if format not in {"svg", "png"}:
            raise ModelError("Diagram export supports .svg and .png files.")
        path.parent.mkdir(parents=True, exist_ok=True)
        if format == "svg":
            path.write_text(self._repr_svg_(), encoding="utf-8")
        else:
            path.write_bytes(self._bytes(format))
        return path

    def __repr__(self):
        return f"ModelDiagram({self.model.name!r}, displayed_nodes={len(self.nodes)})"
