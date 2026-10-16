"""
The shot library: reusable, animated diagram primitives the explainer films compose from.

The first half is the original pattern-explainer set (ArrayRow, pointer, BarSet, TreeViz,
graph_viz, DPGrid, ...). The second half adds deep-learning primitives:

  TensorGrid      2D/3D grid of numbered cells + shape/stride labels, row/col/element focus
  MatMul          row of A times column of B -> one cell of C, with a dot-product readout
  Heatmap         attention-weight matrix with a colour scale and an optional causal mask
  BlockDiagram    boxes + arrows for transformer blocks / data flow, shape labels on edges
  layer_stack     the common case of BlockDiagram: a vertical stack of layers
  CurvePlot       axes + curve (loss curve, softmax, activation) + a tracker-driven dot
  CompGraph       autograd computation graph: value nodes (data/grad) and op nodes

Every shot returns a plain manim mobject (or a small holder with named parts) so scene code
reads like a shot list rather than geometry. Each is drawn to the house palette and stays
inside the safe box; callers place the result, they don't rebuild it.

Design rules kept here so scenes can't get them wrong:
  * bars share a baseline (GrowFromEdge(DOWN)), never FadeIn;
  * labels sit BESIDE a shape (next_to + buff), never move_to(shape.get_center());
  * highlight in place (`Indicate` / `.animate.set_color`), never a duplicate overlay;
  * text inside a box is fit to the box before it's centred.
"""

import math

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Arrow,
    Axes,
    CurvedArrow,
    Rectangle,
    ShowPassingFlash,
    ValueTracker,
    always_redraw,
    interpolate_color,
    Circle,
    DashedLine,
    Dot,
    GrowArrow,
    GrowFromEdge,
    Line,
    RoundedRectangle,
    Text,
    VGroup,
    VMobject,
)

from house import (
    ACCENT,
    BODY_FS,
    GONE,
    GOOD,
    INK,
    MONO,
    MONO_FS,
    MUTED,
    PANEL,
    PRIMARY,
    SMALL_FS,
    STROKE,
    WINDOW,
    card,
)

CELL = 0.62
CELL_BUFF = 0.07


def _txt(value, color=INK, fs=MONO_FS, mono=True) -> Text:
    return Text(str(value), font=MONO if mono else "Helvetica Neue", font_size=fs, color=color)


def _fit_inside(t: Text, box, pad: float = 0.12) -> Text:
    """Downscale-only fit of text into a box, then centre it (guide rule)."""
    if t.width > box.width - pad:
        t.scale_to_fit_width(box.width - pad)
    if t.height > box.height - pad:
        t.scale_to_fit_height(box.height - pad)
    t.move_to(box.get_center())
    return t


# --------------------------------------------------------------------------------------
# ArrayRow — the workhorse: a row of cells + an index rail + pointer/window/mark helpers.
# --------------------------------------------------------------------------------------
class ArrayRow:
    """A row of value cells with an index rail.

    `values` may contain None for a deliberately empty slot. Cells are addressable by index
    so a scene can animate one cell without rebuilding the row.
    """

    def __init__(
        self,
        values,
        *,
        cell: float = CELL,
        buff: float = CELL_BUFF,
        color=STROKE,
        fill=PANEL,
        fs: int = MONO_FS,
        show_index: bool = True,
        index_fs: int = SMALL_FS,
        max_width: float = 11.6,
    ):
        self.values = list(values)
        self.cell_size = cell
        self.cells = VGroup()
        self.boxes = []
        self.texts = []
        for v in self.values:
            box = card(cell, cell, color=color, fill=fill)
            self.boxes.append(box)
            if v is None:
                self.cells.add(VGroup(box))
                self.texts.append(None)
            else:
                t = _txt(v, fs=fs)
                _fit_inside(t, box)
                self.texts.append(t)
                self.cells.add(VGroup(box, t))
        self.cells.arrange(RIGHT, buff=buff)
        if self.cells.width > max_width:
            self.cells.scale_to_fit_width(max_width)
            self.cell_size = cell * (max_width / (cell * len(values) + buff * (len(values) - 1)))

        self.idx = VGroup()
        if show_index:
            for i in range(len(self.values)):
                lb = Text(str(i), font=MONO, font_size=index_fs, color=MUTED)
                lb.next_to(self.cells[i], DOWN, buff=0.14)
                self.idx.add(lb)
        self.group = VGroup(self.cells, self.idx)

    # -- geometry ------------------------------------------------------------------
    def cell(self, i: int):
        return self.cells[i]

    def box(self, i: int) -> VMobject:
        return self.boxes[i]

    def center_of(self, i: int):
        return self.cells[i].get_center()

    def top_of(self, i: int):
        return self.cells[i].get_top()

    def bottom_of(self, i: int):
        return self.cells[i].get_bottom()

    def set_value(self, i: int, value, color=INK):
        """Replace cell i's text in place — used when values change during a beat.

        Note: manim's `Mobject.replace(mobject, dim_to_match)` takes the thing to copy
        geometry FROM, so swapping a child by calling `cell.replace(old, new)` passes the new
        Text as `dim_to_match` and raises TypeError. Remove-then-add is the correct swap, and
        it also keeps the cell's draw order stable.
        """
        box = self.boxes[i]
        new = _txt(value, color=color)
        _fit_inside(new, box)
        old = self.texts[i]
        self.texts[i] = new
        if old is not None:
            self.cells[i].remove(old)
        new.move_to(box.get_center())
        self.cells[i].add(new)
        return new

    def index_labels(self, i: int) -> Text:
        return self.idx[i]

    # -- emphasis (in place, never a duplicate overlay) -----------------------------
    def focus(self, i: int, color=ACCENT, fill_opacity: float = 0.30):
        return [
            self.boxes[i].animate.set_stroke(color=color, width=2.6).set_fill(
                color, opacity=fill_opacity
            )
        ]

    def mark_gone(self, i: int, opacity: float = 0.16):
        """Eliminated candidate: dimmed, red-stroked, still legible."""
        anims = [
            self.boxes[i].animate.set_stroke(color=GONE, width=1.8).set_fill(
                GONE, opacity=opacity
            )
        ]
        if self.texts[i] is not None:
            anims.append(self.texts[i].animate.set_color(MUTED))
        return anims

    def mark_good(self, i: int, opacity: float = 0.32):
        return [
            self.boxes[i].animate.set_stroke(color=GOOD, width=2.6).set_fill(
                GOOD, opacity=opacity
            )
        ]

    def reset(self, i: int, color=STROKE, fill=PANEL):
        anims = [self.boxes[i].animate.set_stroke(color=color, width=1.6).set_fill(fill, opacity=1.0)]
        if self.texts[i] is not None:
            anims.append(self.texts[i].animate.set_color(INK))
        return anims

    def reset_all(self, color=STROKE, fill=PANEL):
        anims = []
        for i in range(len(self.values)):
            anims += self.reset(i, color=color, fill=fill)
        return anims


def pointer(row: ArrayRow, i: int, text: str, *, color=ACCENT, above: bool = True,
            fs: int = BODY_FS) -> VGroup:
    """A labelled pointer chevron aimed at cell i — the thing the narration just named."""
    cell = row.cell(i)
    if above:
        tip = cell.get_top()
        arrow = Arrow(
            tip + UP * 0.62, tip + UP * 0.1, buff=0, color=color, stroke_width=3.4,
            max_tip_length_to_length_ratio=0.28,
        )
        lb = Text(text, font=MONO, font_size=fs, color=color)
        lb.next_to(arrow, UP, buff=0.1)
    else:
        tip = cell.get_bottom()
        arrow = Arrow(
            tip + DOWN * 0.62, tip + DOWN * 0.1, buff=0, color=color, stroke_width=3.4,
            max_tip_length_to_length_ratio=0.28,
        )
        lb = Text(text, font=MONO, font_size=fs, color=color)
        lb.next_to(arrow, DOWN, buff=0.1)
    return VGroup(arrow, lb)


def window_over(row: ArrayRow, lo: int, hi: int, *, color=WINDOW, opacity: float = 0.22,
                label: str | None = None) -> VGroup:
    """A translucent span covering cells lo..hi (inclusive) — the active window/subarray."""
    left = row.cell(lo).get_left()[0] - 0.05
    right = row.cell(hi).get_right()[0] + 0.05
    top = max(row.cell(lo).get_top()[1], row.cell(hi).get_top()[1]) + 0.09
    bottom = min(row.cell(lo).get_bottom()[1], row.cell(hi).get_bottom()[1]) - 0.09
    rect = RoundedRectangle(
        corner_radius=0.13,
        width=right - left,
        height=top - bottom,
        stroke_color=color,
        stroke_width=2.2,
        fill_color=color,
        fill_opacity=opacity,
    )
    rect.move_to([(left + right) / 2, (top + bottom) / 2, 0])
    parts = VGroup(rect)
    if label:
        lb = Text(label, font=MONO, font_size=SMALL_FS, color=color)
        lb.next_to(rect, UP, buff=0.14)
        parts.add(lb)
    return parts


def target_band(value_text, *, color=INK, fs: int = 30, y: float = 2.55) -> VGroup:
    """The 'we want this' readout that sits under the headline for the whole film."""
    t = Text(value_text, font=MONO, font_size=fs, color=color)
    t.move_to([0, y, 0])
    return VGroup(t)


# --------------------------------------------------------------------------------------
# Bars — bars share a baseline; label BELOW after arranging so the baseline stays true.
# --------------------------------------------------------------------------------------
class BarSet:
    def __init__(self, values, *, bar_w: float = 0.52, buff: float = 0.16,
                 height: float = 2.6, color=PRIMARY, label_fs: int = SMALL_FS,
                 show_index: bool = True, max_width: float = 11.4):
        self.values = list(values)
        peak = max(self.values) if any(self.values) else 1
        self.bars = VGroup()
        self.bar_w = bar_w
        for v in self.values:
            h = max(0.06, height * (v / peak if peak else 0))
            bar = RoundedRectangle(
                corner_radius=0.05, width=bar_w, height=h,
                stroke_color=color, stroke_width=1.4,
                fill_color=color, fill_opacity=0.55,
            )
            bar._target_h = h
            self.bars.add(bar)
        self.bars.arrange(RIGHT, buff=buff, aligned_edge=DOWN)
        if self.bars.width > max_width:
            self.bars.scale_to_fit_width(max_width)
        self.baseline = Line(
            self.bars.get_left() + LEFT * 0.12,
            self.bars.get_right() + RIGHT * 0.12,
            color=STROKE, stroke_width=2,
        ).align_to(self.bars, DOWN)
        self.labels = VGroup()
        if show_index:
            for i, bar in enumerate(self.bars):
                lb = Text(str(i), font=MONO, font_size=label_fs, color=MUTED)
                lb.next_to(bar, DOWN, buff=0.14)
                lb.align_to(bar, DOWN)
                self.labels.add(lb)
        self.group = VGroup(self.baseline, self.bars, self.labels)

    def grow(self):
        return [GrowFromEdge(b, DOWN) for b in self.bars]

    def bar(self, i: int):
        return self.bars[i]


# --------------------------------------------------------------------------------------
# LinkedList — nodes chained by arrows; `break_at` shows a reversal in progress.
# --------------------------------------------------------------------------------------
def node_chain(labels, *, node: float = 0.72, buff: float = 0.62, color=PRIMARY,
               fs: int = MONO_FS) -> VGroup:
    nodes = VGroup()
    for lb in labels:
        c = Circle(radius=node / 2, stroke_color=color, stroke_width=2, fill_color=PANEL,
                   fill_opacity=1.0)
        t = _txt(lb, fs=fs)
        _fit_inside(t, c)
        nodes.add(VGroup(c, t))
    nodes.arrange(RIGHT, buff=buff)
    arrows = VGroup()
    for i in range(len(labels) - 1):
        a = Arrow(
            nodes[i].get_right() + RIGHT * 0.04,
            nodes[i + 1].get_left() + LEFT * 0.04,
            buff=0, color=MUTED, stroke_width=2.6, max_tip_length_to_length_ratio=0.3,
        )
        arrows.add(a)
    return VGroup(nodes, arrows)


def arrow_between(a, b, *, color=ACCENT, curved: float = 0.0, stroke: float = 3.0) -> Arrow:
    kwargs = dict(buff=0.06, color=color, stroke_width=stroke,
                  max_tip_length_to_length_ratio=0.3)
    if curved:
        kwargs["path_arc"] = curved
    return Arrow(a.get_center(), b.get_center(), **kwargs)


# --------------------------------------------------------------------------------------
# Hash buckets — key → bucket list, for hash-map pattern.
# --------------------------------------------------------------------------------------
def hash_buckets(keys, *, bucket_w: float = 1.5, row_h: float = 0.52,
                 buff: float = 0.16) -> VGroup:
    rows = VGroup()
    for k in keys:
        box = card(bucket_w, row_h, color=STROKE, fill=PANEL)
        t = _txt(k, fs=SMALL_FS)
        _fit_inside(t, box, pad=0.1)
        rows.add(VGroup(box, t))
    rows.arrange(DOWN, buff=buff, aligned_edge=LEFT)
    return rows


# --------------------------------------------------------------------------------------
# Tree — a perfect binary tree layout with straight edges; returns holder with node list.
# --------------------------------------------------------------------------------------
class TreeViz:
    def __init__(self, levels: int, *, radius: float = 0.30, v_gap: float = 1.15,
                 label_fs: int = SMALL_FS):
        self.positions = []
        self.nodes = VGroup()
        self.edges = VGroup()
        n = 2 ** levels - 1
        # x positions: level L has 2^L nodes spread evenly
        for L in range(levels):
            count = 2 ** L
            span = 8.4
            row = []
            for i in range(count):
                x = -span / 2 + span * (i + 0.5) / count
                y = -(L * v_gap)
                row.append([x, y, 0])
            self.positions.append(row)
        for L, row in enumerate(self.positions):
            for x, y, _ in row:
                c = Circle(radius=radius, stroke_color=PRIMARY, stroke_width=2,
                           fill_color=PANEL, fill_opacity=1.0)
                c.move_to([x, y, 0])
                self.nodes.add(c)
        idx = 0
        starts = []
        for L in range(levels):
            starts.append(idx)
            idx += 2 ** L
        for L in range(levels - 1):
            for i in range(2 ** L):
                parent = self.nodes[starts[L] + i]
                for child_off in (0, 1):
                    child = self.nodes[starts[L + 1] + 2 * i + child_off]
                    self.edges.add(
                        Line(parent.get_center(), child.get_center(), color=STROKE,
                             stroke_width=2.0)
                    )
        self.group = VGroup(self.edges, self.nodes)


# --------------------------------------------------------------------------------------
# Graph — nodes at explicit coords with optional directed edges.
# --------------------------------------------------------------------------------------
def graph_viz(coords: dict, edges, *, radius: float = 0.32, color=PRIMARY,
              directed: bool = False) -> VGroup:
    nodes = {}
    node_mobs = VGroup()
    for name, (x, y) in coords.items():
        c = Circle(radius=radius, stroke_color=color, stroke_width=2.2, fill_color=PANEL,
                   fill_opacity=1.0)
        c.move_to([x, y, 0])
        t = _txt(name, fs=SMALL_FS)
        _fit_inside(t, c, pad=0.06)
        nodes[name] = VGroup(c, t)
        node_mobs.add(nodes[name])
    edge_mobs = VGroup()
    for a, b in edges:
        A, B = nodes[a], nodes[b]
        if directed:
            edge_mobs.add(arrow_between(A, B, color=MUTED, stroke=2.2))
        else:
            edge_mobs.add(Line(A.get_center(), B.get_center(), color=STROKE, stroke_width=2.2))
    return VGroup(edge_mobs, node_mobs)


# --------------------------------------------------------------------------------------
# DP grid — a table that fills in; cells addressed by (r, c).
# --------------------------------------------------------------------------------------
class DPGrid:
    def __init__(self, rows: int, cols: int, *, cell: float = 0.58, buff: float = 0.05,
                 fill=PANEL):
        self.rows, self.cols = rows, cols
        self.cells = VGroup()
        self.boxes = []
        self.texts = []
        for r in range(rows):
            row = VGroup()
            for c in range(cols):
                box = card(cell, cell, color=STROKE, fill=fill)
                self.boxes.append(box)
                self.texts.append(None)
                row.add(VGroup(box))
            # Each row must be laid out RIGHT before the rows are stacked DOWN — without this
            # every cell in a row sits on the same point and the table renders as one column.
            row.arrange(RIGHT, buff=buff)
            self.cells.add(row)
        self.cells.arrange(DOWN, buff=buff)
        self.group = self.cells

    def box(self, r: int, c: int):
        return self.cells[r][c][0]

    def set_value(self, r: int, c: int, value, color=INK, fs: int = SMALL_FS):
        box = self.box(r, c)
        t = _txt(value, fs=fs, color=color)
        _fit_inside(t, box, pad=0.08)
        self.cells[r][c].add(t)
        self.texts[r * self.cols + c] = t
        return t

    def focus(self, r: int, c: int, color=ACCENT, opacity: float = 0.3):
        return self.box(r, c).animate.set_stroke(color, width=2.4).set_fill(color, opacity=opacity)


# --------------------------------------------------------------------------------------
# Interval track — intervals as bars on a shared number line (merge-intervals pattern).
# --------------------------------------------------------------------------------------
def interval_track(intervals, *, length: float = 10.0, height: float = 0.42,
                   buff: float = 0.24, lo: float | None = None, hi: float | None = None,
                   color=PRIMARY) -> VGroup:
    lo = min(s for s, _ in intervals) if lo is None else lo
    hi = max(e for _, e in intervals) if hi is None else hi
    span = max(hi - lo, 1)
    scale = length / span
    bars = VGroup()
    for (s, e) in intervals:
        w = max(0.14, (e - s) * scale)
        bar = RoundedRectangle(corner_radius=0.07, width=w, height=height,
                               stroke_color=color, stroke_width=1.6,
                               fill_color=color, fill_opacity=0.5)
        bars.add(bar)
    bars.arrange(DOWN, buff=buff, aligned_edge=LEFT)
    # position each bar horizontally by its start value
    for bar, (s, _e) in zip(bars, intervals):
        bar.shift(RIGHT * ((s - lo) * scale))
    axis = Line(LEFT * 0.1, RIGHT * (length + 0.1), color=STROKE, stroke_width=2)
    axis.next_to(bars, DOWN, buff=0.3).align_to(bars, LEFT)
    return VGroup(bars, axis)


def chip(text: str, *, color=PRIMARY, fs: int = BODY_FS, pad_w: float = 0.34):
    """A small rounded label — used for complexity readouts and recall cards."""
    t = Text(text, font=MONO, font_size=fs, color=color)
    box = card(t.width + 2 * pad_w, t.height + 0.3, color=color, fill=PANEL)
    t.move_to(box.get_center())
    return VGroup(box, t)


# ======================================================================================
# Deep-learning primitives
# ======================================================================================
def _fmt(v) -> str:
    """Compact cell text: ints stay ints, floats get at most 2 decimals."""
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e4:
            return str(int(v))
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v)


def _is_3d(data) -> bool:
    return bool(data) and isinstance(data[0], (list, tuple)) and bool(data[0]) and \
        isinstance(data[0][0], (list, tuple))


# --------------------------------------------------------------------------------------
# TensorGrid — the tensor as the audience pictures it, plus the metadata that defines it.
# --------------------------------------------------------------------------------------
class TensorGrid:
    """A 2D (rows x cols) or 3D (depth x rows x cols) grid of numbered cells.

    3D tensors are drawn as a 2D sketch (charter rule 4): back slices are offset up-right
    and dimmed behind the front slice. Optional rails:

      * `show_index`   row indices on the left, column indices on top (front slice);
      * `shape`        "shape (2, 3)" label under the grid (True = derive from data);
      * `stride`       "stride (3, 1)" label under the shape label.

    `self.group` holds everything; `self.grid` holds only the cells. Emphasis helpers return
    animation lists (in-place highlight, never a duplicate overlay), like ArrayRow.focus.
    """

    def __init__(self, data, *, cell: float = 0.62, buff: float = 0.06, fs: int = MONO_FS,
                 color=STROKE, fill=PANEL, show_index: bool = False,
                 index_fs: int = SMALL_FS, shape=True, stride=None, name: str | None = None,
                 label_fs: int = SMALL_FS, depth_offset: float | None = None,
                 max_width: float | None = None):
        self.data = data
        self.is3d = _is_3d(data)
        slices = data if self.is3d else [data]
        self.depth = len(slices)
        self.rows = len(slices[0])
        self.cols = len(slices[0][0]) if self.rows else 0
        self.cell_size = cell

        self.slices = []  # list of VGroup(rows of VGroup(box, text))
        self._boxes = {}  # (k, i, j) -> box
        self._texts = {}
        for k, sl in enumerate(slices):
            rows = VGroup()
            for i, r in enumerate(sl):
                row = VGroup()
                for j, v in enumerate(r):
                    box = card(cell, cell, color=color, fill=fill)
                    parts = VGroup(box)
                    if v is not None:
                        t = _txt(_fmt(v), fs=fs)
                        _fit_inside(t, box)
                        parts.add(t)
                        self._texts[(k, i, j)] = t
                    self._boxes[(k, i, j)] = box
                    row.add(parts)
                row.arrange(RIGHT, buff=buff)
                rows.add(row)
            rows.arrange(DOWN, buff=buff)
            self.slices.append(rows)

        # Back slices up-right and dimmed; front slice (k=0) drawn last. The offset is a
        # little over half a cell so every back cell shows a readable corner.
        if depth_offset is None:
            depth_offset = 0.55 * (cell + buff)
        for k in range(1, self.depth):
            self.slices[k].shift(RIGHT * depth_offset * k + UP * depth_offset * k)
            self.slices[k].set_opacity(max(0.35, 0.8 - 0.2 * k))
        self.grid = VGroup(*reversed(self.slices))
        self.front = self.slices[0]

        self.idx = VGroup()
        if show_index:
            for i in range(self.rows):
                lb = Text(str(i), font=MONO, font_size=index_fs, color=MUTED)
                lb.next_to(self.front[i], LEFT, buff=0.16)
                self.idx.add(lb)
            for j in range(self.cols):
                lb = Text(str(j), font=MONO, font_size=index_fs, color=MUTED)
                # 3D: back slices own the space above, so column indices go underneath.
                if self.is3d:
                    lb.next_to(self.front[-1][j], DOWN, buff=0.12)
                else:
                    lb.next_to(self.front[0][j], UP, buff=0.12)
                self.idx.add(lb)

        self.labels = VGroup()
        self.name_label = None
        self.shape_label = None
        self.stride_label = None
        if name:
            self.name_label = Text(name, font=MONO, font_size=label_fs + 2, color=INK)
        if shape:
            dims = (self.depth, self.rows, self.cols) if self.is3d else (self.rows, self.cols)
            text = shape if isinstance(shape, str) else f"shape ({', '.join(map(str, dims))})"
            self.shape_label = Text(text, font=MONO, font_size=label_fs, color=PRIMARY)
            self.labels.add(self.shape_label)
        if stride is not None:
            text = stride if isinstance(stride, str) else \
                f"stride ({', '.join(map(str, stride))})"
            self.stride_label = Text(text, font=MONO, font_size=label_fs, color=ACCENT)
            self.labels.add(self.stride_label)
        if len(self.labels):
            self.labels.arrange(DOWN, buff=0.12)
            self.labels.next_to(VGroup(self.grid, self.idx), DOWN, buff=0.26)
        self.group = VGroup(self.grid, self.idx, self.labels)
        if self.name_label is not None:
            self.name_label.next_to(VGroup(self.grid, self.idx), UP, buff=0.22)
            self.group.add(self.name_label)
        if max_width and self.group.width > max_width:
            self.group.scale_to_fit_width(max_width)

    # -- geometry ------------------------------------------------------------------
    def cell(self, i: int, j: int, k: int = 0):
        return self.slices[k][i][j]

    def box(self, i: int, j: int, k: int = 0):
        return self._boxes[(k, i, j)]

    def row(self, i: int, k: int = 0) -> VGroup:
        return self.slices[k][i]

    def col(self, j: int, k: int = 0) -> VGroup:
        return VGroup(*[self.slices[k][i][j] for i in range(self.rows)])

    # -- emphasis ------------------------------------------------------------------
    def focus(self, i: int, j: int, k: int = 0, color=ACCENT, fill_opacity: float = 0.30):
        return [self.box(i, j, k).animate.set_stroke(color=color, width=2.6)
                .set_fill(color, opacity=fill_opacity)]

    def focus_row(self, i: int, k: int = 0, color=ACCENT, fill_opacity: float = 0.25):
        anims = []
        for j in range(self.cols):
            anims += self.focus(i, j, k, color=color, fill_opacity=fill_opacity)
        return anims

    def focus_col(self, j: int, k: int = 0, color=PRIMARY, fill_opacity: float = 0.25):
        anims = []
        for i in range(self.rows):
            anims += self.focus(i, j, k, color=color, fill_opacity=fill_opacity)
        return anims

    def reset(self, i: int, j: int, k: int = 0):
        return [self.box(i, j, k).animate.set_stroke(color=STROKE, width=1.6)
                .set_fill(PANEL, opacity=1.0)]

    def reset_all(self):
        anims = []
        for (k, i, j) in self._boxes:
            anims += self.reset(i, j, k)
        return anims

    def set_value(self, i: int, j: int, value, k: int = 0, color=INK, fs: int = MONO_FS):
        """Write (or overwrite) a cell's number in place; returns the new Text to animate in."""
        box = self.box(i, j, k)
        new = _txt(_fmt(value), color=color, fs=fs)
        _fit_inside(new, box)
        old = self._texts.get((k, i, j))
        if old is not None:
            self.cell(i, j, k).remove(old)
        self.cell(i, j, k).add(new)
        self._texts[(k, i, j)] = new
        return new


def stride_jump(row: "ArrayRow", src: int, dst: int, text: str, *, color=ACCENT,
                above: bool = True, fs: int = SMALL_FS) -> VGroup:
    """A curved hop over a storage row from offset `src` to `dst`, labelled (e.g. '+3').

    This is how a stride is drawn: a jump distance in flat memory, not a grid direction."""
    a = row.top_of(src) if above else row.bottom_of(src)
    b = row.top_of(dst) if above else row.bottom_of(dst)
    lift = UP * 0.08 if above else DOWN * 0.08
    arc = CurvedArrow(a + lift, b + lift, angle=-1.1 if above else 1.1, color=color,
                      stroke_width=3.0, tip_length=0.16)
    lb = Text(text, font=MONO, font_size=fs, color=color)
    lb.next_to(arc, UP if above else DOWN, buff=0.08)
    return VGroup(arc, lb)


# --------------------------------------------------------------------------------------
# MatMul — row i of A times column j of B lands in C[i, j].
# --------------------------------------------------------------------------------------
class MatMul:
    """`A (n x k) @ B (k x m) = C (n x m)` laid out as  A  x  B  =  C  on one line, or in the
    textbook L (B above C, A left of C) with `layout="L"`. C starts empty and fills via
    `step(i, j)`, which returns (animations, readout Text) with the dot product spelled out.
    """

    def __init__(self, A, B, *, cell: float = 0.52, fs: int = SMALL_FS, layout: str = "row",
                 names=("A", "B", "C"), max_width: float = 12.6):
        self.A_data, self.B_data = A, B
        n, k, m = len(A), len(A[0]), len(B[0])
        self.C_data = [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(m)]
                       for i in range(n)]
        kw = dict(cell=cell, fs=fs, label_fs=SMALL_FS)
        self.A = TensorGrid(A, name=names[0], shape=f"{n} x {k}", **kw)
        self.B = TensorGrid(B, name=names[1], shape=f"{k} x {m}", **kw)
        self.C = TensorGrid([[None] * m for _ in range(n)], name=names[2],
                            shape=f"{n} x {m}", **kw)
        self.times = Text("x", font=MONO, font_size=30, color=MUTED)
        self.equals = Text("=", font=MONO, font_size=30, color=MUTED)
        if layout == "L":
            self.B.group.next_to(self.C.group, UP, buff=0.35)
            self.A.group.next_to(self.C.group, LEFT, buff=0.45)
            self.group = VGroup(self.A.group, self.B.group, self.C.group)
            self.times.set_opacity(0)
            self.equals.set_opacity(0)
        else:
            self.group = VGroup(self.A.group, self.times, self.B.group, self.equals,
                                self.C.group).arrange(RIGHT, buff=0.4)
        if self.group.width > max_width:
            self.group.scale_to_fit_width(max_width)
        self._fs = fs
        self.k = k

    def step(self, i: int, j: int, *, color_row=ACCENT, color_col=PRIMARY):
        terms = [f"{_fmt(self.A_data[i][t])}*{_fmt(self.B_data[t][j])}" for t in range(self.k)]
        val = self.C_data[i][j]
        readout = Text(" + ".join(terms) + f" = {_fmt(val)}", font=MONO,
                       font_size=SMALL_FS, color=INK)
        readout.next_to(self.group, DOWN, buff=0.4)
        new = self.C.set_value(i, j, val, fs=self._fs)
        new.set_opacity(0)
        anims = (self.A.focus_row(i, color=color_row) + self.B.focus_col(j, color=color_col)
                 + self.C.focus(i, j, color=GOOD) + [new.animate.set_opacity(1)])
        return anims, readout

    def clear_focus(self):
        return self.A.reset_all() + self.B.reset_all()


BG_MASK = "#0B0D11"  # masked (future) attention cell: darker than the ground


# --------------------------------------------------------------------------------------
# Heatmap — attention weights, with a colour scale and an optional causal mask.
# --------------------------------------------------------------------------------------
class Heatmap:
    """Matrix of weights in [vmin, vmax] painted from PANEL (low) to `hot` (high).

    * `row_labels` / `col_labels` (e.g. tokens) sit left / above;
    * `show_values` prints each weight inside its cell;
    * `causal=True` draws the upper triangle (j > i) as masked from the start; or build it
      unmasked and play `mask()` to animate the causal mask arriving;
    * `colorbar=True` adds a low..high scale to the right.
    """

    def __init__(self, weights, *, row_labels=None, col_labels=None, cell: float = 0.56,
                 buff: float = 0.04, hot=ACCENT, vmin: float | None = None,
                 vmax: float | None = None, show_values: bool = False, fs: int = 14,
                 causal: bool = False, colorbar: bool = True, label_fs: int = SMALL_FS):
        self.w = weights
        self.n, self.m = len(weights), len(weights[0])
        flat = [v for r in weights for v in r]
        self.vmin = min(flat) if vmin is None else vmin
        self.vmax = max(flat) if vmax is None else vmax
        self.hot = hot
        self.cells = VGroup()
        self._boxes = {}
        for i in range(self.n):
            row = VGroup()
            for j in range(self.m):
                box = Rectangle(width=cell, height=cell, stroke_color=STROKE, stroke_width=1.0,
                                fill_color=self.color_of(weights[i][j]), fill_opacity=1.0)
                parts = VGroup(box)
                if show_values:
                    t = _txt(_fmt(round(weights[i][j], 2)), fs=fs,
                             color=INK if self._alpha(weights[i][j]) < 0.6 else PANEL)
                    _fit_inside(t, box, pad=0.08)
                    parts.add(t)
                self._boxes[(i, j)] = box
                row.add(parts)
            row.arrange(RIGHT, buff=buff)
            self.cells.add(row)
        self.cells.arrange(DOWN, buff=buff)

        self.row_labels = VGroup()
        for i, lb in enumerate(row_labels or []):
            t = Text(str(lb), font=MONO, font_size=label_fs, color=MUTED)
            t.next_to(self.cells[i], LEFT, buff=0.16)
            self.row_labels.add(t)
        self.col_labels = VGroup()
        for j, lb in enumerate(col_labels or []):
            t = Text(str(lb), font=MONO, font_size=label_fs, color=MUTED)
            t.next_to(self.cells[0][j], UP, buff=0.12)
            self.col_labels.add(t)

        self.colorbar = VGroup()
        if colorbar:
            steps = 8
            bar = VGroup(*[
                Rectangle(width=0.22, height=self.cells.height / steps, stroke_width=0,
                          fill_color=self.color_of(self.vmin + (self.vmax - self.vmin)
                                                   * s / (steps - 1)), fill_opacity=1.0)
                for s in reversed(range(steps))
            ]).arrange(DOWN, buff=0)
            bar.next_to(self.cells, RIGHT, buff=0.3)
            hi = Text(_fmt(round(self.vmax, 2)), font=MONO, font_size=14, color=MUTED)
            lo = Text(_fmt(round(self.vmin, 2)), font=MONO, font_size=14, color=MUTED)
            hi.next_to(bar, RIGHT, buff=0.1).align_to(bar, UP)
            lo.next_to(bar, RIGHT, buff=0.1).align_to(bar, DOWN)
            self.colorbar = VGroup(bar, hi, lo)

        self.mask_marks = VGroup()
        for i in range(self.n):
            for j in range(self.m):
                if j > i:
                    b = self._boxes[(i, j)]
                    mark = Rectangle(width=b.width, height=b.height, stroke_width=0,
                                     fill_color=GONE, fill_opacity=0.0).move_to(b)
                    self.mask_marks.add(mark)
        self.group = VGroup(self.cells, self.row_labels, self.col_labels, self.colorbar,
                            self.mask_marks)
        if causal:
            self._apply_mask_static()

    def _alpha(self, v) -> float:
        span = (self.vmax - self.vmin) or 1.0
        return min(1.0, max(0.0, (v - self.vmin) / span))

    def color_of(self, v):
        return interpolate_color(PANEL, self.hot, self._alpha(v))

    def box(self, i: int, j: int):
        return self._boxes[(i, j)]

    def _apply_mask_static(self):
        for (i, j), b in self._boxes.items():
            if j > i:
                b.set_fill(BG_MASK, opacity=1.0)
        for m in self.mask_marks:
            m.set_fill(GONE, opacity=0.25)

    def mask(self):
        """Animations that apply the causal mask: future positions go dark and red-tinted."""
        anims = [self._boxes[(i, j)].animate.set_fill(BG_MASK, opacity=1.0)
                 for (i, j) in self._boxes if j > i]
        anims += [m.animate.set_fill(GONE, opacity=0.25) for m in self.mask_marks]
        return anims

    def focus_row(self, i: int, color=ACCENT):
        return [self._boxes[(i, j)].animate.set_stroke(color, width=2.4) for j in range(self.m)]

    def set_weights(self, weights):
        """Animations that recolour every cell to a new weight matrix (computed state)."""
        self.w = weights
        return [self._boxes[(i, j)].animate.set_fill(self.color_of(weights[i][j]), opacity=1.0)
                for i in range(self.n) for j in range(self.m)]


# --------------------------------------------------------------------------------------
# BlockDiagram / layer_stack — boxes with arrows, shape labels riding the arrows.
# --------------------------------------------------------------------------------------
class BlockDiagram:
    """Named boxes at explicit coordinates, joined by arrows that may carry a shape label.

        bd = BlockDiagram(
            {"x": ((0, -2), "tokens"), "emb": ((0, -0.8), "Embedding"),
             "attn": ((0, 0.4), "Attention"), "mlp": ((0, 1.6), "MLP")},
            [("x", "emb", "(B, T)"), ("emb", "attn", "(B, T, C)"), ("attn", "mlp", "")],
        )
        self.play(*bd.reveal())          # boxes then arrows, in order
        self.play(*bd.flow())            # a pulse travels every edge (data flowing)
        self.play(*bd.focus("attn"))     # amber the block being discussed

    Edge labels sit BESIDE the arrow (to the right of vertical arrows, above horizontal).
    """

    def __init__(self, nodes: dict, edges=(), *, box_w: float = 2.6, box_h: float = 0.62,
                 fs: int = BODY_FS, label_fs: int = 16, color=PRIMARY, widths: dict | None = None):
        self.boxes = {}
        self.node_mobs = VGroup()
        widths = widths or {}
        for name, ((x, y), text) in nodes.items():
            w = widths.get(name, box_w)
            box = card(w, box_h, color=color, fill=PANEL)
            box.move_to([x, y, 0])
            t = Text(text, font="Helvetica Neue", font_size=fs, color=INK)
            _fit_inside(t, box, pad=0.24)
            g = VGroup(box, t)
            self.boxes[name] = g
            self.node_mobs.add(g)
        self.arrows = {}
        self.edge_labels = VGroup()
        self.edge_mobs = VGroup()
        for e in edges:
            a, b = e[0], e[1]
            text = e[2] if len(e) > 2 else ""
            A, B = self.boxes[a], self.boxes[b]
            dx = B.get_center()[0] - A.get_center()[0]
            dy = B.get_center()[1] - A.get_center()[1]
            if abs(dy) >= abs(dx):
                start = A.get_top() if dy > 0 else A.get_bottom()
                end = B.get_bottom() if dy > 0 else B.get_top()
            else:
                start = A.get_right() if dx > 0 else A.get_left()
                end = B.get_left() if dx > 0 else B.get_right()
            arr = Arrow(start, end, buff=0.06, color=MUTED, stroke_width=2.6,
                        max_tip_length_to_length_ratio=0.25)
            self.arrows[(a, b)] = arr
            self.edge_mobs.add(arr)
            if text:
                lb = Text(text, font=MONO, font_size=label_fs, color=ACCENT)
                lb.next_to(arr, RIGHT if abs(dy) >= abs(dx) else UP, buff=0.12)
                self.edge_labels.add(lb)
        self.group = VGroup(self.edge_mobs, self.node_mobs, self.edge_labels)

    def reveal(self, lag: float = 0.0):
        from manim import FadeIn, GrowArrow
        return ([FadeIn(n, shift=UP * 0.1) for n in self.node_mobs]
                + [GrowArrow(a) for a in self.edge_mobs]
                + [FadeIn(lb) for lb in self.edge_labels])

    def flow(self, color=ACCENT, edges=None):
        """A pulse along each edge: data moving through the diagram (ShowPassingFlash)."""
        keys = edges or list(self.arrows)
        return [ShowPassingFlash(self.arrows[k].copy().set_color(color).set_stroke(width=6),
                                 time_width=0.5) for k in keys]

    def focus(self, name: str, color=ACCENT):
        return [self.boxes[name][0].animate.set_stroke(color, width=3.0)
                .set_fill(color, opacity=0.18)]

    def reset(self, name: str, color=PRIMARY):
        return [self.boxes[name][0].animate.set_stroke(color, width=1.6)
                .set_fill(PANEL, opacity=1.0)]


def layer_stack(names, *, shapes=None, gap: float = 0.55, box_w: float = 3.0,
                box_h: float = 0.6, bottom_y: float = -2.2, x: float = 0.0,
                **kw) -> BlockDiagram:
    """A bottom-to-top stack of layers (input at the bottom, as in the transformer figure).

    `shapes[i]` labels the arrow from layer i to layer i+1 (e.g. "(B, T, C)")."""
    nodes = {}
    for i, n in enumerate(names):
        nodes[f"L{i}"] = ((x, bottom_y + i * (box_h + gap)), n)
    edges = []
    for i in range(len(names) - 1):
        lab = shapes[i] if shapes and i < len(shapes) else ""
        edges.append((f"L{i}", f"L{i + 1}", lab))
    return BlockDiagram(nodes, edges, box_w=box_w, box_h=box_h, **kw)


# --------------------------------------------------------------------------------------
# CurvePlot — axes + curve, labelled with Text (no LaTeX), plus a tracker-driven dot.
# --------------------------------------------------------------------------------------
class CurvePlot:
    """Axes and one or more curves. `fn` is a python function or a list of (x, y) points.

        p = CurvePlot(lambda x: 2.5 * math.exp(-x / 3) + 0.4, x_range=(0, 10),
                      y_range=(0, 3), x_label="step", y_label="loss")
        self.play(Create(p.axes), FadeIn(p.labels)); self.play(Create(p.curve))
        dot = p.tracking_dot(t := ValueTracker(0)); self.add(dot)
        self.play(t.animate.set_value(10))   # dot rides the curve (computed, not keyframed)
    """

    def __init__(self, fn, *, x_range=(0, 10), y_range=(0, 1), width: float = 7.0,
                 height: float = 3.6, x_label: str = "", y_label: str = "", color=PRIMARY,
                 ticks: bool = True, tick_fs: int = 14, label_fs: int = SMALL_FS):
        x0, x1 = x_range[:2]
        y0, y1 = y_range[:2]
        self.axes = Axes(
            x_range=[x0, x1, (x1 - x0) / 5], y_range=[y0, y1, (y1 - y0) / 4],
            x_length=width, y_length=height,
            axis_config={"color": STROKE, "stroke_width": 2, "include_tip": False,
                         "include_ticks": ticks},
        )
        self.fn = fn
        self.curve = self.add_curve(fn, color=color)
        self.labels = VGroup()
        if x_label:
            xl = Text(x_label, font=MONO, font_size=label_fs, color=MUTED)
            xl.next_to(self.axes.x_axis, DOWN, buff=0.22).align_to(self.axes.x_axis, RIGHT)
            self.labels.add(xl)
        if y_label:
            yl = Text(y_label, font=MONO, font_size=label_fs, color=MUTED)
            yl.next_to(self.axes.y_axis, UP, buff=0.14)
            self.labels.add(yl)
        if ticks:
            for xv in (x0, x1):
                t = Text(_fmt(float(xv)), font=MONO, font_size=tick_fs, color=MUTED)
                t.next_to(self.axes.c2p(xv, y0), DOWN, buff=0.12)
                self.labels.add(t)
            for yv in (y1,):
                t = Text(_fmt(float(yv)), font=MONO, font_size=tick_fs, color=MUTED)
                t.next_to(self.axes.c2p(x0, yv), LEFT, buff=0.12)
                self.labels.add(t)
        self.group = VGroup(self.axes, self.curve, self.labels)

    def add_curve(self, fn, color=PRIMARY, stroke: float = 3.4):
        if callable(fn):
            return self.axes.plot(fn, color=color, stroke_width=stroke)
        pts = [self.axes.c2p(x, y) for x, y in fn]
        return VMobject(color=color, stroke_width=stroke).set_points_smoothly(pts)

    def y_at(self, x: float) -> float:
        if callable(self.fn):
            return self.fn(x)
        pts = sorted(self.fn)
        for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
            if xa <= x <= xb:
                return ya + (yb - ya) * (x - xa) / ((xb - xa) or 1)
        return pts[-1][1]

    def tracking_dot(self, tracker: "ValueTracker", color=ACCENT, radius: float = 0.08):
        """A dot whose position is computed from `tracker` every frame (charter rule 5)."""
        return always_redraw(lambda: Dot(self.axes.c2p(tracker.get_value(),
                                                       self.y_at(tracker.get_value())),
                                         radius=radius, color=color))


def softmax(xs, temperature: float = 1.0):
    """Plain softmax for computing heatmap / bar targets inside a scene."""
    m = max(xs)
    es = [math.exp((x - m) / temperature) for x in xs]
    s = sum(es)
    return [e / s for e in es]


# --------------------------------------------------------------------------------------
# CompGraph — autograd computation graph (micrograd-style).
# --------------------------------------------------------------------------------------
def _border_point(mob, toward):
    """Where the ray from mob's centre toward `toward` leaves mob's bounding box."""
    c = mob.get_center()
    d = toward - c
    hw, hh = mob.width / 2, mob.height / 2
    tx = hw / abs(d[0]) if abs(d[0]) > 1e-9 else float("inf")
    ty = hh / abs(d[1]) if abs(d[1]) > 1e-9 else float("inf")
    t = min(tx, ty, 1.0)
    return c + d * t


def value_node(name: str, data=None, grad=None, *, w: float = 1.7, h: float = 0.86,
               color=PRIMARY, fs: int = 18) -> VGroup:
    """A value node: rounded box with its name, `data` and `grad` rows."""
    box = card(w, h, color=color, fill=PANEL)
    lines = VGroup(Text(name, font=MONO, font_size=fs, color=INK))
    if data is not None:
        lines.add(Text(f"data {_fmt(data)}", font=MONO, font_size=fs - 4, color=MUTED))
    if grad is not None:
        lines.add(Text(f"grad {_fmt(grad)}", font=MONO, font_size=fs - 4, color=ACCENT))
    lines.arrange(DOWN, buff=0.06)
    _fit_inside(lines, box, pad=0.16)
    return VGroup(box, lines)


def op_node(op: str, *, radius: float = 0.3, color=MUTED, fs: int = 22) -> VGroup:
    """An operation node: a small circle holding the op symbol (+, *, tanh, ...)."""
    c = Circle(radius=radius, stroke_color=color, stroke_width=2.2, fill_color=PANEL,
               fill_opacity=1.0)
    t = Text(op, font=MONO, font_size=fs, color=INK)
    _fit_inside(t, c, pad=0.12)
    return VGroup(c, t)


class CompGraph:
    """Computation graph from explicit coordinates.

        g = CompGraph(
            {"a": ("value", (-4, 1), 2.0), "b": ("value", (-4, -1), -3.0),
             "mul": ("op", (-2, 0), "*"), "c": ("value", (0, 0), -6.0)},
            [("a", "mul"), ("b", "mul"), ("mul", "c")],
        )
        self.play(*g.reveal()); self.play(*g.forward_flash())
        self.play(*g.set_grad("c", 1.0)); self.play(*g.backward_flash(("mul", "c")))

    Value nodes carry (kind, (x, y), data); op nodes carry (kind, (x, y), symbol).
    """

    def __init__(self, nodes: dict, edges, *, show_grad: bool = True):
        self.nodes = {}
        self.data = {}
        self.group_nodes = VGroup()
        for name, (kind, (x, y), payload) in nodes.items():
            if kind == "op":
                m = op_node(str(payload))
            else:
                self.data[name] = payload
                m = value_node(name, data=payload, grad=0.0 if show_grad else None)
            m.move_to([x, y, 0])
            self.nodes[name] = m
            self.group_nodes.add(m)
        self.edges = {}
        self.group_edges = VGroup()
        for a, b in edges:
            A, B = self.nodes[a], self.nodes[b]
            # Anchor on the node borders along the centre line, so the arrow never runs
            # under a box whatever the relative placement.
            arr = Arrow(_border_point(A, B.get_center()), _border_point(B, A.get_center()),
                        buff=0.05, color=MUTED, stroke_width=2.4,
                        max_tip_length_to_length_ratio=0.3)
            self.edges[(a, b)] = arr
            self.group_edges.add(arr)
        self.group = VGroup(self.group_edges, self.group_nodes)

    def reveal(self):
        from manim import FadeIn, GrowArrow
        return [FadeIn(n) for n in self.group_nodes] + [GrowArrow(e) for e in self.group_edges]

    def forward_flash(self, color=PRIMARY):
        return [ShowPassingFlash(e.copy().set_color(color).set_stroke(width=6), time_width=0.5)
                for e in self.group_edges]

    def backward_flash(self, edge, color=ACCENT):
        """A pulse running AGAINST an edge: the gradient flowing back."""
        e = self.edges[edge]
        back = Arrow(e.get_end(), e.get_start(), buff=0, color=color, stroke_width=6,
                     max_tip_length_to_length_ratio=0.25)
        return [ShowPassingFlash(back, time_width=0.6)]

    def set_grad(self, name: str, grad, color=ACCENT):
        """Swap the node's grad row for a new value (returns animations)."""
        from manim import Transform
        node = self.nodes[name]
        lines = node[1]
        if len(lines) < 3:
            return []
        new = Text(f"grad {_fmt(grad)}", font=MONO, font_size=lines[2].font_size, color=color)
        new.match_height(lines[2]).move_to(lines[2])
        return [Transform(lines[2], new), node[0].animate.set_stroke(color, width=2.6)]
