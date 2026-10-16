"""
M01E01 - Tensors: what they actually are.

The argument: a grid is how we picture a tensor, but what is stored is one flat block of
memory. Shape and stride are the reading instructions (offset = i*stride0 + j*stride1), so
a transpose is just new metadata over the same six slots: no copy. dtype and device finish
the definition: tensor = storage + shape + stride + dtype + device.

Continuity chain: grid -> unrolled into storage -> the grid returns as a view of storage ->
stride hops on storage -> one element's offset -> the transposed view -> its read path hops
around the same storage -> bytes and device of that storage -> the recall card.

Narration lives in BEATS and every animation is sized against `tracker.duration`, so the
picture and the voice cannot drift. Layout adapts to the vertical cut via house.VERTICAL.
"""

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Brace,
    FadeIn,
    FadeOut,
    GrowArrow,
    Indicate,
    Restore,
    Text,
    Transform,
    TransformFromCopy,
    VGroup,
    Write,
)

from house import (
    ACCENT,
    BODY_FS,
    GOOD,
    INK,
    MONO,
    MUTED,
    PRIMARY,
    SMALL_FS,
    VERTICAL,
    bg,
    card,
    caption,
    headline,
    rule,
    swap_rails,
)
from prebaked_voice import PrebakedVoiceMovingCameraScene
from shots import ArrayRow, TensorGrid, chip, pointer, stride_jump

EPISODE_CODE = "M01E01"
TITLE = "Tensors: what they actually are"
SUMMARY = ("A tensor is one flat block of storage plus shape, stride, dtype and device; "
           "a transpose is new metadata over the same memory, not a copy.")
SCENE_CLASS = "TensorsWhatTheyAre"
POSTER_AT = 40.0

STORAGE = [10, 20, 30, 40, 50, 60]
ROWS, COLS = 2, 3
GRID = [STORAGE[r * COLS:(r + 1) * COLS] for r in range(ROWS)]  # [[10,20,30],[40,50,60]]
GRID_T = [[GRID[r][c] for r in range(ROWS)] for c in range(COLS)]  # [[10,40],[20,50],[30,60]]
STRIDE = (COLS, 1)
STRIDE_T = (1, COLS)

BEATS = [
    # 0 hook
    "A tensor looks like a grid of numbers. But the grid is not what is stored. "
    "Let's open one up and see what is really there.",
    # 1 storage
    "Underneath is the storage: one flat block of memory. Here it holds six numbers, "
    "side by side. This is the only place the data lives.",
    # 2 shape
    "Next comes the shape. Shape two by three tells us to read those six numbers "
    "as two rows of three.",
    # 3 stride
    "Then the stride. It says how far to jump in storage for one step along each "
    "dimension. Down one row, we jump three. Right one column, we jump one.",
    # 4 offset formula
    "So the element at row i, column j sits at offset i times three, plus j. "
    "Row one, column two lands on offset five, the number sixty.",
    # 5 transpose
    "Now we transpose. We swap the shape to three by two, and swap the strides to "
    "one and three. That is the entire operation.",
    # 6 no copy
    "Walk the new view and the reads hop around storage: zero, three, one, four, two, "
    "five. Same six slots, a different path. Not one number was copied.",
    # 7 dtype + device
    "Two last facts. The dtype says what kind of number each slot holds, here float "
    "thirty two, four bytes each. The device says where the storage lives, the CPU or a GPU.",
    # 8 recall
    "So a tensor is storage plus shape, stride, dtype and device. Change the metadata "
    "and we get a new view for free.",
]

CHAPTERS = {
    0: "A grid is a picture",
    1: "Storage",
    2: "Shape",
    3: "Stride",
    5: "Transpose without a copy",
    7: "dtype and device",
    8: "Recap",
}

# Layout: landscape puts the two views side by side under the storage row; the vertical cut
# keeps the same order but uses the tall frame (views side by side, readouts underneath).
if VERTICAL:
    L = dict(storage_y=3.7, cell=0.62, grid_cell=0.6, a=(-1.85, 0.5), t=(2.2, 0.5),
             mid=(0, -2.3), mid2=(0, -3.2), hook=(0, 0.6), chips_y=(-1.4, -2.5))
else:
    L = dict(storage_y=0.95, cell=0.72, grid_cell=0.62, a=(-4.0, -1.35), t=(4.0, -1.35),
             mid=(0, -1.0), mid2=(0, -1.75), hook=(0, -0.1), chips_y=(-1.45, -2.25))


def _remaining(tr, buff=0.35):
    return max(0.1, tr.get_remaining_duration(buff=buff))


class TensorsWhatTheyAre(PrebakedVoiceMovingCameraScene):
    def construct(self):
        bg(self)
        self.storage = ArrayRow(STORAGE, cell=L["cell"], fs=24, index_fs=SMALL_FS)
        self.storage.group.move_to([0, L["storage_y"], 0])
        self.store_tag = Text("storage", font=MONO, font_size=BODY_FS, color=PRIMARY)
        self.off_tag = Text("offset", font=MONO, font_size=SMALL_FS - 2, color=MUTED)
        if VERTICAL:
            self.store_tag.next_to(self.storage.idx, DOWN, buff=0.22)
            self.off_tag.next_to(self.storage.idx, LEFT, buff=0.18)
        else:
            self.store_tag.next_to(self.storage.cells, LEFT, buff=0.45)
            self.off_tag.next_to(self.storage.idx, LEFT, buff=0.45)
        for i in range(len(BEATS)):
            getattr(self, f"_beat{i}")()

    # ------------------------------------------------------------------------- helpers
    def _view(self, data, stride, pos, fs=22):
        g = TensorGrid(data, cell=L["grid_cell"], fs=fs, stride=stride, label_fs=SMALL_FS)
        g.group.move_to([pos[0], pos[1], 0])
        return g

    # --------------------------------------------------------------------------- beats
    def _beat0(self):
        """Hook: the grid everybody pictures. It becomes the thing we take apart."""
        with self.voiceover(text=BEATS[0]) as tr:
            swap_rails(self, headline("A tensor is not really a grid."),
                       caption("so what is actually stored?"))
            self.play(FadeIn(rule()), run_time=0.3)
            self.hook = TensorGrid(GRID, cell=0.8, fs=28, shape=False)
            self.hook.group.move_to([L["hook"][0], L["hook"][1], 0])
            self.play(FadeIn(self.hook.grid, shift=UP * 0.2),
                      run_time=min(1.2, max(0.6, tr.duration * 0.2)))
            q = Text("?", font=MONO, font_size=40, color=ACCENT)
            q.next_to(self.hook.grid, RIGHT, buff=0.4)
            self.play(FadeIn(q, scale=0.8), run_time=0.4)
            self.wait(_remaining(tr, 0.5))
            self.play(FadeOut(q), run_time=0.25)

    def _beat1(self):
        """The grid unrolls into one flat row: storage, with its offsets."""
        with self.voiceover(text=BEATS[1]) as tr:
            swap_rails(self, headline("Underneath: one flat block of memory."),
                       caption("storage, six numbers side by side"))
            moves = []
            for r in range(ROWS):
                for c in range(COLS):
                    moves.append(TransformFromCopy(self.hook.cell(r, c),
                                                   self.storage.cell(r * COLS + c)))
            self.play(*moves, run_time=min(1.8, max(1.0, tr.duration * 0.25)))
            self.play(FadeOut(self.hook.grid), FadeIn(self.storage.idx),
                      FadeIn(self.store_tag), FadeIn(self.off_tag), run_time=0.6)
            # The copies became the storage cells; swap in the real row object.
            self.remove(*[m for m in self.mobjects if m not in self.storage.cells
                          and m not in (self.storage.idx, self.store_tag, self.off_tag)
                          and m not in self._rails and m is not None
                          and type(m).__name__ == "VGroup" and m not in self._keep()])
            self.add(self.storage.cells)
            self.play(Indicate(self.storage.cells, color=PRIMARY, scale_factor=1.04),
                      run_time=0.7)
            self.wait(_remaining(tr))

    def _keep(self):
        return [m for m in self.mobjects if type(m).__name__ == "Line"]

    def _beat2(self):
        """Shape: the grid comes back, read row by row out of storage."""
        with self.voiceover(text=BEATS[2]) as tr:
            swap_rails(self, headline("Shape says how to read it."),
                       caption("shape (2, 3): two rows of three"))
            self.A = self._view(GRID, STRIDE, L["a"])
            step = min(0.9, max(0.5, tr.duration * 0.14))
            for r in range(ROWS):
                span = VGroup(*[self.storage.cell(r * COLS + c) for c in range(COLS)])
                self.play(Indicate(span, color=ACCENT, scale_factor=1.06),
                          TransformFromCopy(span, self.A.row(r)), run_time=step)
            self.play(FadeIn(self.A.shape_label, shift=UP * 0.1), run_time=0.45)
            self.wait(_remaining(tr))

    def _beat3(self):
        """Stride: camera into the label, then the two hops drawn on storage."""
        with self.voiceover(text=BEATS[3]) as tr:
            swap_rails(self, headline("Stride says how far to jump."),
                       caption("stride (3, 1): row +3, column +1"))
            self.play(FadeIn(self.A.stride_label, shift=UP * 0.1), run_time=0.4)
            frame = self.camera.frame
            frame.save_state()
            self.play(frame.animate.scale(0.55).move_to(self.A.labels.get_center() + UP * 0.6),
                      run_time=0.7)
            self.wait(min(1.4, max(0.4, tr.duration * 0.12)))
            self.play(Restore(frame), run_time=0.6)

            hop_rt = min(0.9, max(0.5, tr.duration * 0.1))
            # Down one row: (0,0) -> (1,0) is offset 0 -> 3.
            self.play(*self.A.focus(0, 0), *self.A.focus(1, 0, color=PRIMARY), run_time=0.35)
            down = stride_jump(self.storage, 0, 3, "+3  next row", color=PRIMARY)
            self.play(FadeIn(down), run_time=hop_rt)
            self.wait(min(1.2, max(0.3, tr.duration * 0.1)))
            self.play(FadeOut(down), *self.A.reset(1, 0), run_time=0.35)
            # Right one column: (0,0) -> (0,1) is offset 0 -> 1.
            self.play(*self.A.focus(0, 1, color=GOOD), run_time=0.3)
            right = stride_jump(self.storage, 0, 1, "+1  next column", color=GOOD)
            self.play(FadeIn(right), run_time=hop_rt)
            self.wait(_remaining(tr, 0.75))
            self.play(FadeOut(right), *self.A.reset_all(), run_time=0.35)

    def _beat4(self):
        """offset = i*3 + j*1, then evaluated for (1, 2) -> offset 5 -> sixty."""
        with self.voiceover(text=BEATS[4]) as tr:
            swap_rails(self, headline("Every element is one offset."),
                       caption("offset = i x 3 + j x 1"))
            f1 = Text("offset = i*3 + j*1", font=MONO, font_size=BODY_FS, color=INK)
            f1.move_to([L["mid"][0], L["mid"][1], 0])
            self.play(Write(f1), run_time=min(1.2, max(0.6, tr.duration * 0.15)))
            self.wait(min(1.6, max(0.3, tr.duration * 0.18)))
            f2 = Text("(1, 2): 1*3 + 2*1 = 5", font=MONO, font_size=BODY_FS, color=ACCENT)
            f2.move_to([L["mid2"][0], L["mid2"][1], 0])
            self.play(*self.A.focus(1, 2), Write(f2), run_time=0.8)
            ptr = pointer(self.storage, 5, "offset 5", color=ACCENT, fs=SMALL_FS)
            self.play(GrowArrow(ptr[0]), FadeIn(ptr[1]), *self.storage.focus(5), run_time=0.6)
            self.wait(_remaining(tr, 0.6))
            self.play(FadeOut(VGroup(f1, f2, ptr)), *self.A.reset(1, 2),
                      *self.storage.reset(5), run_time=0.4)

    def _beat5(self):
        """Transpose: each cell (i, j) travels to (j, i); shape and stride swap."""
        with self.voiceover(text=BEATS[5]) as tr:
            swap_rails(self, headline("Transpose: swap shape and stride."),
                       caption("(2, 3) / (3, 1)   becomes   (3, 2) / (1, 3)"))
            self.T = self._view(GRID_T, STRIDE_T, L["t"])
            arrow = Text("x.T", font=MONO, font_size=BODY_FS, color=MUTED)
            arrow.move_to([L["mid"][0], L["a"][1] + 0.2, 0]) if not VERTICAL else \
                arrow.move_to([(L["a"][0] + L["t"][0]) / 2 + 0.15, L["a"][1] + 1.45, 0])
            self.play(FadeIn(arrow), run_time=0.35)
            moves = [TransformFromCopy(self.A.cell(i, j), self.T.cell(j, i))
                     for i in range(ROWS) for j in range(COLS)]
            self.play(*moves, run_time=min(1.6, max(0.9, tr.duration * 0.22)))
            self.play(TransformFromCopy(self.A.shape_label, self.T.shape_label), run_time=0.6)
            self.play(TransformFromCopy(self.A.stride_label, self.T.stride_label), run_time=0.6)
            self.play(Indicate(self.T.stride_label, color=ACCENT, scale_factor=1.12),
                      run_time=0.6)
            self.wait(_remaining(tr))
            self.xT = arrow

    def _beat6(self):
        """Read the transposed view in order: the storage read-head hops 0,3,1,4,2,5."""
        with self.voiceover(text=BEATS[6]) as tr:
            swap_rails(self, headline("Same storage, different path."),
                       caption("no data moved, nothing was copied"))
            self.play(FadeOut(self.xT), run_time=0.25)
            order = [(r, c) for r in range(COLS) for c in range(ROWS)]
            offsets = [r * STRIDE_T[0] + c * STRIDE_T[1] for r, c in order]
            path = Text("read: " + "  ".join(map(str, offsets)), font=MONO,
                        font_size=BODY_FS, color=INK)
            path.move_to([L["mid"][0], L["mid"][1], 0])
            step = min(0.75, max(0.4, tr.duration * 0.5 / len(order)))
            head = None
            for (r, c), off in zip(order, offsets):
                new_head = pointer(self.storage, off, "read", color=ACCENT, fs=SMALL_FS)
                anims = [*self.T.focus(r, c), *self.storage.focus(off, fill_opacity=0.22)]
                if head is None:
                    anims += [GrowArrow(new_head[0]), FadeIn(new_head[1])]
                else:
                    anims += [Transform(head, new_head)]
                self.play(*anims, run_time=step)
                if head is None:
                    head = new_head
            self.play(FadeIn(path, shift=UP * 0.1), run_time=0.4)
            same = chip("no copy: same 6 slots", color=GOOD, fs=SMALL_FS)
            same.move_to([L["mid2"][0], L["mid2"][1] - 0.1, 0])
            self.play(FadeIn(same, scale=0.95), run_time=0.45)
            self.wait(_remaining(tr, 0.6))
            self.play(FadeOut(VGroup(head, path, same, self.A.group, self.T.group)),
                      *self.storage.reset_all(), run_time=0.5)

    def _beat7(self):
        """dtype and device: bytes per slot, and where the block lives."""
        with self.voiceover(text=BEATS[7]) as tr:
            swap_rails(self, headline("dtype and device finish the picture."),
                       caption("float32 = 4 bytes per slot; cpu, mps or cuda"))
            brace = Brace(VGroup(self.storage.idx), DOWN, color=MUTED, buff=0.12)
            if VERTICAL:
                self.play(self.store_tag.animate.next_to(self.storage.cells, UP, buff=0.25),
                          run_time=0.3)
            nbytes = Text("6 x 4 bytes = 24 bytes", font=MONO, font_size=SMALL_FS, color=MUTED)
            nbytes.next_to(brace, DOWN, buff=0.12)
            dtype = chip("dtype   float32", color=ACCENT, fs=BODY_FS)
            device = chip("device   cpu | mps | cuda", color=PRIMARY, fs=BODY_FS)
            dtype.move_to([0, L["chips_y"][0], 0])
            device.move_to([0, L["chips_y"][1], 0])
            self.play(FadeIn(dtype, shift=UP * 0.15), run_time=0.45)
            self.play(FadeIn(brace), FadeIn(nbytes), run_time=0.5)
            self.wait(min(2.5, max(0.3, tr.duration * 0.3)))
            self.play(FadeIn(device, shift=UP * 0.15), run_time=0.45)
            self.play(Indicate(self.storage.cells, color=PRIMARY, scale_factor=1.04),
                      run_time=0.6)
            self.wait(_remaining(tr, 0.6))
            self.play(FadeOut(VGroup(brace, nbytes, dtype, device, self.storage.group,
                                     self.store_tag, self.off_tag)), run_time=0.5)

    def _beat8(self):
        """Recall card: the five fields, storage first because it is the only data."""
        with self.voiceover(text=BEATS[8]) as tr:
            swap_rails(self, headline("A tensor = storage + metadata."),
                       caption("change the metadata, get a new view for free"))
            parts = [("storage", PRIMARY), ("shape", INK), ("stride", ACCENT),
                     ("dtype", INK), ("device", INK)]
            chips = VGroup(*[chip(n, color=c, fs=BODY_FS) for n, c in parts])
            if VERTICAL:
                chips.arrange(DOWN, buff=0.3).move_to([0, 0.9, 0])
            else:
                chips.arrange(RIGHT, buff=0.35).move_to([0, 0.35, 0])
            meta = Text("metadata", font=MONO, font_size=SMALL_FS, color=MUTED)
            meta_group = VGroup(*chips[1:])
            if VERTICAL:
                meta.next_to(meta_group, RIGHT, buff=0.3)
            else:
                meta.next_to(meta_group, DOWN, buff=0.3)
            box = card(6.0 if not VERTICAL else 7.6, 1.0, color=GOOD)
            box.move_to([0, -1.75 if not VERTICAL else -3.4, 0])
            take = Text("view = new metadata, same storage", font=MONO, font_size=SMALL_FS,
                        color=GOOD)
            take.move_to(box.get_center())
            self.play(*[FadeIn(c, shift=UP * 0.15) for c in chips], lag_ratio=0.25,
                      run_time=min(2.0, max(1.0, tr.duration * 0.3)))
            self.play(FadeIn(meta), run_time=0.35)
            self.play(FadeIn(box), Write(take), run_time=0.7)
            self.wait(_remaining(tr, 0.4))
