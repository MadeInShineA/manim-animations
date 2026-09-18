from manim import *
import random

class QuickSort(Scene):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title = Tex(r"Quick Sort Algorithm", font_size=48)

        self.array_size = 8

        self.array_data = [random.randint(0, 2 * self.array_size) for _ in range(self.array_size)]

        self.array = Matrix([self.array_data], left_bracket="[", right_bracket="]")

        self.array_colors =  [WHITE for _ in range(self.array_size)]

        self.left_index_arrow = None
        self.right_index_arrow = None

    def set_arrow_under_array_index(self, index: int, color, label: str) -> VGroup:
        entry = self.array.get_entries()[index]
        arrow = Tex(r"$\uparrow$", font_size=60, color=color)
        label_mob = Tex(label, font_size=40, color=color)
        group = VGroup(arrow, label_mob).arrange(DOWN, buff=0.05)
        group.next_to(entry, DOWN, buff=0.15)
        self.play(FadeIn(group))
        return group

    def move_arrow_to_index(self, arrow: Tex, index: int) -> None:
        entry = self.array.get_entries()[index]
        self.play(arrow.animate.next_to(entry, DOWN, buff=0.15))


    def swap(self, i: int, j: int) -> None:
        self.array_data[i], self.array_data[j] = self.array_data[j], self.array_data[i]

        row = self.array.get_rows()[0]
        m1, m2 = row[i], row[j]
        p1, p2 = m1.get_center(), m2.get_center()
        self.play(m1.animate.move_to(p2), m2.animate.move_to(p1))
        self.wait()

        row.submobjects[i], row.submobjects[j] = row.submobjects[j], row.submobjects[i]

    def partition(self, array: list[int], left_idx: int, right_idx: int, write_text: bool) -> int:

        pivot_value = array[left_idx]

        i = left_idx + 1

        for j in range(left_idx + 1, right_idx):
            if array[j] < pivot_value:
                self.swap(i, j)
                i += 1

        self.swap(left_idx, i-1)

        return i-1

    def quicksort(self, array: list[int], left_index: int, right_index: int, write_text: bool = False) -> None:
        if left_index >= right_index:
            return

        if write_text:
            select_pivot_text = Tex("Select a pivot at random", font_size=30).next_to(self.title, DOWN, buff=0.4)
            self.play(Write(select_pivot_text))
            self.wait()

        pivot_idx = random.randint(left_index, right_index - 1)

        self.array_colors[pivot_idx] = RED
        self.array.set_column_colors(*self.array_colors)
        self.wait()

        if write_text:
            self.play(FadeOut(select_pivot_text))

            swap_text = Tex("Move the pivot to the left pointer",font_size=30).next_to(self.title, DOWN, buff=0.4)
            self.play(Write(swap_text))
            self.wait()

        self.swap(left_index, pivot_idx)

        if write_text:
            self.play(FadeOut(swap_text))

        return

        new_pivot_idx = self.partition(array, left_index, right_index)

        self.quicksort(array, left_index, new_pivot_idx)
        self.quicksort(array,new_pivot_idx + 1, right_index)

    def construct(self):


        self.play(FadeIn(self.title))
        self.wait(1)

        self.play(self.title.animate.to_edge(UP))


        self.play(Create(self.array))
        self.wait()

        index_text = Tex("Set a pointer to the left and one on the right of the array", font_size=30).next_to(self.title, DOWN, buff=0.4)
        self.play(Write(index_text))
        self.wait()

        self.left_index_arrow = self.set_arrow_under_array_index(0, GREEN, "L")
        self.wait()

        self.right_index_arrow= self.set_arrow_under_array_index(self.array_size - 1, ORANGE, "R")
        self.wait()
        self.play(FadeOut(index_text))


        self.quicksort(self.array_data, 0, self.array_size ,True)
