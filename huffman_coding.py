from manim import *
import re
import numpy as np


class HuffmanCoding(Scene):
    def construct(self):
        # Title
        title = Tex(r"Huffman Coding Algorithm", font_size=48)
        self.play(FadeIn(title))
        self.wait(1)
        self.play(FadeOut(title))

        def wrap_text(text: str, font_size: int, max_width: float) -> list[str]:
            """
            Splits text into lines that fit within max_width when rendered.
            Returns a list of line strings.
            """
            words = text.split(" ")
            lines: list[str] = []
            current = ""
            for w in words:
                candidate = (current + " " + w).strip()
                if (
                    Tex(r"\text{" + candidate + r"}", font_size=font_size).width
                    > max_width
                ):
                    lines.append(current)
                    current = w
                else:
                    current = candidate
            if current:
                lines.append(current)
            return lines

        # 1) Show a brief, two-sentence description of Huffman coding
        desc_raw = (
            "Huffman coding is a lossless compression method that assigns shorter bit-patterns "
            "to more frequent symbols by repeatedly merging the two least probable ones. "
            "The outcome is a set of variable-length, prefix-free binary codes that avoid "
            "ambiguity in decoding and minimize the average code length based on the symbol frequencies."
        )
        desc_lines = wrap_text(desc_raw, font_size=36, max_width=config.frame_width - 1)
        description_group = (
            VGroup(*[Tex(r"\text{" + line + r"}", font_size=36) for line in desc_lines])
            .arrange(DOWN, aligned_edge=LEFT)
            .to_corner(UP)
        )
        self.play(Write(description_group))
        self.wait(7)
        self.play(FadeOut(description_group))
        self.wait()

        # 1.5) Show use cases of Huffman coding as individual bullets
        usecases = [
            "General-purpose file compression (ZIP, GZIP)",
            "Lossless image formats (PNG)",
            "Multimedia codecs (JPEG, MP3)",
            "Communication protocols (IoT, mobile links)",
            "Embedded/hardware systems (ASICs, FPGAs)",
        ]
        usecases_heading = Tex(r"\text{Use Cases:}", font_size=36)
        usecases_heading.to_corner(UL)
        self.play(Write(usecases_heading))
        self.wait(0.5)
        bullets = VGroup()
        previous = usecases_heading
        for u in usecases:
            bullet = Tex(r"\text{• " + u + r"}", font_size=36)
            bullet.next_to(previous, DOWN, aligned_edge=LEFT)
            self.play(Write(bullet))
            self.wait(0.3)
            bullets.add(bullet)
            previous = bullet
        self.wait(2)
        self.play(FadeOut(usecases_heading), *[FadeOut(b) for b in bullets])
        self.wait()

        # 2) Steps of the algorithm
        steps_title = Tex(r"\text{Steps of the algorithm:}", font_size=36)
        steps_title.to_corner(UL)
        self.play(Write(steps_title))
        self.wait(0.5)
        previous = steps_title

        steps = [
            "1. List all symbols with their appearance probabilities.",
            "2. Merge the two symbols with the lowest probabilities into one combined symbol.",
            "3. Repeat merging these lowest-probability symbols until only two remain.",
            "4. Assign 0 to the first of these two symbols and 1 to the second.",
            "5. Split each combined symbol, appending 0 for the first part and 1 for the second, until every original symbol has its code.",
        ]
        colors = [BLUE, GREEN, YELLOW, RED, ORANGE]
        steps_group = VGroup()
        for text, color in zip(steps, colors):
            lines = wrap_text(text, font_size=36, max_width=config.frame_width - 1)
            bullet_num = text.split()[0]
            tex_objects = []
            # Split first line into number and content to color the number
            first_line = lines[0]
            if first_line.startswith(bullet_num):
                rest = first_line[len(bullet_num) :].lstrip()
                num_tex = Tex(r"\text{" + bullet_num + r"}", font_size=36).set_color(
                    color
                )
                content_tex = Tex(r"\text{" + rest + r"}", font_size=36)
                first_group = VGroup(num_tex, content_tex).arrange(
                    RIGHT, aligned_edge=DOWN
                )
                tex_objects.append(first_group)
            else:
                tex_objects.append(Tex(r"\text{" + first_line + r"}", font_size=36))
            # Remaining wrapped lines
            for line in lines[1:]:
                tex_objects.append(Tex(r"\text{" + line + r"}", font_size=36))
            lines_group = VGroup(*tex_objects).arrange(DOWN, aligned_edge=LEFT)
            lines_group.next_to(previous, DOWN, aligned_edge=LEFT)
            self.play(Write(lines_group))
            self.wait(1)
            steps_group.add(lines_group)
            previous = lines_group
        self.wait(2)
        self.play(FadeOut(steps_title), *[FadeOut(s) for s in steps_group])
        self.wait()

        # Helpers for merge and split steps remain unchanged
        def show_merge_step(
            old_entries,
            old_probs,
            new_entries,
            new_probs,
            merge_cols,
            merged_idx,
            desc_text,
        ):
            top = Table(
                [old_entries, old_probs],
                row_labels=[Tex("Symbol"), Tex("Prob")],
                include_outer_lines=True,
            )
            top.scale(0.8).to_edge(UP)
            self.play(Create(top))
            self.wait(1)
            desc = Tex(r"\text{" + desc_text + r"}", font_size=36).to_edge(DOWN)
            self.play(Write(desc))
            self.wait(1)
            bot = Table(
                [new_entries, new_probs],
                row_labels=[Tex("Symbol"), Tex("Prob")],
                include_outer_lines=True,
            )
            bot.scale(0.8).next_to(top, DOWN, buff=1)
            y0 = top.get_bottom()[1] - 0.1
            starts = []
            for col in merge_cols:
                cell = top.get_cell((1, col + 2))
                starts.append(np.array([cell.get_center()[0], y0, 0]))
            merged_cell = bot.get_cell((1, merged_idx + 2))
            end = merged_cell.get_top()
            arrows = [Arrow(start, end, buff=0) for start in starts]
            self.play(Create(bot), *[Create(arr) for arr in arrows])
            self.wait(2)
            self.play(
                FadeOut(top),
                FadeOut(bot),
                FadeOut(desc),
                *[FadeOut(arr) for arr in arrows],
            )

        def show_split_step(
            top_entries,
            top_codes,
            bot_entries,
            bot_codes,
            parent_idx,
            child_idxs,
            desc_text,
        ):
            top = Table(
                [top_entries, top_codes],
                row_labels=[Tex("Symbol"), Tex("Code")],
                include_outer_lines=True,
            )
            top.scale(0.8).to_edge(UP)
            self.play(Create(top))
            self.wait(1)
            desc = Tex(r"\text{" + desc_text + r"}", font_size=36).to_edge(DOWN)
            self.play(Write(desc))
            self.wait(1)
            bot = Table(
                [bot_entries, bot_codes],
                row_labels=[Tex("Symbol"), Tex("Code")],
                include_outer_lines=True,
            )
            bot.scale(0.8).next_to(top, DOWN, buff=1)
            y0 = top.get_bottom()[1] - 0.1
            pc = top.get_cell((1, parent_idx + 2))
            start = np.array([pc.get_center()[0], y0, 0])
            ends = [
                cell.get_top()
                for cell in (bot.get_cell((1, idx + 2)) for idx in child_idxs)
            ]
            arrows = [Arrow(start, end, buff=0) for end in ends]
            self.play(Create(bot), *[Create(arr) for arr in arrows])
            self.wait(2)
            self.play(
                FadeOut(top),
                FadeOut(bot),
                FadeOut(desc),
                *[FadeOut(arr) for arr in arrows],
            )

        # Example merging and splitting
        symbols = ["M", "U", "D", "R", "Y"]
        probs = [0.25, 0.25, 0.20, 0.15, 0.15]

        show_merge_step(
            symbols,
            [str(p) for p in probs],
            ["M", "U", "D", "RY"],
            ["0.25", "0.25", "0.20", "0.30"],
            merge_cols=(3, 4),
            merged_idx=3,
            desc_text="Select the two smallest probabilities (R, Y) and merge into RY (0.15 + 0.15 = 0.30).",
        )
        show_merge_step(
            ["M", "U", "D", "RY"],
            ["0.25", "0.25", "0.20", "0.30"],
            ["MD", "U", "RY"],
            ["0.45", "0.25", "0.30"],
            merge_cols=(0, 2),
            merged_idx=0,
            desc_text="Next, merge M and D into MD (0.25 + 0.20 = 0.45).",
        )
        show_merge_step(
            ["MD", "U", "RY"],
            ["0.45", "0.25", "0.30"],
            ["MD", "URY"],
            ["0.45", "0.55"],
            merge_cols=(1, 2),
            merged_idx=1,
            desc_text="Finally, merge U and RY into URY (0.25 + 0.30 = 0.55).",
        )

        show_split_step(
            ["MD", "URY"],
            ["0", "1"],
            ["M", "D", "URY"],
            ["00", "01", "1"],
            parent_idx=0,
            child_idxs=(0, 1),
            desc_text="Assign '0' to branch MD, '1' to URY, splitting MD into M and D.",
        )
        show_split_step(
            ["M", "D", "URY"],
            ["00", "01", "1"],
            ["M", "D", "U", "RY"],
            ["00", "01", "10", "11"],
            parent_idx=2,
            child_idxs=(2, 3),
            desc_text="Split URY: assign '0' to U, '1' to RY, propagating codes.",
        )
        show_split_step(
            ["M", "D", "U", "RY"],
            ["00", "01", "10", "11"],
            ["M", "D", "U", "R", "Y"],
            ["00", "01", "10", "110", "111"],
            parent_idx=3,
            child_idxs=(3, 4),
            desc_text="Split RY: assign '0' to R, '1' to Y, ending the algorithm",
        )

        # Final table
        final = Table(
            [symbols, ["00", "10", "01", "110", "111"]],
            row_labels=[Tex("Symbol"), Tex("Code")],
            include_outer_lines=True,
        )
        final.scale(0.8).to_edge(UP)
        final_desc = Tex(
            r"\text{Final Huffman codes for each symbol.}", font_size=36
        ).next_to(final, DOWN, buff=0.5)
        self.play(Create(final), Write(final_desc))
        self.wait(2)
