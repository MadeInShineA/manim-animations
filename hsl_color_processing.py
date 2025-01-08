from manim import *


class HSLColorProcessing(Scene):
    def draw_hue_circle(self, position_item, position_direction, position_buff):
        radius = 0.5
        lines = 120
        hue_circle = VGroup(
            *[
                Polygon(
                    ORIGIN,
                    radius
                    * np.array(
                        [
                            np.cos(2 * PI * (i + 1.05) / lines),
                            np.sin(2 * PI * (i + 1.05) / lines),
                            0,
                        ]
                    ),
                    radius
                    * np.array(
                        [np.cos(2 * PI * (i) / lines), np.sin(2 * PI * (i) / lines), 0]
                    ),
                    stroke_width=0,
                    stroke_opacity=0,
                    fill_opacity=1,
                    fill_color=ManimColor.from_hsv([i / lines, 1.0, 1.0]),
                )
                for i in range(lines)
            ]
        )

        hue_circle.next_to(position_item, position_direction, buff=position_buff)

        counter = DecimalNumber(0, edge_to_fix=([0, 0, 0])).to_edge(DOWN)

        animation_time = 3
        counter.add_updater(lambda m, dt: m.increment_value(dt * 360 / animation_time))
        self.add(counter)
        self.play(
            Create(hue_circle), run_time=animation_time, rate_func=rate_functions.linear
        )
        counter.suspend_updating()

        counter.set_value(360)
        self.wait()

        return hue_circle, counter

    def hsl_to_hsv(self, h, s, l):
        if l == 0:
            return h, 0, 0
        elif l == 1:
            return h, 0, 1
        else:
            v = l + s * min(l, 1 - l)
            s_hsv = 2 * (1 - l / v) if v != 0 else 0
            return h, s_hsv, v

    def hsv_to_hsl(self, h, s, v):
        l = v * (1 - s / 2)
        if l == 0 or l == 1:
            s_hsl = 0
        else:
            s_hsl = (v - l) / min(l, 1 - l)

        return h, s_hsl, l

    def draw_s_or_l_rectangle(
        self,
        original_color,
        position_item,
        position_direction,
        position_buff,
        draw_s=False,
        draw_l=False,
    ):
        width = 4.0
        height = 1.0
        rectangles = 400
        color_h, color_s, color_v = original_color.to_hsv()
        hsl_h, hsl_s, hsl_l = self.hsv_to_hsl(color_h, color_s, color_v)

        rectangle = VGroup(
            *[
                Rectangle(
                    width=width / rectangles,
                    height=height,
                    stroke_width=0,
                    stroke_opacity=0,
                    fill_opacity=1,
                    fill_color=ManimColor.from_hsv(
                        self.hsl_to_hsv(
                            hsl_h,
                            i / rectangles if draw_s else hsl_s,
                            i / rectangles if draw_l else hsl_l,
                        )
                    ),
                ).shift(RIGHT * (width / rectangles * i))
                for i in range(rectangles)
            ]
        )

        rectangle.next_to(position_item, position_direction, buff=position_buff)

        counter = DecimalNumber(0, edge_to_fix=[0, 0, 0]).to_edge(DOWN)

        animation_time = 3
        counter.add_updater(lambda m, dt: m.increment_value(dt / animation_time))
        self.add(counter)
        self.play(
            Create(
                rectangle,
                run_time=animation_time,
                rate_func=rate_functions.linear,
            )
        )
        counter.suspend_updating()
        counter.set_value(1.0)

        self.wait()

        return rectangle, counter

    def construct(self):
        color_hex = "#00FFFF"
        color = ManimColor.from_hex(color_hex)
        color_hex_text = Text(color_hex)

        rectangle = SurroundingRectangle(
            color_hex_text, color=color, buff=0.2
        ).set_fill(opacity=1.0)
        rectangle.set_z_index(-1)

        self.play(Write(color_hex_text))
        self.play(DrawBorderThenFill(rectangle))

        h_text = Text("H")
        s_text = Text("S")
        l_text = Text("L")

        hsl_text = VGroup(h_text, s_text, l_text)
        hsl_text.arrange(RIGHT, buff=0.5).next_to(rectangle, DOWN, buff=1.5)

        left_arrow = Arrow(start=rectangle.get_bottom(), end=h_text.get_top())
        down_arrow = Arrow(start=rectangle.get_bottom(), end=s_text.get_top())
        right_arrow = Arrow(start=rectangle.get_bottom(), end=l_text.get_top())

        self.play(Create(left_arrow))
        self.play(Create(h_text))

        self.play(Create(down_arrow))
        self.play(Create(s_text))

        self.play(Create(right_arrow))
        self.play(Create(l_text))

        items_to_fade_out = [
            color_hex_text,
            rectangle,
            left_arrow,
            down_arrow,
            right_arrow,
        ]

        self.play([FadeOut(item) for item in items_to_fade_out])

        hsl_text.generate_target()
        hsl_text.target.shift(UP * 5)
        self.play(MoveToTarget(hsl_text))

        self.play(s_text.animate.set_color(GRAY), l_text.animate.set_color(GRAY))
        hue_text = Text(
            "H represents the hue angle of the color, ranging from 0° (red) to 360° (back to red).",
            font_size=24,
        )
        self.play(Write(hue_text))
        hue_circle, counter = self.draw_hue_circle(hue_text, DOWN, 1.0)
        self.wait()

        self.play(FadeOut(hue_text, hue_circle, counter))

        self.play(h_text.animate.set_color(GRAY), s_text.animate.set_color(WHITE))
        saturation_text = Text(
            "S represents the saturation of the color, ranging from 0 (gray) to 1 (full color).",
            font_size=24,
        )
        self.play(Write(saturation_text))
        saturation_rectangle, counter = self.draw_s_or_l_rectangle(
            color, saturation_text, DOWN, 1.0, draw_s=True
        )

        self.play(FadeOut(saturation_text, saturation_rectangle, counter))
        self.play(s_text.animate.set_color(GRAY), l_text.animate.set_color(WHITE))
        lightness_text = Text(
            "L represents the lightness of the color, from 0 (black) to 1 (white).",
            font_size=24,
        )

        self.play(Write(lightness_text))
        lightness_rectangle, counter = self.draw_s_or_l_rectangle(
            color, lightness_text, DOWN, 1.0, draw_l=True
        )

        self.play(FadeOut(lightness_text, lightness_rectangle, counter))
        self.play(l_text.animate.set_color(GRAY))

        self.wait()


class DrawHueCircle(Scene):
    def construct(self):
        radius = 1.5
        lines = 120
        wheel = VGroup(
            *[
                Polygon(
                    ORIGIN,
                    radius
                    * np.array(
                        [
                            np.cos(2 * PI * (i + 1.05) / lines),
                            np.sin(2 * PI * (i + 1.05) / lines),
                            0,
                        ]
                    ),
                    radius
                    * np.array(
                        [np.cos(2 * PI * (i) / lines), np.sin(2 * PI * (i) / lines), 0]
                    ),
                    stroke_width=0,
                    stroke_opacity=0,
                    fill_opacity=1,
                    fill_color=ManimColor.from_hsv([i / lines, 1.0, 1.0]),
                )
                for i in range(lines)
            ]
        )
        ctr = DecimalNumber(0).to_edge(DOWN)

        animation_time = 3
        ctr.add_updater(lambda m, dt: m.increment_value(dt * 360 / animation_time))
        self.add(ctr)
        self.play(
            Create(wheel), run_time=animation_time, rate_func=rate_functions.linear
        )
        ctr.suspend_updating()

        ctr.set_value(360)
        self.wait()
