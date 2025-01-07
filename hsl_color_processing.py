from manim import *


class HSLColorProcessing(Scene):
    def construct(self):
        color_hex = "#00FFFF"
        color = ManimColor.from_hex(color_hex)
        text = Text(color_hex)

        rectangle = SurroundingRectangle(text, color=color, buff=0.2).set_fill(
            opacity=1.0
        )
        rectangle.set_z_index(-1)

        self.play(Write(text))
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

        self.wait()


class DrawHueCircle(Scene):
    def construct(self):
        theta = ValueTracker(0)

        angle = always_redraw(
            lambda: Arc(
                radius=0.3,
                angle=theta.get_value(),
                arc_center=ORIGIN,
                color=YELLOW,
            )
        )

        circle = always_redraw(
            lambda: Arc(
                radius=1.0,
                angle=theta.get_value(),
                arc_center=ORIGIN,
                color=ManimColor.from_hsv(np.array([theta.get_value() / 2 / PI, 1, 1])),
            )
        )

        angle_line = always_redraw(
            lambda: Line(
                start=ORIGIN,
                end=RIGHT * np.cos(theta.get_value()) + UP * np.sin(theta.get_value()),
            )
        )

        text = always_redraw(
            lambda: Text(str(round(theta.get_value() * 180 / PI, 2)) + " °").next_to(
                ORIGIN, DOWN, buff=1.5
            )
        )

        radius_line = Line(ORIGIN, np.array([1, 0, 0]))

        self.add(text, circle, angle, angle_line, radius_line)
        self.play(theta.animate(rate_func=linear, run_time=1).set_value(2 * PI))

        self.wait()
