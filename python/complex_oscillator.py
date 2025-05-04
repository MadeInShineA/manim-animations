from math import pi
from manim import *
import cmath


class ComplexOscillator(Scene):
    def construct(self):
        # Axes setup
        plane = ComplexPlane().add_coordinates()
        self.add(plane)

        # Moving dot
        dot = Dot(color=RED)

        # Path tracer
        path = TracedPath(dot.get_center, stroke_color=RED, stroke_width=2)

        self.add(dot, path)

        def get_z_value_at_t(t) -> complex:
            return 2 * cmath.exp((1j * (t + cmath.pi / 4)))

        def get_z_value_at_t_2(t) -> complex:
            return cmath.exp((1j * (2 * t + cmath.pi / 3))) + 3 * cmath.exp(
                (1j * (2 * t + cmath.pi))
            )

        def get_z_value_at_t_3(t) -> complex:
            return cmath.exp((1j * (2 * t + cmath.pi / 3))) + 3 * cmath.exp(
                (1j * (5 * t + cmath.pi))
            )

        def update_dot(m, dt):
            t = self.renderer.time
            z_t = get_z_value_at_t_3(t)
            m.move_to([z_t.real, z_t.imag, 0])  # Move dot to new position

        dot.add_updater(update_dot)
        self.wait(2 * pi)  # Run animation for 8 seconds
