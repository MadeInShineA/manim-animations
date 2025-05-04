from math import sin, pi
import cmath
from manim import *
import numpy as np

NUM_OBSERVATIONS: int = 1000


class DFT(Scene):
    def construct(self):
        t = np.arange(0, 1, 1 / NUM_OBSERVATIONS)

        f1, f2, f3 = 5, 50, 120  # Frequencies in Hz
        A1, A2, A3 = 1.0, 0.5, 0.3  # Amplitudes

        signal = (
            A1 * np.sin(2 * np.pi * f1 * t)
            + A2 * np.sin(2 * np.pi * f2 * t)
            + A3 * np.sin(2 * np.pi * f3 * t)
        )

        base = [
            [
                cmath.exp(-1j * (2 * cmath.pi / NUM_OBSERVATIONS * n * k))
                for n in range(NUM_OBSERVATIONS)
            ]
            for k in range(NUM_OBSERVATIONS)
        ]

        self.wait(0.1)

        signal_in_frequency_domain = np.dot(base, signal).real

        dft_text = Text("Discrete Fourier Transform (DFT)")
        self.play(Write(dft_text))
        self.wait()

        self.play(dft_text.animate.to_edge(UP))

        time_axes = (
            (
                Axes(
                    x_range=[0, NUM_OBSERVATIONS, NUM_OBSERVATIONS // 10],
                    y_range=[-3, 3, 1],
                    tips=False,
                    axis_config={"include_numbers": True},
                )
                .add_coordinates()
                .next_to(dft_text, DOWN, buff=0.75)
            )
            .scale(0.5)
            .shift(UP * 2)
        )

        self.play(Create(time_axes))
        self.wait(0.5)

        time_signal_points = VGroup()

        for t, signal in enumerate(signal):
            dot = Dot(time_axes.c2p(t, signal, 0), radius=0.02, color=RED)
            time_signal_points.add(dot)

        self.play(Create(time_signal_points), run_time=2.0)
        self.wait(0.5)

        frequency_axis = (
            (
                Axes(
                    x_range=[0, 500, 50],  # Fix frequency range
                    y_range=[0, 500, 100],
                    tips=False,
                    axis_config={"include_numbers": True},
                )
                .add_coordinates()
                .next_to(time_axes, DOWN, buff=0.75)
            )
            .scale(0.5)
            .shift(UP * 2)
        )

        self.play(Create(frequency_axis))
        self.wait(0.5)

        frequency_signal_points = VGroup()
        frequency_signal_points = VGroup()

        for k in range(NUM_OBSERVATIONS):
            value = np.abs(signal_in_frequency_domain[k])  # Magnitude spectrum

            dot = Dot(frequency_axis.c2p(k, value, 0), radius=0.02, color=BLUE)
            frequency_signal_points.add(dot)

        self.play(Create(frequency_signal_points), run_time=2.0)

        self.wait(0.5)
