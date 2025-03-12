from math import sin, cos
import cmath
from manim import *
import numpy as np

NUM_OBSERVATIONS: int = 300


class DFT(Scene):
    def construct(self):
        signal = [sin(x / 20) + 2 * cos(x / 20) for x in range(NUM_OBSERVATIONS)]
        N = len(signal)

        base = [
            [cmath.exp(-1j * (2 * cmath.pi / N * n * k)) for n in range(N)]
            for k in range(N)
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
                    x_range=[0, N // 2, N // 20],  # Fix frequency range
                    y_range=[0, max(np.abs(signal_in_frequency_domain[: N // 2]))],
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

        freq_resolution = 1 / NUM_OBSERVATIONS

        frequency_signal_points = VGroup()

        for k in range(N // 2):
            freq = k * freq_resolution  # Convert index to actual frequency
            value = np.abs(signal_in_frequency_domain[k])  # Magnitude spectrum

            dot = Dot(frequency_axis.c2p(k, value, 0), radius=0.02, color=BLUE)
            frequency_signal_points.add(dot)

        self.play(Create(frequency_signal_points), run_time=2.0)

        self.wait(0.5)


NUM_OBSERVATIONS: int = 300


class DFT2(Scene):
    def construct(self):
        signal = [sin(x / 20) + 2 * cos(x / 20) for x in range(NUM_OBSERVATIONS)]
        N = len(signal)

        base = np.array(
            [
                [cmath.exp(-1j * (2 * cmath.pi / N * n * k)) for n in range(N)]
                for k in range(N)
            ]
        )

        self.wait(0.1)

        # Compute the DFT correctly
        signal_in_frequency_domain = np.dot(base, signal)

        dft_text = Text("Discrete Fourier Transform (DFT)")
        self.play(Write(dft_text))
        self.wait()

        self.play(dft_text.animate.to_edge(UP))

        # Time domain signal plot
        time_axes = (
            Axes(
                x_range=[0, NUM_OBSERVATIONS, NUM_OBSERVATIONS // 10],
                y_range=[-3, 3, 1],
                tips=False,
                axis_config={"include_numbers": True},
            )
            .add_coordinates()
            .next_to(dft_text, DOWN, buff=0.75)
            .scale(0.5)
            .shift(UP * 2)
        )

        self.play(Create(time_axes))
        self.wait(0.5)

        time_signal_points = VGroup()
        for t, sig in enumerate(signal):
            dot = Dot(time_axes.c2p(t, sig, 0), radius=0.02, color=RED)
            time_signal_points.add(dot)

        self.play(Create(time_signal_points), run_time=2.0)
        self.wait(0.5)

        # Frequency domain plot
        max_freq_magnitude = max(
            np.abs(signal_in_frequency_domain[: N // 2])
        )  # Fixed max magnitude
        frequency_axis = (
            Axes(
                x_range=[0, N // 2, N // 20],  # Fix frequency range
                y_range=[0, max_freq_magnitude],  # Fix y range based on magnitude
                tips=False,
                axis_config={"include_numbers": True},
            )
            .add_coordinates()
            .next_to(time_axes, DOWN, buff=0.2)
            .scale(0.5)
        )

        self.play(Create(frequency_axis))
        self.wait(0.5)

        frequency_signal_points = VGroup()
        for k in range(N // 2):
            value = np.abs(signal_in_frequency_domain[k])  # Magnitude spectrum
            dot = Dot(frequency_axis.c2p(k, value, 0), radius=0.02, color=BLUE)
            frequency_signal_points.add(dot)

        self.play(Create(frequency_signal_points), run_time=2.0)
        self.wait(0.5)
