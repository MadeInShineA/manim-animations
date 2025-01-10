from manim import *
from manim.utils.color import random_bright_color
import random
import numpy as np


class KMeansClustering(Scene):
    def construct(self):
        k_means_text = Text("K means clustering")
        self.play(Write(k_means_text))
        self.wait()

        self.play(k_means_text.animate.to_edge(UP))

        axes = (
            Axes(
                x_range=[0, 10, 1],
                y_range=[0, 10, 1],
                tips=False,
                axis_config={"include_numbers": True},
            )
            .add_coordinates()
            .next_to(k_means_text, DOWN, buff=0.75)
        ).scale(0.8)

        self.play(Create(axes))
        self.wait(0.1)

        number_of_clusters = 3
        number_of_dots = 50

        dots = []
        random.seed(666)

        for i in range(number_of_dots):
            point_x, point_y, point_z = random.uniform(0, 10), random.uniform(0, 10), 0

            point_coordinates = (point_x, point_y, point_z)
            dot = Dot(axes.c2p(*point_coordinates), radius=0.1)
            dots.append(dot)

            self.add(axes, dot)
            self.wait(0.05)

        self.wait(0.1)

        cluster_dots = []
        available_cluster_colors = [
            # random_bright_color() for _ in range(number_of_clusters)
            RED,
            BLUE,
            GREEN,
        ]

        dots_used_for_cluster = set()

        cluster_text = Text(
            "Select k random points to promote as clusters", font_size=22
        ).next_to(k_means_text, DOWN, buff=0.2)
        self.play(Write(cluster_text))
        self.wait()

        for i in range(number_of_clusters):
            dot = random.choice(dots)

            while dot in dots_used_for_cluster:
                dot = random.choice(dots)

            dots_used_for_cluster.add(dot)

            cluster_color = available_cluster_colors.pop()

            cluster_dot = Dot(dot.get_center(), radius=0.2)
            self.add(axes, cluster_dot)
            self.play(cluster_dot.animate.set_color(cluster_color))

            cluster_dots.append(cluster_dot)

        number_of_itterations = 3
        self.play(FadeOut(cluster_text))

        point_coloring_text = Text(
            "Assign each point to its closest cluster", font_size=22
        ).next_to(k_means_text, DOWN, buff=0.2)
        self.play(Write(point_coloring_text))

        cluster_dots = sorted(cluster_dots, key=lambda p: p.get_center()[0])
        dots = sorted(dots, key=lambda p: p.get_center()[0])

        for itteration in range(number_of_itterations):
            point_num_coordinates_sum_per_cluster = {
                cluster: {"point_number": 1.0, "coordinates_sum": cluster.get_center()}
                for cluster in cluster_dots
            }
            for dot in dots:
                dot_center = dot.get_center()

                distances_to_clusters = [
                    np.linalg.norm(
                        np.array(cluster.get_center()) - np.array(dot_center)
                    )
                    for cluster in cluster_dots
                ]

                closest_cluster_index = np.argmin(distances_to_clusters)
                closest_cluster = cluster_dots[closest_cluster_index]

                point_num_coordinates_sum_per_cluster[closest_cluster][
                    "point_number"
                ] += 1
                point_num_coordinates_sum_per_cluster[closest_cluster][
                    "coordinates_sum"
                ] += dot_center

                if tuple(dot_center) not in [
                    tuple(cluster_dot.get_center()) for cluster_dot in cluster_dots
                ]:
                    line_to_draw = Line(
                        closest_cluster.get_center(),
                        dot_center,
                        color=closest_cluster.get_color(),
                    )
                    self.play(Create(line_to_draw), run_time=0.01)

                    self.play(
                        dot.animate.set_color(closest_cluster.get_color()),
                        run_time=0.05,
                    )

                    self.play(FadeOut(line_to_draw), run_time=0.01)
                else:
                    self.play(
                        dot.animate.set_color(closest_cluster.get_color()),
                        run_time=0.01,
                    )

            clusters_moved = False

            if itteration == 0:
                move_cluster_center_text = Text(
                    "Move the cluster points to their new center", font_size=22
                ).next_to(k_means_text, DOWN, buff=0.2)

                self.play(FadeOut(point_coloring_text))
                self.play(Write(move_cluster_center_text))
                self.wait()

            for cluster_dot in cluster_dots:
                new_coordinates = (
                    point_num_coordinates_sum_per_cluster[cluster_dot][
                        "coordinates_sum"
                    ]
                    / point_num_coordinates_sum_per_cluster[cluster_dot]["point_number"]
                )

                if tuple(new_coordinates) != tuple(cluster_dot.get_center()):
                    clusters_moved = True
                    self.play(cluster_dot.animate.move_to(new_coordinates))
                    self.wait(0.1)

            if itteration == 0:
                repeat_text = Text(
                    "Repeat until the clusters stop moving or the number of iteration wanted was reached",
                    font_size=22,
                ).next_to(k_means_text, DOWN, buff=0.2)
                self.play(FadeOut(move_cluster_center_text))
                self.play(Write(repeat_text))
                self.wait()

            if not clusters_moved:
                break

        self.wait()
