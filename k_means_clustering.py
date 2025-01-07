from __future__ import absolute_import
from manim import *
from manim.utils.color import random_bright_color
import random
import numpy as np

from networkx.algorithms.bipartite.cluster import clustering


class KMeansClustering(Scene):
    def construct(self):
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            tips=False,
            axis_config={"include_numbers": True},
        ).add_coordinates()

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
            self.wait(0.1)

        self.wait(0.1)

        cluster_dots = []
        available_cluster_colors = [
            # random_bright_color() for _ in range(number_of_clusters)
            RED,
            BLUE,
            GREEN,
        ]

        dots_used_for_cluster = set()

        for i in range(number_of_clusters):
            dot = random.choice(dots)

            while dot in dots_used_for_cluster:
                dot = random.choice(dots)

            dots_used_for_cluster.add(dot)

            cluster_color = available_cluster_colors.pop()

            cluster_dot = Dot(dot.get_center(), radius=0.3)
            self.add(axes, cluster_dot)
            self.play(cluster_dot.animate.set_color(cluster_color))

            cluster_dots.append(cluster_dot)

        number_of_itterations = 3

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
                    self.play(Create(line_to_draw), run_time=0.05)

                    self.play(
                        dot.animate.set_color(closest_cluster.get_color()),
                        run_time=0.05,
                    )

                    self.play(FadeOut(line_to_draw), run_time=0.05)
                else:
                    self.play(
                        dot.animate.set_color(closest_cluster.get_color()),
                        run_time=0.05,
                    )

            clusters_moved = False
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

            if not clusters_moved:
                break

        self.wait()
