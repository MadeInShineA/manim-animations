from manim import *


class DFS(Scene):
    def construct(self):
        self.camera.background_color = GRAY_A

        graph_dict = {
            "2_depth_0": ["2_depth_1", "-2_depth_1"],  # Use list instead of set
            "2_depth_1": ["4_depth_2", "2_depth_2"],
            "-2_depth_1": ["-4_depth_2", "-2_depth_2"],
            "4_depth_2": [],
            "2_depth_2": [],
            "-2_depth_2": [],
            "-4_depth_2": [],
        }

        visited = []

        def dfs(vertex, parent):
            visited.append((vertex, parent))
            for neighbor in graph_dict[vertex]:
                if neighbor not in visited:
                    dfs(neighbor, vertex)

        dfs("2_depth_0", None)

        vertices = [node for node, parent in visited]
        edges = [(parent, node) for node, parent in visited if parent is not None]

        even_depth_nodes_colors = {
            node: "WHITE" for node in graph_dict.keys() if int(node[-1]) % 2 == 0
        }
        odd_depth_nodes_colors = {
            node: "BLACK" for node in graph_dict.keys() if int(node[-1]) % 2 != 0
        }

        vertices_colors = even_depth_nodes_colors | odd_depth_nodes_colors
        vertices_labels = {
            node: node[0] if node[0] != "-" else node[0:2] for node in graph_dict.keys()
        }

        vertex_config = {
            vertex: {
                "fill_color": vertices_colors[vertex],
                "label": vertices_labels[vertex],
                "label_color": "WHITE"
                if vertices_colors[vertex] == "BLACK"
                else "BLACK",
            }
            for vertex in vertices
        }

        manim_graph = DiGraph(
            vertices,
            edges,
            layout="tree",
            root_vertex="2_depth_0",
            vertex_config={
                v: {"fill_color": vertex_config[v]["fill_color"]} for v in vertex_config
            },
            labels={
                v: Text(
                    vertex_config[v]["label"], color=vertex_config[v]["label_color"]
                )
                for v in vertex_config
            },
        )

        self.play(Create(manim_graph))
        self.wait()
