from manim import *

class MinMax(Scene):
    def construct(self):
        self.camera.background_color = GRAY_A

        graph_dict = {
            "2_depth_0": ["2_depth_1", "-4_depth_1"],  # Use list instead of set
            "2_depth_1": ["4_depth_2", "2_depth_2"],
            "-4_depth_1": ["-4_depth_2", "-2_depth_2"],
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
        vertices_values = {node: int(node.split("_")[0]) for node in graph_dict.keys()}

        vertex_config = {
            vertex: {
                "fill_color": vertices_colors[vertex],
                "value": vertices_values[vertex],
                "label_color": "WHITE"
                if vertices_colors[vertex] == "BLACK"
                else "BLACK",
            }
            for vertex in vertices
        }

        labels = {
            v: Text("?", color=vertex_config[v]["label_color"]) for v in vertex_config
        }

        manim_graph = DiGraph(
            vertices,
            edges=edges[::-1],
            layout="tree",
            root_vertex="2_depth_0",
            vertex_config={
                v: {"fill_color": vertex_config[v]["fill_color"]} for v in vertex_config
            },
            labels=labels,
        )

        def all_descendants_added(vertice, added_vertices, graph_dict):
            children = graph_dict[vertice]
            if not children:
                return True
            return all(
                child in added_vertices
                and all_descendants_added(child, added_vertices, graph_dict)
                for child in children
            )

        def change_label_to_value(vertice):
            old_label = labels[vertice]

            new_label = Text(
                str(vertex_config[vertice]["value"]),
                color=vertex_config[vertice]["label_color"],
            )
            new_label.move_to(manim_graph[vertice].get_center())
            labels[vertice] = new_label
            self.play(Transform(old_label, new_label))

        added_vertices = set()

        for vertice, parent in visited:
            if parent:
                self.play(Create(manim_graph.edges[(parent, vertice)][0]))
                self.wait(0.2)
            self.play(Create(manim_graph.vertices[vertice][0]))
            self.wait(0.2)

            added_vertices.add(vertice)

            if not graph_dict[vertice]:
                change_label_to_value(vertice)

            if parent and all_descendants_added(parent, added_vertices, graph_dict):
                change_label_to_value(parent)

        change_label_to_value("2_depth_0")
        self.wait()
