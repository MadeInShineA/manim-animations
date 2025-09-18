from manim import *
import numpy as np
import random
import re


class LinearRegression(Scene):
    def construct(self):
        # ---------------- Timing knobs (slower steps) ----------------
        T_FOCUS = 0.80  # focus_on transition (slower)
        T_RESTORE = 0.60  # restore_all transition (slower)
        T_NUM = 1.00  # number updates (a,b,grads,cost) (slower)
        T_LINE = 0.80  # draw line (slower)
        T_RES = 0.80  # residuals update (slower)
        T_COST = 0.80  # cost update (slower)
        T_SEG = 0.50  # MSE segment draw (slower)
        T_HIST = 0.60  # histogram transform (slower)
        T_WAIT = 0.50  # small pause between logical steps (longer)

        # ---------------- Title ----------------
        title = Tex("Linear Regression (Gradient Descent)", font_size=48)
        self.play(FadeIn(title), run_time=0.9)
        self.play(title.animate.to_edge(UP, buff=0.06), run_time=0.6)

        # ---------------- Math-aware wrap ($...$ safe) ----------------
        def wrap_text_tex(text: str, font_size: int, max_width: float) -> VGroup:
            segments = re.split(r"(\$[^$]*\$)", text)  # keep $...$ groups
            tokens: list[str] = []
            for seg in segments:
                if not seg:
                    continue
                if seg.startswith("$") and seg.endswith("$"):
                    tokens.append(seg)
                else:
                    tokens.extend(seg.split(" "))
            lines: list[str] = []
            cur = ""
            for tok in tokens:
                cand = (cur + " " + tok).strip() if cur else tok
                if cur and Tex(cand, font_size=font_size).width > max_width:
                    lines.append(cur)
                    cur = tok
                else:
                    cur = cand
            if cur:
                lines.append(cur)
            return VGroup(*[Tex(ln, font_size=font_size) for ln in lines]).arrange(
                DOWN, aligned_edge=LEFT, buff=0.28
            )

        # ===================== Screen 1: What is Linear Regression? =====================
        desc_heading = Tex("What is Linear Regression?", font_size=40)
        desc_heading.next_to(title, DOWN, buff=0.5).to_edge(LEFT, buff=0.5)
        self.play(Write(desc_heading), run_time=1.2)

        desc_text = (
            "Linear regression finds the best straight line through data points to predict one variable from another. "
            r"Given data pairs $(x_i, y_i)$, we find the line $\hat{y} = ax + b$ that minimizes prediction errors. "
            "This makes it one of the most fundamental tools in statistics and machine learning."
        )
        desc_group = wrap_text_tex(desc_text, 34, config.frame_width - 1)
        desc_group.next_to(desc_heading, DOWN, buff=0.4).to_edge(LEFT, buff=0.5)
        self.play(Write(desc_group), run_time=3.5)
        self.wait(2.5)
        self.play(FadeOut(desc_heading), FadeOut(desc_group), run_time=1.0)

        # ===================== Screen 2: Real-World Applications =====================
        usecases_heading = Tex("Real-World Applications", font_size=40)
        usecases_heading.next_to(title, DOWN, buff=0.5).to_edge(LEFT, buff=0.5)
        self.play(Write(usecases_heading), run_time=1.2)

        usecases_intro = "Linear regression powers countless applications across industries:"
        intro_text = Tex(usecases_intro, font_size=32)
        intro_text.next_to(usecases_heading, DOWN, buff=0.3).to_edge(LEFT, buff=0.5)
        self.play(Write(intro_text), run_time=1.8)

        usecases = [
            r"\textbullet{} \textbf{Finance:} Stock price prediction, risk assessment",
            r"\textbullet{} \textbf{Healthcare:} Drug dosage optimization, disease progression",
            r"\textbullet{} \textbf{Marketing:} Sales forecasting, customer lifetime value",
            r"\textbullet{} \textbf{Engineering:} System calibration, quality control",
            r"\textbullet{} \textbf{Research:} Experimental analysis, hypothesis testing",
        ]
        bullets = VGroup()
        prev = intro_text
        for i, u in enumerate(usecases):
            b = Tex(u, font_size=30)
            b.next_to(prev, DOWN, buff=0.25, aligned_edge=LEFT)
            self.play(Write(b), run_time=0.9)
            bullets.add(b)
            prev = b

        self.wait(2.0)
        self.play(
            FadeOut(usecases_heading), FadeOut(intro_text), *[FadeOut(b) for b in bullets], run_time=1.0
        )

        # ===================== Screen 3: Gradient Descent Algorithm =====================
        steps_title = Tex("Gradient Descent Algorithm", font_size=40, color=ORANGE)
        steps_title.next_to(title, DOWN, buff=0.5).to_edge(LEFT, buff=0.5)
        self.play(Write(steps_title), run_time=0.8)

        steps_intro = "We'll use gradient descent to find optimal parameters $a$ and $b$:"
        intro_alg = Tex(steps_intro, font_size=32)
        intro_alg.next_to(steps_title, DOWN, buff=0.3).to_edge(LEFT, buff=0.5)
        self.play(Write(intro_alg), run_time=1.0)

        steps = [
            r"1.\ \textbf{Define Model:} $\hat{y} = a x + b$ (linear relationship)",
            r"2.\ \textbf{Set Cost Function:} $J(a,b)=\frac{1}{n}\sum (y_i-\hat{y}_i)^2$ (minimize errors)",
            r"3.\ \textbf{Calculate Gradients:} $\frac{\partial J}{\partial a}$ and $\frac{\partial J}{\partial b}$ (find slope)",
            r"4.\ \textbf{Update Parameters:} $a \leftarrow a-\eta\,\frac{\partial J}{\partial a}$, $b \leftarrow b-\eta\,\frac{\partial J}{\partial b}$",
            r"5.\ \textbf{Repeat:} Until cost $J$ converges to minimum",
        ]
        COLS = [BLUE_C, GREEN_C, YELLOW_C, ORANGE, GREY_B]
        step_group = VGroup()
        prev = intro_alg
        for i, (text, col) in enumerate(zip(steps, COLS)):
            line = Tex(text, font_size=30)
            # Color specific parts of each step
            if i == 0:  # Model
                line.set_color_by_tex(r"\hat{y}", BLUE_C)
                line.set_color_by_tex("Model", BLUE_C)
            elif i == 1:  # Cost function
                line.set_color_by_tex("J", GREEN_C)
                line.set_color_by_tex("Cost Function", GREEN_C)
            elif i == 2:  # Gradients
                line.set_color_by_tex(r"\frac{\partial J}{\partial a}", YELLOW_C)
                line.set_color_by_tex(r"\frac{\partial J}{\partial b}", YELLOW_C)
                line.set_color_by_tex("Gradients", YELLOW_C)
            elif i == 3:  # Updates
                line.set_color_by_tex(r"\eta", ORANGE)
                line.set_color_by_tex(r"\leftarrow", ORANGE)
                line.set_color_by_tex("Parameters", ORANGE)
            elif i == 4:  # Repeat
                line.set_color_by_tex("Repeat", GREY_B)
                line.set_color_by_tex("J", GREEN_C)

            line.next_to(prev, DOWN, buff=0.35, aligned_edge=LEFT)
            self.play(Write(line), run_time=0.8)
            step_group.add(line)
            prev = line

        self.wait(1.5)
        self.play(FadeOut(steps_title), FadeOut(intro_alg), *[FadeOut(s) for s in step_group], run_time=0.8)

        # ===================== Graph Screen — Step label, status, then grid =====================

        # Fixed spacer under title; step label anchored to it
        probe = Tex("Step X: Sample", font_size=32)
        spacer = Rectangle(
            width=0.001, height=probe.height, stroke_width=0, fill_opacity=0
        )
        spacer.next_to(title, DOWN, buff=0.04)
        self.add(spacer)

        # Step label (fixed position)
        step_label = Tex("", font_size=32).move_to(spacer.get_center())
        self.add(step_label)

        def set_step_label(text: str, color):
            step_label.become(
                Tex(text, font_size=32).set_color(color).move_to(spacer.get_center())
            )

        # Status (iteration) anchored UNDER the step label - include iteration, eta, and MSE
        status_probe = Tex(
            r"Iteration 0 $\cdot$ $\eta$ = 0.350 $\cdot$ a = 0.000, b = 0.000 $\cdot$ MSE = 0.000",
            font_size=26,
        )
        status_spacer = Rectangle(
            width=0.001, height=status_probe.height, stroke_width=0, fill_opacity=0
        )
        status_spacer.next_to(spacer, DOWN, buff=0.06)
        self.add(status_spacer)
        status = Tex("", font_size=26).move_to(status_spacer.get_center())
        self.add(status)

        # --- Data
        n = 60
        random.seed(2025)
        np.random.seed(2025)
        true_m, true_b = 0.7, 2.0
        xs = np.random.uniform(0.5, 9.5, size=n)
        noise = np.random.normal(loc=0.0, scale=1.15, size=n)
        ys = true_m * xs + true_b + noise
        ys = np.clip(ys, 0.5, 8.5)

        # --- Math helpers
        def mse(a, b):
            return np.mean((ys - (a * xs + b)) ** 2)

        def residuals(a, b):
            return ys - (a * xs + b)

        x_mu, x_sigma = xs.mean(), xs.std()
        x_scaled = (xs - x_mu) / x_sigma

        def scaled_to_plot(a_s, b_s):
            a = a_s / x_sigma
            b = b_s - (a_s * x_mu / x_sigma)
            return a, b

        def grads_scaled(a_s, b_s):
            yhat = a_s * x_scaled + b_s
            err = ys - yhat
            n_ = len(xs)
            da = -(2.0 / n_) * np.sum(x_scaled * err)
            db = -(2.0 / n_) * np.sum(err)
            return da, db

        def grads_display(a, b):
            yhat = a * xs + b
            err = ys - yhat
            n_ = len(xs)
            da = -(2.0 / n_) * np.sum(xs * err)
            db = -(2.0 / n_) * np.sum(err)
            return da, db

        # Seed params for initial scales
        a_s, b_s = -0.2, 0.8
        a_plot, b_plot = scaled_to_plot(a_s, b_s)
        mse0 = mse(a_plot, b_plot)

        # Histogram scale based on initial residuals
        def hist_counts_edges(a, b, bins=10):
            r = residuals(a, b)
            return np.histogram(r, bins=bins, range=(-4, 4))

        counts0, edges0 = hist_counts_edges(a_plot, b_plot, bins=10)
        hist_ymax = max(20, int(np.ceil(counts0.max() * 1.2)))
        hist_step = max(1, int(np.ceil(hist_ymax / 5)))

        # --- Top-left: Graph (maximize size)
        axes = (
            Axes(
                x_range=[0, 10, 1],
                y_range=[0, 10, 1],
                tips=False,
                axis_config={
                    "include_numbers": True,
                    "font_size": 26,
                    "stroke_width": 2,  # Slightly thicker axis lines
                },
            )
            .add_coordinates()
            .scale(1.3)  # Much bigger
        )
        x_label = Tex(r"x", font_size=32)
        y_label = Tex(r"y", font_size=32).rotate(PI / 2)
        x_label.next_to(axes.x_axis, DOWN, buff=0.2)
        y_label.next_to(axes.y_axis, LEFT, buff=0.2)
        axes_labels = VGroup(x_label, y_label)
        dots = VGroup(
            *[
                Dot(axes.c2p(x, y), radius=0.065, color=WHITE, fill_opacity=0.8)
                for x, y in zip(xs, ys)
            ]
        )
        graph_cell = VGroup(axes, axes_labels, dots)

        def line_from_params(m, b, color=ORANGE, width=6):
            x0, x1 = axes.x_range[0], axes.x_range[1]
            return Line(
                axes.c2p(x0, m * x0 + b),
                axes.c2p(x1, m * x1 + b),
                color=color,
                stroke_width=width,
            )

        def residual_segments(m, b):
            g = VGroup()
            for x, y in zip(xs, ys):
                yhat = m * x + b
                a = axes.c2p(x, min(y, yhat))
                bb = axes.c2p(x, max(y, yhat))
                g.add(Line(a, bb, color=YELLOW_C, stroke_width=1.2, stroke_opacity=0.9))
            return g

        # --- Top-right: Formulas (larger and centered vertically in their cell)
        yhat_left = MathTex(r"\hat{y} = ", font_size=58).set_color(BLUE_C)
        a_val = DecimalNumber(-0.20, num_decimal_places=3, color=BLUE_C, font_size=58)
        ax_tex = MathTex(r"x + ", font_size=58).set_color(BLUE_C)
        b_val = DecimalNumber(0.80, num_decimal_places=3, color=BLUE_C, font_size=58)
        yhat_group = VGroup(yhat_left, a_val, ax_tex, b_val).arrange(RIGHT, buff=0.10)

        cost_left = MathTex(
            r"J(a,b) = \frac{1}{n}\sum_{i=1}^{n}\big(y_i - (a x_i + b)\big)^2",
            font_size=52,
        ).set_color(GREEN_C)
        cost_eq = MathTex(r"=", font_size=52).set_color(GREEN_C)
        cost_val = DecimalNumber(0.0, num_decimal_places=3, color=GREEN_C, font_size=52)
        cost_group = VGroup(cost_left, cost_eq, cost_val).arrange(RIGHT, buff=0.10)

        grad_a_left = MathTex(
            r"\frac{\partial J}{\partial a} = -\frac{2}{n}\sum x_i\big(y_i - (a x_i + b)\big)",
            font_size=48,
        ).set_color(YELLOW_C)
        grad_a_eq = MathTex(r"=", font_size=48).set_color(YELLOW_C)
        grad_a_val = DecimalNumber(
            0.0, num_decimal_places=3, color=YELLOW_C, font_size=48
        )
        grad_a_group = VGroup(grad_a_left, grad_a_eq, grad_a_val).arrange(
            RIGHT, buff=0.10
        )

        grad_b_left = MathTex(
            r"\frac{\partial J}{\partial b} = -\frac{2}{n}\sum \big(y_i - (a x_i + b)\big)",
            font_size=48,
        ).set_color(YELLOW_C)
        grad_b_eq = MathTex(r"=", font_size=48).set_color(YELLOW_C)
        grad_b_val = DecimalNumber(
            0.0, num_decimal_places=3, color=YELLOW_C, font_size=48
        )
        grad_b_group = VGroup(grad_b_left, grad_b_eq, grad_b_val).arrange(
            RIGHT, buff=0.10
        )

        formulas_col = VGroup(
            yhat_group, cost_group, grad_a_group, grad_b_group
        ).arrange(DOWN, buff=0.3)

        # --- Bottom-left: Histogram (moderate size)
        res_axes = Axes(
            x_range=[-4, 4, 2],
            y_range=[0, hist_ymax, hist_step],
            tips=False,
            axis_config={"include_numbers": True, "font_size": 22},
            x_axis_config={"include_tip": False},
            y_axis_config={"include_tip": False},
        ).scale(1.0)  # Moderate size

        # Create axis labels with bigger font
        res_x_label = Tex(r"Residual (Error)", font_size=28)
        res_y_label = Tex(r"Frequency", font_size=28)
        res_x_label.next_to(res_axes.x_axis, DOWN, buff=0.1)
        res_y_label.next_to(res_axes.y_axis, LEFT, buff=0.1).rotate(PI / 2)
        res_axis_labels = VGroup(res_x_label, res_y_label)

        bl_cell = VGroup(res_axes, res_axis_labels)

        def residual_histogram_group(a, b, bins=10):
            counts, edges = hist_counts_edges(a, b, bins=bins)
            bars = VGroup()
            for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
                p00 = res_axes.c2p(e0, 0)
                p01 = res_axes.c2p(e0, c)
                p11 = res_axes.c2p(e1, c)
                p10 = res_axes.c2p(e1, 0)
                bars.add(
                    Polygon(p00, p01, p11, p10)
                    .set_fill(YELLOW_C, opacity=0.6)
                    .set_stroke(YELLOW_C, width=1)
                )
            return bars

        def zero_hist_group(bins=10):
            edges = np.linspace(-4, 4, bins + 1)
            bars = VGroup()
            for e0, e1 in zip(edges[:-1], edges[1:]):
                p00 = res_axes.c2p(e0, 0)
                p01 = res_axes.c2p(e0, 0)
                p11 = res_axes.c2p(e1, 0)
                p10 = res_axes.c2p(e1, 0)
                bars.add(
                    Polygon(p00, p01, p11, p10)
                    .set_fill(YELLOW_C, opacity=0.3)
                    .set_stroke(width=0)
                )
            return bars

        # --- Bottom-right: MSE plot (moderate size)
        steps = 5
        y_max_mse = max(10.0, mse0 * 1.4)
        y_step_mse = round(y_max_mse / 5)  # Round the step size
        mse_axes = Axes(
            x_range=[0, steps, 1],
            y_range=[0, round(y_max_mse), y_step_mse],  # Round the max value
            tips=False,
            axis_config={
                "include_numbers": True,
                "font_size": 20,
                "decimal_number_config": {"num_decimal_places": 0},
            },  # No decimals
        ).scale(1.0)  # Moderate size

        # Create axis labels with bigger font
        mse_x_label = Tex(r"Iteration Number", font_size=28)
        mse_y_label = Tex(r"Mean Squared Error", font_size=28)
        mse_x_label.next_to(mse_axes.x_axis, DOWN, buff=0.1)
        mse_y_label.next_to(mse_axes.y_axis, LEFT, buff=0.1).rotate(PI / 2)
        mse_axis_labels = VGroup(mse_x_label, mse_y_label)

        br_cell = VGroup(mse_axes, mse_axis_labels)
        mse_graph = VMobject(stroke_width=3)

        # --- Assemble grid to maximize screen usage
        tl_cell = VGroup(graph_cell)
        tr_cell = VGroup(formulas_col)

        # Create top row with center alignment for formulas
        top_row = VGroup(tl_cell, tr_cell).arrange(RIGHT, buff=0.2)

        bottom_row = VGroup(bl_cell, br_cell).arrange(RIGHT, buff=0.2)
        grid = VGroup(top_row, bottom_row).arrange(DOWN, buff=0.15, aligned_edge=LEFT)

        # Calculate maximum available space and scale aggressively to fill screen
        available_width = config.frame_width - 0.1  # More padding for visibility
        available_height = (
            config.frame_height
            - (title.height + spacer.height + status_spacer.height + 0.15)
            - 0.4  # More padding to ensure bottom axes are visible
        )

        # Scale to fill the maximum available space
        width_scale = available_width / grid.width
        height_scale = available_height / grid.height
        scale_factor = (
            min(width_scale, height_scale)
            * 0.85  # Use 85% of available space for better fit
        )

        grid.scale(scale_factor)

        # Center the grid in available space
        grid.next_to(status_spacer, DOWN, buff=0.03)
        grid.set_x(0)

        # Center formulas within the top-right cell after scaling
        # Use the tr_cell as reference and center the formulas within it
        formulas_col.move_to(tr_cell.get_center())

        # --- Eta tracker (needed for status display)
        eta_tracker = ValueTracker(0.35)  # bigger steps

        # --- Initial step + status text
        set_step_label(r"Initial Setup: Define model and starting parameters", BLUE_E)
        status.become(
            Tex(
                rf"Iteration 0 $\cdot$ $\eta$ = {eta_tracker.get_value():.3f} $\cdot$ a = {a_plot:.3f}, b = {b_plot:.3f} $\cdot$ MSE = {mse0:.3f}",
                font_size=26,
            ).move_to(status_spacer.get_center())
        )

        # --- Reveal scaffolding
        self.play(Create(axes), run_time=0.7)
        self.add(axes_labels)
        self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.02, run_time=1.0))
        self.play(
            LaggedStart(
                *[
                    Write(m)
                    for m in [yhat_group, cost_group, grad_a_group, grad_b_group]
                ],
                lag_ratio=0.10,
                run_time=1.0,
            )
        )
        self.play(Create(res_axes), run_time=0.6)
        self.play(Create(mse_axes), run_time=0.6)
        self.add(res_axis_labels, mse_axis_labels)
        self.add(mse_graph)
        hist = zero_hist_group()
        self.add(hist)
        self.wait(T_WAIT)

        # --- Focus helpers
        all_groups = [yhat_group, cost_group, grad_a_group, grad_b_group]
        BASE_COL = {
            yhat_group: BLUE_C,
            cost_group: GREEN_C,
            grad_a_group: YELLOW_C,
            grad_b_group: YELLOW_C,
        }
        FOCUS_COL = {
            yhat_group: BLUE_E,
            cost_group: GREEN_E,
            grad_a_group: YELLOW_E,
            grad_b_group: YELLOW_E,
        }

        def focus_on(target: VMobject, label_tex: str, label_color):
            set_step_label(label_tex, label_color)
            anims = []
            for grp in all_groups:
                if grp is target:
                    anims.append(grp.animate.set_opacity(1.0))
                else:
                    anims.append(grp.animate.set_opacity(0.25))
            self.play(*anims, run_time=T_FOCUS)

        def restore_all():
            self.play(
                *[grp.animate.set_opacity(1.0) for grp in all_groups],
                run_time=T_RESTORE,
            )

        # --- Readout (under formulas)
        def info_readout(iter_no, a, b):
            return Tex(
                "",  # Remove iteration display
                font_size=24,
            ).next_to(formulas_col, DOWN, buff=0.25, aligned_edge=LEFT)

        readout = info_readout(0, a_plot, b_plot)
        self.play(FadeIn(readout), run_time=0.45)
        self.wait(T_WAIT)

        # --- MSE incremental drawing
        mse_points = []

        def add_mse_point(i, a, b):
            y = min(mse(a, b), y_max_mse)
            mse_points.append(mse_axes.c2p(i, y))
            if len(mse_points) == 1:
                mse_graph.set_points_as_corners([mse_points[0], mse_points[0]])
            else:
                seg = Line(mse_points[-2], mse_points[-1], stroke_width=3)
                self.play(Create(seg), run_time=T_SEG)
                mse_graph.set_points_as_corners(mse_points)

        # Seed displays
        self.play(
            a_val.animate.set_value(a_plot),
            b_val.animate.set_value(b_plot),
            run_time=T_NUM,
        )
        self.wait(T_WAIT)

        # Draw initial model line and show initial metrics immediately after setting parameters
        initial_line = line_from_params(a_plot, b_plot, color=BLUE_C, width=6)
        initial_hist = residual_histogram_group(a_plot, b_plot)

        # Show initial line, histogram, and MSE simultaneously (no step text)
        self.play(
            Create(initial_line),
            Transform(hist, initial_hist),
            run_time=T_LINE,
        )
        add_mse_point(0, a_plot, b_plot)
        self.wait(T_WAIT)

        focus_on(cost_group, r"Step 2: Cost $J(a,b)$", GREEN_E)
        self.play(cost_val.animate.set_value(mse0), run_time=T_COST)
        restore_all()
        self.wait(T_WAIT)

        da0, db0 = grads_display(a_plot, b_plot)
        focus_on(grad_a_group, r"Step 3: Gradient $\partial J/\partial a$", YELLOW_E)
        self.play(grad_a_val.animate.set_value(da0), run_time=T_NUM)
        restore_all()
        self.wait(T_WAIT)

        focus_on(grad_b_group, r"Step 3: Gradient $\partial J/\partial b$", YELLOW_E)
        self.play(grad_b_val.animate.set_value(db0), run_time=T_NUM)
        restore_all()
        self.wait(T_WAIT)

        # Dynamic visuals storage
        cur_line = initial_line  # Start with the initial blue line
        errs = VGroup()
        trail = VGroup()
        MAX_TRAIL = 4
        TOL = 1e-3

        # ---- Optimization loop (3 steps)
        for i in range(1, 3 + 1):
            # Calculate gradients once per iteration
            da_s, db_s = grads_scaled(a_s, b_s)
            a_disp, b_disp = scaled_to_plot(a_s, b_s)
            da_disp, db_disp = grads_display(a_disp, b_disp)

            # Skip gradient display for first iteration (already shown in setup)
            if i > 1:
                focus_on(
                    grad_a_group, r"Recalculate gradients for new parameters", YELLOW_E
                )
                self.play(grad_a_val.animate.set_value(da_disp), run_time=T_NUM)
                restore_all()
                self.wait(T_WAIT)

                focus_on(
                    grad_b_group, r"Recalculate gradients for new parameters", YELLOW_E
                )
                self.play(grad_b_val.animate.set_value(db_disp), run_time=T_NUM)
                restore_all()
                self.wait(T_WAIT)

            if np.hypot(da_s, db_s) < TOL and i > 2:
                note = Tex("Converged (small gradient)", font_size=24, color=GREY_B)
                note.next_to(formulas_col, DOWN, buff=0.1, aligned_edge=LEFT)
                self.play(FadeIn(note), run_time=0.3)
                break

            # Update params (bigger step with higher eta)
            a_s = a_s - eta_tracker.get_value() * da_s
            b_s = b_s - eta_tracker.get_value() * db_s
            a_plot, b_plot = scaled_to_plot(a_s, b_s)

            focus_on(yhat_group, r"Update parameters using gradient descent", ORANGE)
            # Update both the formula values and the status bar simultaneously
            self.play(
                a_val.animate.set_value(a_plot),
                b_val.animate.set_value(b_plot),
                status.animate.become(
                    Tex(
                        rf"Iteration {i} $\cdot$ $\eta$ = {eta_tracker.get_value():.3f} $\cdot$ a = {a_plot:.3f}, b = {b_plot:.3f} $\cdot$ MSE = {mse(a_plot, b_plot):.3f}",
                        font_size=26,
                    ).move_to(status_spacer.get_center())
                ),
                run_time=T_NUM,
            )
            restore_all()
            self.wait(T_WAIT)

            # gentle decay once
            if i == 3:
                self.play(
                    eta_tracker.animate.set_value(eta_tracker.get_value() * 0.7),
                    run_time=0.35,
                )

            # Draw line & update residuals + cost
            new_line = line_from_params(a_plot, b_plot, color=ORANGE, width=6)
            if cur_line is not None:
                self.play(
                    cur_line.animate.set_color(GREY_B)
                    .set_opacity(0.6)
                    .set_stroke(width=3),
                    run_time=0.28,
                )
                trail.add(cur_line)
                if len(trail) > MAX_TRAIL:
                    self.play(FadeOut(trail[0], run_time=0.14))
                    trail.remove(trail[0])
            self.play(Create(new_line), run_time=T_LINE)
            self.wait(T_WAIT)

            focus_on(cost_group, r"Evaluate new cost and show residuals", GREEN_E)
            new_errs = residual_segments(a_plot, b_plot)
            if len(errs) == 0:
                self.play(Create(new_errs), run_time=T_RES)
                errs = new_errs
            else:
                self.play(Transform(errs, new_errs), run_time=T_RES)
            self.play(cost_val.animate.set_value(mse(a_plot, b_plot)), run_time=T_COST)
            restore_all()
            self.wait(T_WAIT)

            # Readout (lower area) & MSE point
            self.play(
                Transform(readout, info_readout(i, a_plot, b_plot)), run_time=0.35
            )
            add_mse_point(i, a_plot, b_plot)

            # Histogram update
            target_hist = residual_histogram_group(a_plot, b_plot)
            self.play(Transform(hist, target_hist), run_time=T_HIST)

            cur_line = new_line

        # ---------------- Polished Ending ----------------
        # Fade trail & residuals; highlight final line and final stats
        if len(trail) > 0:
            self.play(FadeOut(trail), run_time=0.3)
        if len(errs) > 0:
            self.play(FadeOut(errs), run_time=0.3)

        if cur_line is not None:
            final_line = cur_line.copy().set_color(BLUE_E).set_stroke(width=7)
            self.play(Transform(cur_line, final_line), run_time=0.45)

        final_text = readout.copy().set_color(GREEN_E)
        self.play(Transform(readout, final_text), run_time=0.35)

        set_step_label(r"Finished", GREEN_E)

        # Keep final state
        self.add(
            graph_cell,
            formulas_col,
            bl_cell,  # histogram + axis labels only
            br_cell,  # mse + axis labels only
            mse_graph,
            status,
            step_label,
            spacer,
            status_spacer,
            axes_labels,
        )
        self.wait(1.4)
