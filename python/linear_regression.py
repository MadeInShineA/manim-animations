from manim import *
import numpy as np
import random
import re


class LinearRegression(Scene):
    def construct(self):
        # ---------------- Title ----------------
        title = Tex("Linear Regression (Gradient Descent)", font_size=48)
        self.play(FadeIn(title))
        self.wait(0.6)
        self.play(title.animate.to_edge(UP))

        # ---------------- Math-aware text wrap (doesn't split $...$) ----------------
        def wrap_text_tex(text: str, font_size: int, max_width: float) -> list[str]:
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
            current = ""
            for tok in tokens:
                if tok == "":
                    continue
                cand = (current + " " + tok).strip() if current else tok
                if Tex(cand, font_size=font_size).width > max_width and current:
                    lines.append(current)
                    current = tok
                else:
                    current = cand
            if current:
                lines.append(current)
            return lines

        # ===================== Screen 1: Description =====================
        desc_raw = (
            "We observe data points $(x_i,y_i)$ and fit a straight line to predict $y$ from $x$. "
            "Linear regression chooses slope $a$ and intercept $b$ to minimize the mean squared error (MSE)."
        )
        desc_lines = wrap_text_tex(
            desc_raw, font_size=36, max_width=config.frame_width - 1
        )
        description_group = (
            VGroup(*[Tex(ln, font_size=36) for ln in desc_lines])
            .arrange(DOWN, aligned_edge=LEFT)
            .next_to(title, DOWN, buff=0.6)
            .to_edge(LEFT, buff=0.6)
        )
        self.play(Write(description_group))
        self.wait(3)
        self.play(FadeOut(description_group))
        self.wait(0.3)

        # ===================== Screen 2: Use cases =====================
        usecases_heading = (
            Tex("Use Cases:", font_size=36)
            .next_to(title, DOWN, buff=0.6)
            .to_edge(LEFT, buff=0.6)
        )
        self.play(Write(usecases_heading))
        self.wait(0.2)
        usecases = [
            "Trend prediction (e.g., sales vs. time)",
            "Calibration (sensor reading → true value)",
            "Simple forecasting (temperature vs. day)",
            "Econometrics (consumption vs. income)",
            "ML baseline for regression tasks",
        ]
        bullets = VGroup()
        prev = usecases_heading
        for u in usecases:
            b = Tex("• " + u, font_size=32)
            b.next_to(prev, DOWN, aligned_edge=LEFT)
            self.play(Write(b))
            self.wait(0.15)
            bullets.add(b)
            prev = b
        self.wait(1.0)
        self.play(FadeOut(usecases_heading), *[FadeOut(b) for b in bullets])
        self.wait(0.3)

        # ===================== Screen 3: Steps (colors match formulas) =====================
        steps_title = (
            Tex("Steps of the algorithm:", font_size=36)
            .next_to(title, DOWN, buff=0.6)
            .to_edge(LEFT, buff=0.6)
        )
        self.play(Write(steps_title))
        self.wait(0.2)
        steps = [
            r"1. Choose the model $\hat{y} = a x + b$.",  # BLUE
            r"2. Define the cost: $J(a,b)=\frac{1}{n}\sum (y_i-\hat{y}_i)^2$.",  # GREEN
            r"3. Compute gradients $\frac{\partial J}{\partial a},\,\frac{\partial J}{\partial b}$.",  # YELLOW
            r"4. Update with learning rate $\eta$: $a \leftarrow a-\eta\,\partial J/\partial a,\; b \leftarrow b-\eta\,\partial J/\partial b$.",  # ORANGE
            r"5. Repeat until $J$ stabilizes.",  # GREY
        ]
        STEP_COLORS = [BLUE, GREEN, YELLOW, ORANGE, GREY_B]
        prev = steps_title
        step_group = VGroup()
        for text, col in zip(steps, STEP_COLORS):
            line = Tex(text, font_size=32)
            num_token = text.split()[0]  # "1.", "2.", ...
            line.set_color_by_tex(num_token, col)
            if col is BLUE:
                line.set_color_by_tex(r"\hat{y}", BLUE)
            elif col is GREEN:
                line.set_color_by_tex("J", GREEN)
            elif col is YELLOW:
                line.set_color_by_tex(r"\frac{\partial J}{\partial a}", YELLOW)
                line.set_color_by_tex(r"\frac{\partial J}{\partial b}", YELLOW)
            elif col is ORANGE:
                line.set_color_by_tex(r"\eta", ORANGE)
                line.set_color_by_tex(r"\leftarrow", ORANGE)
            line.next_to(prev, DOWN, aligned_edge=LEFT)
            self.play(Write(line))
            self.wait(0.2)
            step_group.add(line)
            prev = line
        self.wait(1.0)
        self.play(FadeOut(steps_title), *[FadeOut(s) for s in step_group])
        self.wait(0.2)

        # ===================== Screen 4: Graph + 2x2 formulas + Animation =====================
        axes = (
            Axes(
                x_range=[0, 10, 1],
                y_range=[0, 10, 1],
                tips=False,
                axis_config={"include_numbers": True},
            )
            .add_coordinates()
            .scale(0.78)
        )

        # Data
        n = 50
        random.seed(666)
        np.random.seed(666)
        true_m, true_b = 0.7, 2.0
        xs = np.random.uniform(0, 10, size=n)
        noise = np.random.uniform(-1.0, 1.0, size=n)
        ys = true_m * xs + true_b + noise

        # Dots inside axes
        dots = VGroup(*[Dot(axes.c2p(x, y), radius=0.055) for x, y in zip(xs, ys)])
        axes.add(*dots)

        # -------- Static left parts + live numeric parts (keeps layout fixed) --------
        # yhat = a x + b   (BLUE)
        yhat_left = MathTex(r"\hat{y} = ").set_color(BLUE)
        yhat_ax = MathTex("x + ").set_color(BLUE)
        a_val = DecimalNumber(-0.20, num_decimal_places=2, color=BLUE, font_size=36)
        b_val = DecimalNumber(0.80, num_decimal_places=2, color=BLUE, font_size=36)
        yhat_group = VGroup(yhat_left, a_val, yhat_ax, b_val).arrange(RIGHT, buff=0.05)

        # Cost J(a,b) = ... = value (GREEN)
        cost_left = MathTex(
            r"J(a,b) = \frac{1}{n}\sum_{i=1}^{n}\big(y_i - (a x_i + b)\big)^2",
            font_size=32,
        ).set_color(GREEN)
        cost_eq = MathTex(r"=", font_size=32).set_color(GREEN)
        cost_val = DecimalNumber(0.0, num_decimal_places=3, color=GREEN, font_size=32)
        cost_group = VGroup(cost_left, cost_eq, cost_val).arrange(RIGHT, buff=0.12)

        # Grad a (YELLOW)
        grad_a_left = MathTex(
            r"\frac{\partial J}{\partial a} = -\frac{2}{n}\sum x_i\big(y_i - (a x_i + b)\big)",
            font_size=30,
        ).set_color(YELLOW)
        grad_a_eq = MathTex(r"=", font_size=30).set_color(YELLOW)
        grad_a_val = DecimalNumber(
            0.0, num_decimal_places=3, color=YELLOW, font_size=30
        )
        grad_a_group = VGroup(grad_a_left, grad_a_eq, grad_a_val).arrange(
            RIGHT, buff=0.12
        )

        # Grad b (YELLOW)
        grad_b_left = MathTex(
            r"\frac{\partial J}{\partial b} = -\frac{2}{n}\sum \big(y_i - (a x_i + b)\big)",
            font_size=30,
        ).set_color(YELLOW)
        grad_b_eq = MathTex(r"=", font_size=30).set_color(YELLOW)
        grad_b_val = DecimalNumber(
            0.0, num_decimal_places=3, color=YELLOW, font_size=30
        )
        grad_b_group = VGroup(grad_b_left, grad_b_eq, grad_b_val).arrange(
            RIGHT, buff=0.12
        )

        # Arrange into 2x2 grid (fixed positions)
        formula_grid = VGroup(
            yhat_group, cost_group, grad_a_group, grad_b_group
        ).arrange_in_grid(
            rows=2,
            cols=2,
            buff=(0.28, 0.28),
            row_alignments=["c", "c"],
            col_alignments=["c", "c"],
        )

        # Center graph + formulas together
        graph_block = VGroup(axes, formula_grid).arrange(DOWN, buff=0.45)
        graph_block.next_to(title, DOWN, buff=0.45)
        graph_block.move_to(ORIGIN)

        if formula_grid.width > (config.frame_width - 0.8):
            formula_grid.scale((config.frame_width - 0.8) / formula_grid.width)

        self.play(Create(axes))
        self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.03, run_time=0.9))
        self.play(
            LaggedStart(
                *[
                    Write(m)
                    for m in [yhat_group, cost_group, grad_a_group, grad_b_group]
                ],
                lag_ratio=0.12,
            )
        )
        self.wait(0.2)

        # -------- Helpers --------
        def line_from_params(m, b, color=ORANGE, width=6):
            x0, x1 = axes.x_range[0], axes.x_range[1]
            p0 = axes.c2p(x0, m * x0 + b)
            p1 = axes.c2p(x1, m * x1 + b)
            return Line(p0, p1, color=color, stroke_width=width)

        def error_group(m, b):
            g = VGroup()
            for x, y in zip(xs, ys):
                yhat = m * x + b
                a = axes.c2p(x, min(y, yhat))
                bb = axes.c2p(x, max(y, yhat))
                g.add(Line(a, bb, color=YELLOW, stroke_width=1.1))
            return g

        def grads_display(a, b):
            yhat = a * xs + b
            err = ys - yhat
            n_ = len(xs)
            da = -(2.0 / n_) * np.sum(xs * err)
            db = -(2.0 / n_) * np.sum(err)
            return da, db

        def mse(a, b):
            return np.mean((ys - (a * xs + b)) ** 2)

        # Optimizer uses scaled x for stability
        x_mu, x_sigma = xs.mean(), xs.std()
        x_scaled = (xs - x_mu) / x_sigma

        def grads_scaled(a_s, b_s):
            yhat = a_s * x_scaled + b_s
            err = ys - yhat
            n_ = len(xs)
            da = -(2.0 / n_) * np.sum(x_scaled * err)
            db = -(2.0 / n_) * np.sum(err)
            return da, db

        def scaled_to_plot(a_s, b_s):
            a = a_s / x_sigma
            b = b_s - (a_s * x_mu / x_sigma)
            return a, b

        def info_readout(iter_no, a, b):
            t = Tex(
                f"Iteration {iter_no}   a={a:.3f}, b={b:.3f}   MSE={mse(a, b):.3f}",
                font_size=26,
            )
            return t.next_to(formula_grid, DOWN, buff=0.25)

        # -------- Animation: Gradient Descent (with η) --------
        a_s, b_s = -0.2, 0.8
        eta = 0.12
        steps = 8

        a_plot, b_plot = scaled_to_plot(a_s, b_s)
        cur_line = line_from_params(a_plot, b_plot, color=ORANGE, width=6)
        errs = error_group(a_plot, b_plot)

        # Set visible numbers (yhat, grads, cost)
        self.play(a_val.animate.set_value(a_plot), b_val.animate.set_value(b_plot))
        da_disp, db_disp = grads_display(a_plot, b_plot)
        self.play(
            grad_a_val.animate.set_value(da_disp),
            grad_b_val.animate.set_value(db_disp),
            cost_val.animate.set_value(mse(a_plot, b_plot)),
        )

        readout = info_readout(0, a_plot, b_plot)
        self.play(Create(cur_line), Create(errs), FadeIn(readout))

        trail = VGroup()
        for i in range(1, steps + 1):
            da_s, db_s = grads_scaled(a_s, b_s)
            a_s = a_s - eta * da_s
            b_s = b_s - eta * db_s
            a_plot, b_plot = scaled_to_plot(a_s, b_s)

            new_line = line_from_params(a_plot, b_plot, color=ORANGE, width=6)
            new_errs = error_group(a_plot, b_plot)

            da_disp, db_disp = grads_display(a_plot, b_plot)
            new_readout = info_readout(i, a_plot, b_plot)

            self.play(
                cur_line.animate.set_color(GREY_B).set_opacity(0.6).set_stroke(width=3),
                run_time=0.2,
            )
            trail.add(cur_line)
            self.play(Create(new_line), run_time=0.4)

            self.play(
                Transform(errs, new_errs),
                a_val.animate.set_value(a_plot),
                b_val.animate.set_value(b_plot),
                grad_a_val.animate.set_value(da_disp),
                grad_b_val.animate.set_value(db_disp),
                cost_val.animate.set_value(mse(a_plot, b_plot)),
                Transform(readout, new_readout),
                run_time=0.45,
            )

            cur_line = new_line
            self.wait(0.05)

        # -------- End Visuals --------
        # 1) Remove ALL orange lines & residuals
        self.play(FadeOut(trail), FadeOut(cur_line), FadeOut(errs))

        # 2) Keep the true GREEN line
        true_line = Line(
            axes.c2p(0, true_m * 0 + true_b),
            axes.c2p(10, true_m * 10 + true_b),
            color=GREEN,
            stroke_width=5,
        )
        self.play(Create(true_line))

        # 3) Final y-hat shows the last numbers (BLUE, layout fixed)
        self.play(
            a_val.animate.set_value(true_m),
            b_val.animate.set_value(true_b),
        )

        # 4) OPTIONAL: trail of the data-generating function y = true_m x + true_b (GREEN)
        #    A sequence of small green dots fades in from left to right along the true line.
        trail_dots = VGroup()
        samples = 90
        for k in range(samples):
            x = 0 + (10 - 0) * (k / (samples - 1))
            y = true_m * x + true_b
            d = Dot(axes.c2p(x, y), radius=0.03, color=GREEN)
            d.set_opacity(0.85)
            trail_dots.add(d)
        # ensure left-to-right animation order
        trail_dots.submobjects.sort(key=lambda d: d.get_center()[0])

        self.play(
            LaggedStart(*[FadeIn(d) for d in trail_dots], lag_ratio=0.02, run_time=1.4)
        )

        # Keep a clean final frame: points, axes, green line, green trail, formulas + readout
        self.add(axes, true_line, trail_dots, formula_grid, readout)
        self.wait(1.4)
