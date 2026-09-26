"""Emit the closed-loop joint-MAE chart for the website, in both layouts.

The numbers are Table 1 of ``crossbfm_ws_4.tex`` (mean +/- std over 3 seeds, every
policy at ep_frac = 1.0). Two arms differ only in conditioning, so the two
tracking bars for one robot sit side by side under a single robot label; goal
reaching has no reference-fed counterpart at all, so it lives past a dashed rule
rather than beside them. Which bar is which arm is carried by the legend in the
chart card, not by repeating the robot name under every bar.

Writes two ``<svg>`` blocks to stdout: a wide column layout and, for phones, the
same nine values turned on their side. Paste both into ``index.html`` inside the
chart card; exactly one is ever displayed.

  python3 tools/make_mae_chart.py > /tmp/mae.svg
"""

from __future__ import annotations

# robot -> (tracking latent, tracking joint, goal latent), each (mean, std) in rad
DATA = {
    "M3": ((0.2024, 0.0067), (0.1777, 0.0033), (0.2345, 0.0011)),
    "T1": ((0.1901, 0.0032), (0.1844, 0.0043), (0.1967, 0.0012)),
    "N1": ((0.1391, 0.0038), (0.1362, 0.0032), (0.1512, 0.0023)),
}
ORDER = ["M3", "T1", "N1"]
YMAX = 25.0            # MAE x 100
TICKS = [5, 10, 15, 20, 25]
R = 4                  # rounded data-end radius


def bars():
    """(robot, series, mean, std) in drawing order: tracking pairs, then goals."""
    out = []
    for r in ORDER:
        out.append((r, "latent", *DATA[r][0]))
        out.append((r, "joint", *DATA[r][1]))
    for r in ORDER:
        out.append((r, "goal", *DATA[r][2]))
    return [(r, s, m * 100, d * 100) for r, s, m, d in out]


def cls(series):
    return "mae-latent" if series in ("latent", "goal") else "mae-joint"


def desc():
    p = []
    for r in ORDER:
        (lm, _), (jm, _), (gm, _) = DATA[r]
        p.append(f"On {r}, tracking MAE is {lm:.4f} rad with the latent arm and "
                 f"{jm:.4f} with the reference-fed joint arm, a gap of {lm - jm:+.4f}; "
                 f"goal reaching, which the joint arm cannot do, is {gm:.4f}")
    return ". ".join(p) + "."


# ---------------------------------------------------------------- wide layout
def wide():
    W, H = 880, 400
    L, TOP, BASE = 92, 44, 344      # y axis x, plot top, baseline y
    BW, PAIR, GROUP, SPLIT, GOALGAP = 52, 8, 40, 74, 40
    y = lambda v: BASE - (v / YMAX) * (BASE - TOP)

    track_w = 3 * (2 * BW + PAIR) + 2 * GROUP
    goal_w = 3 * BW + 2 * GOALGAP
    x0 = L + ((862 - L) - (track_w + SPLIT + goal_w)) / 2

    xs, x = [], x0
    for i in range(3):                       # tracking pairs
        xs += [x, x + BW + PAIR]
        x += 2 * BW + PAIR + (GROUP if i < 2 else 0)
    split_x = x + SPLIT / 2
    x += SPLIT
    for i in range(3):                       # goal singles
        xs.append(x)
        x += BW + (GOALGAP if i < 2 else 0)

    s = [f'<svg class="barchart chart-wide" viewBox="0 0 {W} {H}" role="img" '
         f'aria-labelledby="mae-t mae-d" preserveAspectRatio="xMidYMid meet">',
         '  <title id="mae-t">Closed-loop joint MAE for the latent and reference-fed arms, '
         'and for goal reaching</title>',
         f'  <desc id="mae-d">{desc()}</desc>',
         '  <g class="grid">']
    for t in TICKS:
        s.append(f'    <line x1="{L}" y1="{y(t):.1f}" x2="862" y2="{y(t):.1f}"/>')
    s.append(f'    <line class="axis" x1="{L}" y1="{BASE}" x2="862" y2="{BASE}"/>')
    s.append('  </g>')
    s.append('  <g class="ytick">')
    for t in TICKS:
        s.append(f'    <text x="78" y="{y(t) + 4:.1f}" text-anchor="end">{t}</text>')
    s.append('  </g>')
    s.append(f'  <text class="axis-title" transform="translate(30 {(TOP + BASE) / 2:.0f}) '
             'rotate(-90)" text-anchor="middle">Joint MAE (&#215;100)</text>')

    # The dashed rule: goal reaching is a different task, not a third arm.
    s.append(f'  <line class="split-rule" x1="{split_x:.1f}" y1="12" '
             f'x2="{split_x:.1f}" y2="{BASE}"/>')
    s.append(f'  <text class="band" x="{(xs[0] + xs[5] + BW) / 2:.1f}" y="26" '
             'text-anchor="middle">Tracking</text>')
    s.append(f'  <text class="band" x="{(xs[6] + xs[8] + BW) / 2:.1f}" y="26" '
             'text-anchor="middle">Goal reaching</text>')

    s.append('  <g class="bars">')
    for (robot, series, m, sd), bx in zip(bars(), xs):
        top, cx = y(m), bx + BW / 2
        s.append(f'    <g class="bar {cls(series)}">')
        s.append(f'      <title>{robot} {series}: {m / 100:.4f} &#177; {sd / 100:.4f} rad</title>')
        s.append(f'      <path d="M{bx:.1f} {BASE} V{top + R:.1f} A{R} {R} 0 0 1 {bx + R:.1f} {top:.1f} '
                 f'H{bx + BW - R:.1f} A{R} {R} 0 0 1 {bx + BW:.1f} {top + R:.1f} V{BASE} Z"/>')
        s.append(f'      <path class="err" d="M{cx - 7:.1f} {y(m - sd):.1f} H{cx + 7:.1f} '
                 f'M{cx:.1f} {y(m - sd):.1f} V{y(m + sd):.1f} '
                 f'M{cx - 7:.1f} {y(m + sd):.1f} H{cx + 7:.1f}"/>')
        s.append(f'      <text class="val" x="{cx:.1f}" y="{y(m + sd) - 11:.1f}" '
                 f'text-anchor="middle">{m:.2f}</text>')
        s.append('    </g>')
    s.append('  </g>')

    # One robot label per tracking pair, one per goal bar: the legend, not a
    # repeated name, says which of the two bars is which arm.
    s.append('  <g class="cats">')
    for i, r in enumerate(ORDER):
        cx = xs[2 * i] + (2 * BW + PAIR) / 2
        s.append(f'    <text class="cat" x="{cx:.1f}" y="{BASE + 24}" text-anchor="middle">{r}</text>')
    for i, r in enumerate(ORDER):
        s.append(f'    <text class="cat" x="{xs[6 + i] + BW / 2:.1f}" y="{BASE + 24}" '
                 f'text-anchor="middle">{r}</text>')
    s.append('  </g>')
    s.append('</svg>')
    return "\n".join(s)


# -------------------------------------------------------------- narrow layout
def narrow():
    W, RIGHT = 420, 380
    L = 74
    BH, PAIRGAP, ROBOTGAP, GOALGAP, SPLIT = 22, 6, 24, 16, 52
    x = lambda v: L + (v / YMAX) * (RIGHT - L)

    top = 34                                  # room for the "Tracking" band label
    rows = []
    for i, r in enumerate(ORDER):             # tracking pairs
        yy = top + i * (2 * BH + PAIRGAP + ROBOTGAP)
        rows.append(((r, "latent", *DATA[r][0]), yy))
        rows.append(((r, "joint", *DATA[r][1]), yy + BH + PAIRGAP))
    track_end = rows[-1][1] + BH
    split_y = track_end + SPLIT / 2 - 10
    goal_top = track_end + SPLIT
    for i, r in enumerate(ORDER):             # goal singles
        rows.append(((r, "goal", *DATA[r][2]), goal_top + i * (BH + GOALGAP)))
    plot_end = rows[-1][1] + BH
    H = plot_end + 44

    rows = [((r, s, m * 100, d * 100), yy) for (r, s, m, d), yy in rows]

    s = [f'<svg class="barchart chart-narrow" viewBox="0 0 {W} {H:.0f}" role="img" '
         f'aria-labelledby="mae-t2 mae-d2" preserveAspectRatio="xMidYMid meet">',
         '  <title id="mae-t2">Closed-loop joint MAE for the latent and reference-fed arms, '
         'and for goal reaching</title>',
         f'  <desc id="mae-d2">{desc()}</desc>',
         '  <g class="grid">']
    for t in TICKS:
        s.append(f'    <line x1="{x(t):.1f}" y1="{top - 10}" x2="{x(t):.1f}" y2="{plot_end + 4:.0f}"/>')
    s.append(f'    <line class="axis" x1="{L}" y1="{top - 10}" x2="{L}" y2="{plot_end + 4:.0f}"/>')
    s.append('  </g>')
    s.append('  <g class="ytick">')
    for t in TICKS:
        s.append(f'    <text x="{x(t):.1f}" y="{plot_end + 22:.0f}" text-anchor="middle">{t}</text>')
    s.append('  </g>')
    s.append(f'  <text class="axis-title" x="{(L + RIGHT) / 2:.0f}" y="{H - 6:.0f}" '
             'text-anchor="middle">Joint MAE (&#215;100)</text>')
    s.append(f'  <line class="split-rule" x1="8" y1="{split_y:.0f}" x2="{RIGHT}" y2="{split_y:.0f}"/>')
    s.append(f'  <text class="band" x="8" y="{top - 16}">Tracking</text>')
    s.append(f'  <text class="band" x="8" y="{goal_top - 10:.0f}">Goal reaching</text>')

    s.append('  <g class="bars">')
    for (robot, series, m, sd), by in rows:
        w, cy = x(m), by + BH / 2
        s.append(f'    <g class="bar {cls(series)}">')
        s.append(f'      <title>{robot} {series}: {m / 100:.4f} &#177; {sd / 100:.4f} rad</title>')
        s.append(f'      <path d="M{L} {by:.0f} H{w - R:.1f} A{R} {R} 0 0 1 {w:.1f} {by + R:.0f} '
                 f'V{by + BH - R:.0f} A{R} {R} 0 0 1 {w - R:.1f} {by + BH:.0f} H{L} Z"/>')
        s.append(f'      <path class="err" d="M{x(m - sd):.1f} {cy - 5:.1f} V{cy + 5:.1f} '
                 f'M{x(m - sd):.1f} {cy:.1f} H{x(m + sd):.1f} '
                 f'M{x(m + sd):.1f} {cy - 5:.1f} V{cy + 5:.1f}"/>')
        s.append(f'      <text class="val" x="{x(m + sd) + 9:.1f}" y="{cy + 5:.1f}">{m:.2f}</text>')
        s.append('    </g>')
    s.append('  </g>')

    # Robot names sit once per tracking pair, centred across its two rows.
    s.append('  <g class="cats">')
    for i in range(3):
        cy = rows[2 * i][1] + (2 * BH + PAIRGAP) / 2 + 5
        s.append(f'    <text class="cat" x="{L - 8}" y="{cy:.0f}" text-anchor="end">{ORDER[i]}</text>')
    for i in range(3):
        cy = rows[6 + i][1] + BH / 2 + 5
        s.append(f'    <text class="cat" x="{L - 8}" y="{cy:.0f}" text-anchor="end">{ORDER[i]}</text>')
    s.append('  </g>')
    s.append('</svg>')
    return "\n".join(s)


if __name__ == "__main__":
    print(wide())
    print()
    print(narrow())
