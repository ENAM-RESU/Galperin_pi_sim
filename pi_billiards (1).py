"""
Galperin's billiard-ball method of computing pi -- with non-ideal physics.

Setup (one dimension, wall on the left):

    |wall|  (small ball) ........ [ BIG BLOCK ] <-- moving toward the wall

The block is 100**n times heavier than the ball. Count every collision
(ball-block and ball-wall). In the ideal case the count is the first n+1
digits of pi: 31, 314, 3141, ...

This file has two simulators:

  1. simulate()         EVENT-DRIVEN. Objects are points, collisions are
                        instant, and we compute exactly WHEN each collision
                        happens. Supports restitution (e) and friction.

  2. simulate_spring()  CONTACT MODEL. Objects have size, and a collision is
                        a spring + damper squeezing for a short time. Uses
                        small time steps. Supports e (via damping) and friction.

Plus animate() for a visual and sweep_restitution() for a graph.

Units: the block starts at distance 2 from the wall moving at speed 1, and
the ball starts at distance 1 from the wall, at rest. The ball's mass is 1.
Positive velocity = moving away from the wall.
"""

import math

# ---------------------------------------------------------------------------
# 1. EVENT-DRIVEN SIMULATION
# ---------------------------------------------------------------------------


def _first_root(A, B, C):
    """Smallest t > 0 solving A*t^2 + B*t + C = 0, or None if there isn't one."""
    if A == 0:
        if B == 0:
            return None
        t = -C / B
        return t if t > 0 else None
    disc = B * B - 4 * A * C
    if disc < 0:
        return None
    s = math.sqrt(disc)
    q = -0.5 * (B + math.copysign(s, B))   # numerically stable quadratic formula
    roots = [q / A]
    if q != 0:
        roots.append(C / q)
    positive = [r for r in roots if r > 0]
    return min(positive) if positive else None


def _sign(v):
    return (v > 0) - (v < 0)


def simulate(n, e=1.0, friction=0.0, max_collisions=1_000_000, record=False):
    """
    Event-driven simulation with mass ratio 100**n.

    e              coefficient of restitution (1.0 = perfectly elastic)
    friction       constant deceleration of both objects, in the units above.
                   (If you think in terms of a friction coefficient mu, this is
                   mu * g. It is the same for both objects, because heavier
                   objects have proportionally more friction force.)
    max_collisions safety cap so the program can't run forever
    record         if True, keep snapshots for animate()

    Returns a dict with:
      "count"     number of collisions (ball-block + ball-wall)
      "status"    "done"            -> everything finished normally
                  "collapsed"       -> infinite collisions in finite time
                                       (inelastic collapse); count is
                                       where we gave up
                  "max_collisions"  -> hit the safety cap
      "snapshots" list of (x1, x2, count) if record=True, else None
    """
    m1 = 1.0
    m2 = 100.0 ** n
    x1, x2 = 1.0, 2.0          # ball, block
    v1, v2 = 0.0, -1.0
    count = 0
    status = "done"
    snaps = [(x1, x2, 0)] if record else None

    while True:
        if count >= max_collisions:
            status = "max_collisions"
            break
        if x2 < 1e-12:   # block squashed against the wall: collision pile-up
            status = "collapsed"
            break

        d1 = friction * _sign(v1)    # deceleration of each object right now
        d2 = friction * _sign(v2)

        # Candidate next events: (time until it happens, what it is)
        events = []
        if friction > 0:
            if v1 != 0:
                events.append((abs(v1) / friction, "stop1"))
            if v2 != 0:
                events.append((abs(v2) / friction, "stop2"))
        t_wall = _first_root(-0.5 * d1, v1, x1)
        if t_wall is not None:
            events.append((t_wall, "wall"))
        t_block = _first_root(-0.5 * (d2 - d1), v2 - v1, x2 - x1)
        if t_block is not None:
            events.append((t_block, "block"))

        if not events:
            break                      # nothing can happen anymore: done

        dt, kind = min(events)

        # Jump forward to that moment
        x1 += v1 * dt - 0.5 * d1 * dt * dt
        x2 += v2 * dt - 0.5 * d2 * dt * dt
        v1 -= d1 * dt
        v2 -= d2 * dt

        if kind == "stop1":
            v1 = 0.0
        elif kind == "stop2":
            v2 = 0.0
        elif kind == "wall":
            x1 = 0.0
            v1 = -e * v1
            count += 1
        else:  # "block"
            x1 = x2                    # remove tiny rounding gap
            total = m1 + m2
            new_v1 = ((m1 - e * m2) * v1 + (1 + e) * m2 * v2) / total
            new_v2 = ((m2 - e * m1) * v2 + (1 + e) * m1 * v1) / total
            v1, v2 = new_v1, new_v2
            count += 1

        if record:
            snaps.append((x1, x2, count))

    return {"count": count, "status": status, "snapshots": snaps}


def count_collisions(n, e=1.0, friction=0.0, max_collisions=1_000_000):
    """Convenience wrapper: just the collision count."""
    return simulate(n, e, friction, max_collisions)["count"]


# ---------------------------------------------------------------------------
# 2. SPRING-AND-DAMPER CONTACT SIMULATION
# ---------------------------------------------------------------------------


def simulate_spring(n=1, e=1.0, stiffness=1e8, friction=0.0,
                    ball_radius=0.01, block_half=0.02,
                    steps_per_contact=40, max_steps=20_000_000, record=False):
    """
    Same experiment, but collisions take time.

    The ball has radius `ball_radius` and the block has half-width
    `block_half`. When two things overlap, a spring pushes them apart with
    force  k * overlap + c * (overlap speed)  (the second term is the damper,
    which removes energy).

    e            roughly the restitution you want. We pick the damper strength
                 c for each contact so that an isolated collision would give
                 about this e. (It is approximate; check with a test.)
    stiffness    spring constant k. Bigger k = harder objects = shorter
                 collisions = closer to the instant-collision model. The
                 time step shrinks as k grows, so the run gets slower.
    friction     same meaning as in simulate().

    Keep the sizes small compared to the distances (1 and 2), or the finite
    size itself will change the count. Also keep the squeeze depth small
    compared to the ball radius: depth is about  speed / sqrt(k).
    The ball can reach speeds near sqrt(100**n), so for bigger n you need
    a bigger k (and a longer wait).

    Returns a dict with:
      "count"         ball-wall + ball-block collisions (comparable to simulate())
      "ball_wall", "ball_block", "block_wall"   counts of each kind
      "status"        "done" or "max_steps"
      "snapshots"     list of (x1, x2, count) in the same 'point' coordinates
                      that animate() expects, if record=True
    A collision is counted each time two things go from not touching to touching.
    """
    m1 = 1.0
    m2 = 100.0 ** n
    k = stiffness
    r, h = ball_radius, block_half

    # Damping for each contact, chosen to give roughly the requested e
    if e >= 1:
        zeta = 0.0
    elif e <= 0:
        zeta = 1.0
    else:
        L = -math.log(e)
        zeta = L / math.sqrt(math.pi ** 2 + L ** 2)
    m_bb = m1 * m2 / (m1 + m2)                    # effective mass, ball-block
    c_bw = 2 * zeta * math.sqrt(k * m1)           # ball-wall
    c_bb = 2 * zeta * math.sqrt(k * m_bb)         # ball-block
    c_kw = 2 * zeta * math.sqrt(k * m2)           # block-wall

    dt_min = math.pi * math.sqrt(min(m1, m_bb) / k) / steps_per_contact
    dt_coarse = 1e-2

    x1, x2 = 1.0, 2.0
    v1, v2 = 0.0, -1.0
    in_bw = in_bb = in_kw = False
    n_bw = n_bb = n_kw = 0
    status = "done"

    def to_point(a, b):    # convert to the point-model coordinates for animate()
        return (a - r, b - h - 2 * r)

    snaps = [(*to_point(x1, x2), 0)] if record else None

    for _ in range(max_steps):
        # Overlaps (positive = squeezed)
        o_bw = r - x1
        o_bb = (x1 + r) - (x2 - h)
        o_kw = h - x2
        touching = o_bw > 0 or o_bb > 0 or o_kw > 0

        if not touching:
            # Free flight: if nothing is approaching anything, we're finished
            times = []
            if v1 < 0:
                times.append((x1 - r) / -v1)
            if v1 > v2:
                times.append(((x2 - h) - (x1 + r)) / (v1 - v2))
            if v2 < 0:
                times.append((x2 - h) / -v2)
            # (if the next approach is more than 100 time units away -- the
            # whole experiment takes about 2 -- it is just numerical dust)
            if not times or min(times) > 100:
                break
            dt = min(dt_coarse, max(dt_min, 0.5 * min(times)))
        else:
            dt = dt_min

        # Forces (springs can only push, never pull)
        f1 = f2 = 0.0
        if o_bw > 0:
            f1 += max(0.0, k * o_bw + c_bw * (-v1))
        if o_bb > 0:
            F = max(0.0, k * o_bb + c_bb * (v1 - v2))
            f1 -= F
            f2 += F
        if o_kw > 0:
            f2 += max(0.0, k * o_kw + c_kw * (-v2))

        v1 += f1 / m1 * dt
        v2 += f2 / m2 * dt

        # Friction: slow each object toward zero, but never reverse it
        if friction > 0:
            dv = friction * dt
            v1 = 0.0 if abs(v1) <= dv else v1 - dv * _sign(v1)
            v2 = 0.0 if abs(v2) <= dv else v2 - dv * _sign(v2)

        x1 += v1 * dt
        x2 += v2 * dt

        # Count new collisions (not-touching -> touching)
        new_bw, new_bb, new_kw = (r - x1) > 0, ((x1 + r) - (x2 - h)) > 0, (h - x2) > 0
        hit = False
        if new_bw and not in_bw:
            n_bw += 1
            hit = True
        if new_bb and not in_bb:
            n_bb += 1
            hit = True
        if new_kw and not in_kw:
            n_kw += 1
        in_bw, in_bb, in_kw = new_bw, new_bb, new_kw
        if record and hit:
            snaps.append((*to_point(x1, x2), n_bw + n_bb))
    else:
        status = "max_steps"

    return {"count": n_bw + n_bb, "ball_wall": n_bw, "ball_block": n_bb,
            "block_wall": n_kw, "status": status, "snapshots": snaps}


# ---------------------------------------------------------------------------
# 3. VISUALS
# ---------------------------------------------------------------------------


def animate(snapshots, title="", frames_per_collision=None, save=None):
    """
    Animate a run (pass the "snapshots" from simulate() or simulate_spring(),
    which need record=True).

    The animation is schematic: every collision gets the same amount of screen
    time, so you can follow all of them (in real time, most collisions would
    be over in a blink). The counter shows the collision count.
    The sizes of the ball and block are only for drawing.

    save="run.gif" writes a GIF instead of opening a window.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle, Circle

    # Thin out very long runs so the animation stays a reasonable length
    stride = max(1, math.ceil(len(snapshots) / 1500))
    snaps = snapshots[::stride]
    if snaps[-1] != snapshots[-1]:
        snaps.append(snapshots[-1])
    if frames_per_collision is None:
        frames_per_collision = max(1, min(8, 600 // max(1, len(snaps))))

    frames = []
    for a, b in zip(snaps, snaps[1:]):
        for i in range(frames_per_collision):
            f = i / frames_per_collision
            frames.append((a[0] + f * (b[0] - a[0]),
                           a[1] + f * (b[1] - a[1]), a[2]))
    frames.append(snaps[-1])

    rd, bw = 0.04, 0.25       # drawn ball radius and block width (cosmetic)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.set_xlim(-0.2, 2.7)
    ax.set_ylim(-0.1, 0.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)
    ax.add_patch(Rectangle((-0.12, 0), 0.12, 0.4, color="0.4"))        # wall
    ax.plot([-0.12, 2.7], [0, 0], color="0.6", lw=1)                   # floor
    ball = Circle((0, rd), rd, color="tab:red")
    block = Rectangle((0, 0), bw, bw, color="tab:blue")
    ax.add_patch(ball)
    ax.add_patch(block)
    text = ax.text(1.3, 0.38, "", ha="center", fontsize=14)

    def draw(i):
        x1, x2, c = frames[i]
        ball.center = (x1 + rd, rd)
        block.set_x(x2 + 2 * rd)
        text.set_text(f"collisions: {c}")
        return ball, block, text

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=40, blit=False)
    if save:
        anim.save(save, writer="pillow", fps=25)
        plt.close(fig)
    else:
        plt.show()
    return anim


def sweep_restitution(n, e_values, friction=0.0, show=True):
    """Run the event-driven simulation for several e values; print and plot."""
    import matplotlib.pyplot as plt

    counts = []
    print(f"mass ratio 100^{n}, friction {friction}")
    for e in e_values:
        r = simulate(n, e=e, friction=friction)
        counts.append(r["count"])
        print(f"  e = {e:<7} -> {r['count']:>8} collisions  ({r['status']})")
    plt.figure()
    plt.plot(e_values, counts, "o-")
    plt.axhline(count_collisions(n), color="gray", ls="--", label="ideal (e = 1)")
    plt.xlabel("coefficient of restitution e")
    plt.ylabel("collisions counted")
    plt.title(f"mass ratio 100^{n}, friction {friction}")
    plt.legend()
    if show:
        plt.show()


# ---------------------------------------------------------------------------
# Try things out: edit this part
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. The ideal case: should print 3, 31, 314, 3141, ...
    for n in range(0, 6):
        print(f"mass ratio 100^{n}: {count_collisions(n)} collisions")

    # 2. Inelastic collisions, with the safety exits
    print(simulate(3, e=0.9)["status"])        # -> "collapsed"

    # 3. Friction
    print(simulate(2, friction=0.1)["count"])

    # 4. Visual (uncomment one line at a time)
    animate(simulate(1, record=True)["snapshots"], title="ideal, mass ratio 100")
    animate(simulate(1, e=0.95, record=True)["snapshots"], title="e = 0.95")

    # 5. Spring-and-damper contacts (slower!)
    print(simulate_spring(n=1, e=1.0)["count"])
    animate(simulate_spring(n=1, e=0.95, record=True)["snapshots"], title="spring contact")

    # 6. Graph of count vs e
    sweep_restitution(2, [1.0, 0.999, 0.99, 0.98, 0.95, 0.9])
