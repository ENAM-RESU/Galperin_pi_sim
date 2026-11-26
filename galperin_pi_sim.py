"""
Galperin's pi calc with real pysicis
Syliva Smith
11/22/2026
"""
"""
Main functions:
    simulate_spring() runs one simualtion
    ideal_count() gives the answer that Galperin's math would give, for comparason 
    collect_data() run siumalate_spring and ideal_count for increasing N values, then exports the results
    animate() make a little animation for the simulation 
  

Units:
    the block starts 2 "distence" from the wall moving at block_speed (defaults to 1 "speed")
    the smaller block starts 1 from the wall at rest and has mass 1.
    Positive velocity = moving away from the wall.
"""

#import one of the various libarys you will need
import math

#just a bit of math i wasn't sure where else to put in the code
def _sign(v):
    return (v > 0) - (v < 0)



def simulate_spring(n=1, e=0.95, friction=0.0, block_speed=1.0, stiffness=None,
                    small_block_half=0.01, block_half=0.02,
                    steps_per_contact=40, max_steps=500_000_000, record=False):
    """
    Run one experiment with mass ratio 100**n.

    e:           roughly the bounciness you want (1 = perfectly bouncy). The
                 code is writen so a single isolated collision would give
                 about this e(probably). It is approximate.
    friction:     constant deceleration of both objects (see the top of the file)
    block_speed:  starting speed of the block
    stiffness:    spring constant k. Leave as None to pick a good value
                 automatically (1e6 * 100**n * block_speed**2), which works well
                 
    small_block_half, block_half:
                 half-widths of the two objects. 
    record:      if True also return snapshots for animate()

    Returns a dict with:
      "count"       ball-wall + ball-block collisions (the number i used)
      "ball_wall", "ball_block", "block_wall"   counts of each kind
      "status"      "done", or "max_steps" if it hit the safety cap
      "snapshots"   for animate(), if record=True

    A collision is counted each time two things go from not touching to
    touching.

    Run time (seems to) grow exponentualy as N gets higher. doesn't normaly go higher than 6 so it doen't take hours to resolve
    """
    m1 = 1.0                  # ball mass
    m2 = 100.0 ** n           # block mass
    k = stiffness if stiffness is not None else 1e6 * 100.0 ** n * block_speed ** 2
    r, h = small_block_half, block_half

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

    x1, x2 = 1.0, 2.0                 # ball and block centres
    v1, v2 = 0.0, -block_speed
    in_bw = in_bb = in_kw = False
    n_bw = n_bb = n_kw = 0
    status = "done"

    def to_point(a, b):    # convert to the kinf of coordinates animate() wants 
        return (a - r, b - h - 2 * r)

    if record:
        snaps = [(*to_point(x1, x2), 0)]
    else:
        snaps = None

    for _ in range(max_steps):
        # Overlaps (positive = squeezed together)
        o_bw = r - x1
        o_bb = (x1 + r) - (x2 - h)
        o_kw = h - x2
        touching = o_bw > 0 or o_bb > 0 or o_kw > 0

        if not touching:
            # Free flight: take big steps until something is about to touch.
            # If nothing is approaching anything, no more colsisions will happen 
            times = []
            if v1 < 0:
                times.append((x1 - r) / -v1)
            if v1 > v2:
                times.append(((x2 - h) - (x1 + r)) / (v1 - v2))
            if v2 < 0:
                times.append((x2 - h) / -v2)
            # (an approach more than 100 time units away is just numerical dust)
            if not times or min(times) > 100:
                break
            dt = min(dt_coarse, max(dt_min, 0.5 * min(times)))
        else:
            dt = dt_min

        # Forces 
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

        # Friction: slow each object toward zero, but never change direction (by fliping the sign)
        if friction > 0:
            dv = friction * dt
            v1 = 0.0 if abs(v1) <= dv else v1 - dv * _sign(v1)
            v2 = 0.0 if abs(v2) <= dv else v2 - dv * _sign(v2)

        x1 += v1 * dt
        x2 += v2 * dt

        # Count new collisions (not touching -> touching)
        new_bw = (r - x1) > 0
        new_bb = ((x1 + r) - (x2 - h)) > 0
        new_kw = (h - x2) > 0
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
# The ideal answer, for comparison
# ---------------------------------------------------------------------------

_PI = "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899"


def ideal_count(n):
    """
    Exact collision count for the IDEAL case (perfectly elastic, no friction)
    at mass ratio 100**n, by formula (any n up to 50): the number of whole k
    with  k * atan(10**-n) < pi.  It gives the digits of pi: 3, 31, 314, ...
    """
    from decimal import Decimal, localcontext, ROUND_CEILING
    if n == 0:
        return 3          # the one case where pi / theta is exactly a whole number
    if n > 50:
        raise ValueError("ideal_count supports n up to 50")
    with localcontext() as ctx:
        ctx.prec = 120
        x = Decimal(10) ** (-n)
        theta, power, k = Decimal(0), x, 0     # atan(x) = x - x^3/3 + x^5/5 - ...
        while True:
            term = power / (2 * k + 1)
            theta += term if k % 2 == 0 else -term
            if term < Decimal(10) ** -110:
                break
            power *= x * x
            k += 1
        return int((Decimal(_PI) / theta).to_integral_value(ROUND_CEILING)) - 1


# ---------------------------------------------------------------------------
# Data export
# ---------------------------------------------------------------------------


def collect_data(n_values=range(0, 7), e=0.95, friction=0.1, block_speed=1.0,
                 spring_n_max=10, csv_path="pi_billiards_data.csv"):
    """
    Run the simulation for each n at a fixed e and friction, print a table,
    and save it as a CSV file you can open in Excel / Google Sheets.

    Columns: n, mass_ratio, e, friction, block_speed,
             ideal_count    (the pi digits, for comparison),
             spring_count   (your result), ball_wall, ball_block, block_wall, spring_status.
    """
    import csv
    import os

    rows = []
    print(f"e = {e}, friction = {friction}, block speed = {block_speed}")
    print(f"{'n':>2} {'mass ratio':>22} {'ideal':>14} {'result':>8} {'status':>9}")
    for n in n_values:
        ideal = ideal_count(n)
        # print the first part of the row now: the simulation can take a while
        print(f"{n:>2} {100 ** n:>22,} {ideal:>14,} ", end="", flush=True)
        if n <= spring_n_max:
            r = simulate_spring(n=n, e=e, friction=friction, block_speed=block_speed)
        else:
            r = {"count": "", "ball_wall": "", "ball_block": "",
                 "block_wall": "", "status": "skipped"}
        print(f"{r['count']:>8} {r['status']:>9}")
        rows.append({"n": n, "mass_ratio": 100 ** n, "e": e, "friction": friction,
                     "block_speed": block_speed, "ideal_count": ideal,
                     "spring_count": r["count"], "ball_wall": r["ball_wall"],
                     "ball_block": r["ball_block"], "block_wall": r["block_wall"],
                     "spring_status": r["status"]})

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {os.path.abspath(csv_path)}")
    return rows


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def animate(snapshots, title="", frames_per_collision=None, save=None,
            ball_size=0.08, block_size=0.25):
    """
    Animate a run. Pass the "snapshots" from simulate_spring(..., record=True).

    The animation is schematic: every collision gets the same amount of screen
    time, so you can follow all of them (in real time, most collisions would
    be over in a blink). The counter shows the collision count.
    Both objects are drawn as squares. Their sizes (ball_size, block_size)
    are only for drawing and do not affect the physics; set them equal if
    you want identical squares.

    save="run.gif" writes a GIF instead of opening a window.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle

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

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.set_xlim(-0.2, 2.7)
    ax.set_ylim(-0.1, 0.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)
    ax.add_patch(Rectangle((-0.12, 0), 0.12, 0.4, color="0.4"))        # wall
    ax.plot([-0.12, 2.7], [0, 0], color="0.6", lw=1)                   # floor
    ball = Rectangle((0, 0), ball_size, ball_size, color="tab:red")
    block = Rectangle((0, 0), block_size, block_size, color="tab:blue")
    ax.add_patch(ball)
    ax.add_patch(block)
    text = ax.text(1.3, 0.38, "", ha="center", fontsize=14)

    def draw(i):
        x1, x2, c = frames[i]
        ball.set_x(x1)
        block.set_x(x2 + ball_size)
        text.set_text(f"collisions: {c}")
        return ball, block, text

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=40, blit=False)
    if save:
        anim.save(save, writer="pillow", fps=25)
        plt.close(fig)
    else:
        plt.show()
    return anim

collect_data(n_values=range(0, 9), e=0.95, friction=0.1)

"""
for n in range(0,5):
    animate(simulate_spring(n, e=0.95, friction=0.1, record=True)["snapshots"],title="mass ratio 100^" + str(n))
"""
