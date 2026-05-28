#!/usr/bin/env python3
"""Test SROI gripper: open/close characterization via DAMIAO DM-FDCAN serial bridge.

Calibrates by default (auto-homing).  Use --calibrate manual for interactive
user-assisted calibration, or --calibrate none to skip.

Modes:
  cycle   Run N open/close cycles, measure timing and repeatability (default)
  sweep   Visit N evenly-spaced positions across the full range

Metrics collected per move:
  - Settling time (position within tolerance)
  - Steady-state position error
  - Steady-state torque
  - Motor temperature

Usage:
  python tests/test_gripper.py --port /dev/ttyACM0
  python tests/test_gripper.py --port /dev/ttyACM0 --calibrate auto --cycles 10
  python tests/test_gripper.py --port /dev/ttyACM0 --calibrate manual
  python tests/test_gripper.py --port /dev/ttyACM0 --calibrate none
  python tests/test_gripper.py --port /dev/ttyACM0 --mode sweep --sweep_points 20
"""

import argparse
import logging
import statistics
import sys
import time
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Helpers
# ===========================================================================


@dataclass
class MoveResult:
    label: str
    target: float
    actual: float
    torque: float
    settle_time: float
    temp_mos: int

    @property
    def error(self) -> float:
        return abs(self.actual - self.target)


def _linspace(start: float, stop: float, n: int) -> list[float]:
    if n < 2:
        return [start]
    return [start + (stop - start) * i / (n - 1) for i in range(n)]


def _wait_settle(g, target: float, kp: float, kd: float,
                 tolerance: float = 0.03, timeout: float = 2.0) -> tuple:
    """Wait for motor position to settle within *tolerance* of *target*.

    Repeatedly sends MIT commands (the DM4310 only responds with state
    when it receives a command).  Returns (MotorState, settle_time_seconds).
    """
    t0 = time.monotonic()
    deadline = t0 + timeout
    while time.monotonic() < deadline:
        state = g.send_command(kp, kd, target)
        if abs(state.position - target) < tolerance:
            return state, time.monotonic() - t0
        time.sleep(0.02)
    actual = g.state.position
    torque = g.state.torque
    error = abs(actual - target)
    hint = ""
    if abs(torque) > 1.0:
        hint = (
            "  Motor is pushing hard (torque={torque:.2f} Nm) — "
            "likely at a mechanical limit.\n"
            "  Try --calibrate auto or --calibrate manual."
        ).format(torque=torque)
    raise TimeoutError(
        f"Position did not settle within {timeout}s: "
        f"target={target:.3f}, actual={actual:.3f}, error={error:.3f}\n"
        f"{hint}"
    )


def _compute_stats(values: list[float]) -> dict:
    if not values:
        return {"min": float("nan"), "mean": float("nan"), "max": float("nan")}
    return {
        "min": min(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def _format_stats(name: str, stats: dict, unit: str = "") -> str:
    return (
        f"  {name:<24s} "
        f"min={stats['min']:7.3f}  "
        f"mean={stats['mean']:7.3f}  "
        f"max={stats['max']:7.3f}"
        + (f"  {unit}" if unit else "")
    )


# ===========================================================================
# Calibration
# ===========================================================================


def _calibrate_auto(g, args) -> tuple[float, float]:
    """Homing: close until torque threshold, zero encoder."""
    logger.info("Auto-calibrating: closing until torque=%.1f Nm ...",
                 args.torque_threshold)
    stop_state = g.home(
        close_direction=args.close_direction,
        torque_threshold=args.torque_threshold,
    )
    range_rad = abs(stop_state.position)
    closed_rad = 0.0
    open_rad = -range_rad
    logger.info("Calibrated: range=%.3f rad (%.1f deg), closed=%.3f, open=%.3f",
                 range_rad, range_rad * 57.2958, closed_rad, open_rad)
    return open_rad, closed_rad


def _calibrate_manual(g, args) -> tuple[float, float]:
    """Interactive calibration: user moves gripper to limits, presses Enter.

    The DM4310 only sends state when it receives a command, so we
    actively send low-kp MIT commands to poll the actual position
    (kp=0.5 is soft enough for the user to backdrive by hand).
    """
    manual_kp = 0.5
    manual_kd = 0.5

    logger.info("Manual calibration mode (kp=%.1f — backdrivable)", manual_kp)
    logger.info("  Move the gripper by hand to each limit, then press Enter.")

    def _poll_state():
        """Send a soft MIT command to trigger a motor response.
        Uses the last-known position as target so the motor doesn't jump."""
        return g.send_command(manual_kp, manual_kd, g.position)

    def _prompt(msg: str):
        input(f"\n  >>> {msg}\n  Press Enter when ready...")
        # Send several commands so the motor responds with fresh position
        for _ in range(3):
            _poll_state()
            time.sleep(0.02)

    # Initial poll to get a fresh starting position
    _poll_state()
    time.sleep(0.05)
    logger.info("  Initial position: %.3f rad", g.position)

    # Step 1: position at closed limit
    _prompt("Move gripper to FULLY CLOSED position, then press Enter")
    state = g.state
    closed_rad = state.position
    logger.info("  Recorded CLOSED: %.3f rad, torque=%.2f Nm",
                 closed_rad, state.torque)

    # Step 2: position at open limit
    _prompt("Move gripper to FULLY OPEN position, then press Enter")
    state = g.state
    open_rad = state.position
    logger.info("  Recorded OPEN:  %.3f rad, torque=%.2f Nm",
                 open_rad, state.torque)

    # Validate
    if closed_rad <= open_rad:
        logger.error(
            "closed_rad (%.3f) must be > open_rad (%.3f). "
            "Swap the limits or redo calibration.", closed_rad, open_rad
        )
        sys.exit(1)

    range_rad = closed_rad - open_rad
    logger.info("Manual calibration done: range=%.3f rad (%.1f deg)",
                 range_rad, range_rad * 57.2958)
    return open_rad, closed_rad


# ===========================================================================
# Main
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Test SROI gripper open/close",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ── Connection ──────────────────────────────────────────────────
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    # ── Control ─────────────────────────────────────────────────────
    parser.add_argument("--kp", type=float, default=10.0,
                        help="Impedance stiffness (Nm/rad)")
    parser.add_argument("--kd", type=float, default=1.0,
                        help="Impedance damping (Nm.s/rad)")
    parser.add_argument("--settle_tolerance", type=float, default=0.03,
                        help="Position tolerance for settling check (rad)")
    # ── Calibration ─────────────────────────────────────────────────
    parser.add_argument("--calibrate", choices=["auto", "manual", "none"],
                        default="auto",
                        help="Calibration mode: 'auto' = torque homing (default), "
                             "'manual' = user-assisted, 'none' = skip")
    parser.add_argument("--closed_rad", type=float, default=0.734,
                        help="Closed limit (rad) — only used with --calibrate none")
    parser.add_argument("--open_rad", type=float, default=-0.139,
                        help="Open limit (rad) — only used with --calibrate none")
    parser.add_argument("--close_direction", type=int, default=1,
                        help="Motor direction for closing (1 or -1)")
    parser.add_argument("--torque_threshold", type=float, default=2.0,
                        help="Torque threshold (Nm) for auto-calibration")
    # ── Test mode ───────────────────────────────────────────────────
    parser.add_argument("--mode", choices=["cycle", "sweep"], default="cycle",
                        help="Test mode: 'cycle' for open/close loops, "
                             "'sweep' for range scan (default: cycle)")
    parser.add_argument("--cycles", type=int, default=3,
                        help="Number of open/close cycles (cycle mode)")
    parser.add_argument("--sweep_points", type=int, default=15,
                        help="Number of positions across range (sweep mode)")
    parser.add_argument("--leave_open", action="store_true",
                        help="Leave gripper open after test")
    args = parser.parse_args()

    # ── Connect ─────────────────────────────────────────────────────
    from modules.gripper import Gripper

    g = Gripper(port=args.port)
    logger.info("Connecting to %s ...", args.port)
    try:
        g.connect()
    except Exception:
        logger.error(
            "Connection failed. Check: "
            "1) USB serial dongle plugged in?  "
            "2) 24V power on?  "
            "3) Correct port? (currently %s)", args.port
        )
        sys.exit(1)

    # ── Calibrate ───────────────────────────────────────────────────
    if args.calibrate == "auto":
        open_rad, closed_rad = _calibrate_auto(g, args)
        did_home = True
    elif args.calibrate == "manual":
        open_rad, closed_rad = _calibrate_manual(g, args)
        did_home = False
    else:
        logger.warning(
            "Skipping calibration. Zeroing at current position — "
            "cached values may be stale."
        )
        g.set_zero()
        time.sleep(0.5)
        closed_rad = args.closed_rad
        open_rad = args.open_rad
        did_home = False

    # ── Run test ────────────────────────────────────────────────────
    results: list[MoveResult] = []
    t_start = time.monotonic()

    if args.mode == "sweep":
        positions = _linspace(open_rad, closed_rad, args.sweep_points)
        logger.info("Sweeping %d positions from %.3f → %.3f rad",
                     len(positions), open_rad, closed_rad)
        for i, pos in enumerate(positions):
            g.send_command(kp=args.kp, kd=args.kd, position=pos)
            state, dt = _wait_settle(g, pos, args.kp, args.kd,
                                     tolerance=args.settle_tolerance)
            results.append(MoveResult(
                label=f"{'OPEN' if i == 0 else 'CLOSE' if i == len(positions) - 1 else f'PT{i}'}",
                target=pos, actual=state.position, torque=state.torque,
                settle_time=dt, temp_mos=state.temp_mos,
            ))
            logger.info("  [%2d/%2d] %5s  target=%+.3f  actual=%+.3f  err=%+.3f  "
                         "torque=%+.2f  dt=%.3fs  temp=%dC",
                         i + 1, len(positions), results[-1].label,
                         pos, state.position, results[-1].error,
                         state.torque, dt, state.temp_mos)

    else:  # cycle
        logger.info("Cycling %d times (kp=%.1f, kd=%.1f)", args.cycles, args.kp, args.kd)
        logger.info("  Range: open=%+.3f rad  ←→  closed=%+.3f rad", open_rad, closed_rad)
        for cycle in range(args.cycles):
            # ── Open ──
            g.send_command(kp=args.kp, kd=args.kd, position=open_rad)
            state, dt = _wait_settle(g, open_rad, args.kp, args.kd,
                                     tolerance=args.settle_tolerance)
            results.append(MoveResult(
                label="OPEN", target=open_rad, actual=state.position,
                torque=state.torque, settle_time=dt, temp_mos=state.temp_mos,
            ))
            logger.info("  [%d/%d] OPEN   target=%+.3f  actual=%+.3f  err=%+.3f  "
                         "torque=%+.2f  dt=%.3fs  temp=%dC",
                         cycle + 1, args.cycles, open_rad, state.position,
                         results[-1].error, state.torque, dt, state.temp_mos)

            # ── Close ──
            g.send_command(kp=args.kp, kd=args.kd, position=closed_rad)
            state, dt = _wait_settle(g, closed_rad, args.kp, args.kd,
                                     tolerance=args.settle_tolerance)
            results.append(MoveResult(
                label="CLOSE", target=closed_rad, actual=state.position,
                torque=state.torque, settle_time=dt, temp_mos=state.temp_mos,
            ))
            logger.info("  [%d/%d] CLOSE  target=%+.3f  actual=%+.3f  err=%+.3f  "
                         "torque=%+.2f  dt=%.3fs  temp=%dC",
                         cycle + 1, args.cycles, closed_rad, state.position,
                         results[-1].error, state.torque, dt, state.temp_mos)

    elapsed = time.monotonic() - t_start

    # ── Summary ─────────────────────────────────────────────────────
    if args.mode == "cycle":
        open_results  = [r for r in results if r.label == "OPEN"]
        close_results = [r for r in results if r.label == "CLOSE"]

        open_times   = [r.settle_time for r in open_results]
        close_times  = [r.settle_time for r in close_results]
        open_errors  = [r.error for r in open_results]
        close_errors = [r.error for r in close_results]
        open_torques = [r.torque for r in open_results]
        close_torques = [r.torque for r in close_results]
        temps = [r.temp_mos for r in results]

        print()
        print("=" * 68)
        print("  Gripper Test Summary")
        print("=" * 68)
        print(f"  Cycles: {args.cycles}  |  Range: {open_rad:+.3f} → {closed_rad:+.3f} rad "
              f"({closed_rad - open_rad:.3f} rad, {(closed_rad - open_rad) * 57.2958:.1f} deg)")
        print(f"  Calibration: {args.calibrate}  |  "
              f"Gains: kp={args.kp:.1f}, kd={args.kd:.1f}  |  "
              f"Tolerance: {args.settle_tolerance:.3f} rad  |  "
              f"Duration: {elapsed:.1f}s")
        print()

        print("  Settling time (s):")
        print(_format_stats("OPEN",  _compute_stats(open_times),  "s"))
        print(_format_stats("CLOSE", _compute_stats(close_times), "s"))
        print()

        print("  Position error (rad):")
        print(_format_stats("OPEN",  _compute_stats(open_errors),  "rad"))
        print(_format_stats("CLOSE", _compute_stats(close_errors), "rad"))
        if any(e > args.settle_tolerance * 2 for e in open_errors + close_errors):
            print("  ⚠  Large position errors detected — check for mechanical binding")
        print()

        print("  Torque at limits (Nm):")
        open_tq  = _compute_stats(open_torques)
        close_tq = _compute_stats(close_torques)
        print(_format_stats("OPEN",  open_tq,  "Nm"))
        print(_format_stats("CLOSE", close_tq, "Nm"))

        # Torque sanity
        if open_tq["mean"] > 1.0:
            print("  ⚠  High torque at open — check for mechanical interference")
        if did_home and close_tq["mean"] < 0.3:
            print("  ⚠  Low torque at closed — homing may have missed the stop")
        print()

        print("  MOS temperature (C):")
        temp_stats = _compute_stats(temps)
        print(f"  {'':24s} "
              f"min={temp_stats['min']:7.1f}  "
              f"mean={temp_stats['mean']:7.1f}  "
              f"max={temp_stats['max']:7.1f}  C")
        if temps and temps[-1] - temps[0] > 15:
            print("  ⚠  Temperature rise > 15C — reduce kp or duty cycle")
        print("=" * 68)

    else:  # sweep
        errors  = [r.error for r in results]
        torques = [r.torque for r in results]
        times   = [r.settle_time for r in results]

        print()
        print("=" * 68)
        print("  Gripper Sweep Summary")
        print("=" * 68)
        print(f"  Points: {len(results)}  |  "
              f"Range: {open_rad:+.3f} → {closed_rad:+.3f} rad "
              f"({closed_rad - open_rad:.3f} rad)")
        print(f"  Calibration: {args.calibrate}  |  "
              f"Gains: kp={args.kp:.1f}, kd={args.kd:.1f}  |  "
              f"Tolerance: {args.settle_tolerance:.3f} rad  |  "
              f"Duration: {elapsed:.1f}s")
        print()
        print(_format_stats("Position error", _compute_stats(errors), "rad"))
        print(_format_stats("Settling time",  _compute_stats(times),  "s"))
        print(_format_stats("Torque",         _compute_stats(torques), "Nm"))
        print("=" * 68)

    # ── Cleanup ─────────────────────────────────────────────────────
    if args.leave_open:
        logger.info("Leaving gripper open (--leave_open)")
        g.send_command(kp=args.kp, kd=args.kd, position=open_rad)
        time.sleep(0.3)

    g.disconnect()
    logger.info("Done")


if __name__ == "__main__":
    main()
