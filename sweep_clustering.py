"""Angular Sweep (sweep clustering) for the *truck customer partition*.

This module replaces KMeans **only** for the decision "which customers belong to
which truck's delivery system".  Everything downstream -- truck route
construction, drone sortie creation, the energy / SOC model, truck-drone
synchronisation, ALNS, RVND and simulated annealing -- is untouched and still
reads exactly the data structures it read before (customer ids ``1..n``,
unchanged demands, unchanged distance units).

Why sweep instead of KMeans
---------------------------
KMeans minimises intra-cluster Euclidean distance, which produces *radially*
compact but *angularly interleaved* clusters: two trucks can end up driving
through each other's territory, so their routes cross many times.  An angular
sweep cuts the plane into sectors around the depot, so each truck owns one
relatively continuous wedge and the initial truck routes cross far less.

The load of a sector is **not** simply "number of customers": customer demands
differ, and customers with ``demand > drone_payload_capacity`` can only ever be
served by the truck, so a sector full of heavy truck-only customers is much
more expensive than a sector full of light drone-eligible ones.  A balanced
sweep therefore splits on a *load score*, not on a customer count.

A single sweep is also biased by where it starts: a customer at 1 degree and one
at 359 degrees are immediate neighbours but end up at opposite ends of the
sorted sequence.  The fix is to try several rotations of the sweep start and
keep the rotation whose **fully evaluated solution** (truck + drone + energy +
SOC) has the smallest makespan -- not the geometrically prettiest partition.

Units: coordinates m, demands kg, energy Wh (see ``model.py``).

Public API
----------
``compute_polar_features``      angle / radius of one customer w.r.t. the depot
``angular_sort``                customer indices sorted by rotated angle, radius
``balanced_sweep_partition``    one rotation -> ``num_trucks`` continuous sectors
``generate_sweep_partitions``   all rotations -> list of candidate partitions
``build_best_sweep_initial_solution``  all rotations -> best fully evaluated solution
``describe_partition``          per-cluster statistics for logging / tests
``format_sweep_log``            human readable initialisation log (section 21)
"""

import math

__all__ = [
    'INITIAL_CLUSTERING_METHOD', 'USE_ANGULAR_SWEEP', 'USE_KMEANS_INITIALIZATION',
    'SWEEP_NUM_ROTATIONS', 'SWEEP_DEMAND_WEIGHT', 'SWEEP_TRUCK_ONLY_WEIGHT',
    'SWEEP_EXCLUDE_DEPOT_DUPLICATES', 'SWEEP_LOG_TO_CONSOLE',
    'TWO_PI',
    'set_initial_clustering_method', 'get_initial_clustering_method',
    'compute_polar_features', 'angular_sort', 'sweep_rotations',
    'minimal_arc_span_rad', 'customer_load_score', 'find_depot_duplicate_indices',
    'balanced_sweep_partition', 'generate_sweep_partitions',
    'describe_partition', 'format_sweep_log',
    'build_best_sweep_initial_solution', 'SweepInitialSolution',
]

TWO_PI = 2.0 * math.pi

# --------------------------------------------------------------------------
# Configuration (section 20).  Centralised here so an experiment can compare
# initialisation strategies in one place.  Read *at call time* (never cached at
# import time) so ``set_initial_clustering_method`` can switch strategies inside
# a running process.
# --------------------------------------------------------------------------
INITIAL_CLUSTERING_METHOD = "sweep"   # "sweep" | "kmeans"
USE_ANGULAR_SWEEP = True
USE_KMEANS_INITIALIZATION = False

# Number of evenly spaced sweep starting angles tried per instance.
SWEEP_NUM_ROTATIONS = 1

# Load-score weights: 1.0 + w_demand * normalised_demand + w_truck_only * truck_only
SWEEP_DEMAND_WEIGHT = 0.0
SWEEP_TRUCK_ONLY_WEIGHT = 0.0

# Defensive: if a dataset ever carries the depot as a duplicate row inside the
# customer list (Meng-style "row 0 = start depot, row n+1 = duplicate end
# depot"), that row must not take part in the angular sort or the balancing.
SWEEP_EXCLUDE_DEPOT_DUPLICATES = True

# Print the sweep initialisation log to stdout.  Runners keep the console to a
# couple of number lines, so the log is normally written to a file instead.
SWEEP_LOG_TO_CONSOLE = False


def set_initial_clustering_method(method):
    """Switch the truck-partition strategy at runtime (comparison experiments).

    ``"sweep"`` uses :func:`balanced_sweep_partition`, ``"kmeans"`` restores the
    legacy KMeans partition.  Returns the previous value.
    """
    global INITIAL_CLUSTERING_METHOD, USE_ANGULAR_SWEEP, USE_KMEANS_INITIALIZATION
    if method not in ("sweep", "kmeans"):
        raise ValueError("initial clustering method must be 'sweep' or 'kmeans', got %r" % (method,))
    previous = INITIAL_CLUSTERING_METHOD
    INITIAL_CLUSTERING_METHOD = method
    USE_ANGULAR_SWEEP = (method == "sweep")
    USE_KMEANS_INITIALIZATION = (method == "kmeans")
    return previous


def get_initial_clustering_method(method=None):
    """Resolve a method name; ``None`` means "use the configured default"."""
    return method or INITIAL_CLUSTERING_METHOD


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def compute_polar_features(customer, depot):
    """Polar coordinates of one customer seen from the depot.

    ``angle`` is measured counter-clockwise from the depot's +x axis and mapped
    into ``[0, 2*pi)``; ``radius`` is the straight-line distance to the depot.
    """
    dx = float(customer[0]) - float(depot[0])
    dy = float(customer[1]) - float(depot[1])
    angle = math.atan2(dy, dx) % TWO_PI
    radius = math.hypot(dx, dy)
    return angle, radius


def sweep_rotations(num_rotations=None):
    """Deterministic, evenly spaced sweep starting angles (no randomness)."""
    count = SWEEP_NUM_ROTATIONS if num_rotations is None else int(num_rotations)
    count = max(1, count)
    return [TWO_PI * index / count for index in range(count)]


def find_depot_duplicate_indices(customer_coords, depot, tolerance=1e-9):
    """Indices of customer rows sitting exactly on the depot (defensive).

    The project's loader keeps the depot in its own ``depot`` field, so
    ``locations`` never contains it and this returns ``[]``.  It exists so that
    a Meng-format dataset carrying a duplicate depot row (row 0 = start depot,
    last row = duplicate end depot) cannot silently distort the sweep.
    """
    depot_x, depot_y = float(depot[0]), float(depot[1])
    duplicates = []
    for index, customer in enumerate(customer_coords):
        if (abs(float(customer[0]) - depot_x) <= tolerance
                and abs(float(customer[1]) - depot_y) <= tolerance):
            duplicates.append(index)
    return duplicates


def customer_load_score(demand, max_demand, drone_payload_capacity,
                        demand_weight=None, truck_only_weight=None):
    """Clustering load of one customer (section 5).

    ``1.0`` baseline + a demand term + a penalty for customers the drone can
    never lift, so sectors with heavy / truck-only customers receive fewer
    customers.  This is purely a *clustering* weight: the customer's real
    ``demand`` is never modified.
    """
    w_demand = SWEEP_DEMAND_WEIGHT if demand_weight is None else float(demand_weight)
    w_truck_only = (
        SWEEP_TRUCK_ONLY_WEIGHT if truck_only_weight is None else float(truck_only_weight)
    )
    demand = float(demand)
    if max_demand > 1e-12:
        normalized_demand = demand / float(max_demand)
    else:
        normalized_demand = 0.0
    truck_only = 1.0 if demand > float(drone_payload_capacity) + 1e-12 else 0.0
    return 1.0 + w_demand * normalized_demand + w_truck_only * truck_only


def angular_sort(customer_coords, depot, rotation=0.0):
    """Customer indices ordered by rotated polar angle, ties broken by radius.

    ``rotation`` shifts the sweep start so the 0/360 degree seam moves; the
    rotated angle stays in ``[0, 2*pi)``.  A customer closer to the depot comes
    first when two customers point in (nearly) the same direction.
    """
    decorated = []
    for index, customer in enumerate(customer_coords):
        angle, radius = compute_polar_features(customer, depot)
        rotated_angle = (angle - float(rotation)) % TWO_PI
        decorated.append((rotated_angle, radius, index))
    decorated.sort(key=lambda item: (item[0], item[1]))
    return [index for _, _, index in decorated]


# --------------------------------------------------------------------------
# Balanced, angle-continuous partition
# --------------------------------------------------------------------------
def _split_sorted_items(items, num_trucks):
    """Cut the sorted ``items`` into ``num_trucks`` contiguous, balanced blocks.

    ``items`` is a list of ``(customer_index, rotated_angle, radius, load)``.
    The cut point between two sectors compares the absolute error to the target
    load *before* and *after* taking the next customer (section 18), which is
    more stable than a plain ``load >= target`` threshold.

    Two invariants are enforced (section 6 / 7):
      * every block is a **contiguous** slice of the angularly sorted sequence;
      * every block is non-empty -- the scan may only advance while enough
        customers remain for the trucks that are still unserved, which is what
        prevents the first ``K-1`` trucks from eating the whole instance.
    """
    total = len(items)
    clusters = []
    start = 0
    remaining_load = sum(item[3] for item in items)

    for truck_index in range(num_trucks):
        trucks_left = num_trucks - truck_index
        if trucks_left <= 1:
            clusters.append(list(range(start, total)))
            break

        target_load = remaining_load / trucks_left
        # Customers that must be left over for the trucks after this one.
        reserved = trucks_left - 1
        # Taking items[end] is only legal while it still leaves `reserved`
        # customers behind, i.e. end < total - reserved.  Because the previous
        # iteration respected its own reserved count, `start` always satisfies
        # this, so each cluster is guaranteed at least one customer.
        limit = total - reserved
        if start >= limit:
            # Cannot happen while num_trucks <= total; fail loudly instead of
            # silently producing an empty sector.
            raise ValueError(
                "balanced sweep cannot fill %d sectors from %d customers"
                % (num_trucks, total)
            )

        end = start
        cluster_load = 0.0
        while end < limit:
            next_load = items[end][3]
            error_before = abs(cluster_load - target_load)
            error_after = abs(cluster_load + next_load - target_load)
            if end > start and error_after >= error_before:
                break
            cluster_load += next_load
            end += 1

        clusters.append(list(range(start, end)))
        start = end
        remaining_load -= cluster_load

    return [[items[position][0] for position in block] for block in clusters]


def balanced_sweep_partition(customer_coords, depot, num_trucks,
                            drone_payload_capacity, customer_demands,
                            rotation=0.0, demand_weight=None,
                            truck_only_weight=None,
                            exclude_depot_duplicates=None):
    """Partition customers into ``num_trucks`` angle-continuous sectors.

    Returns a list of ``num_trucks`` lists of **0-based indices into
    ``customer_coords``**.  The union of the returned lists is exactly
    ``range(len(customer_coords))`` -- every customer appears once and only once.

    The sweep decides *which truck's delivery system* a customer belongs to.  It
    deliberately does **not** decide how that customer is finally served: drone
    sortie creation, ALNS and RVND can still move a customer between the truck
    and a drone, or re-assign it entirely.
    """
    total = len(customer_coords)
    num_trucks = int(num_trucks)
    if num_trucks <= 0:
        raise ValueError("num_trucks must be >= 1")
    if total == 0:
        return [[] for _ in range(num_trucks)]

    if exclude_depot_duplicates is None:
        exclude_depot_duplicates = SWEEP_EXCLUDE_DEPOT_DUPLICATES
    if exclude_depot_duplicates:
        duplicate_indices = set(find_depot_duplicate_indices(customer_coords, depot))
    else:
        duplicate_indices = set()

    max_demand = max(float(d) for d in customer_demands) if customer_demands else 0.0
    drone_payload_capacity = float(drone_payload_capacity)

    items = []
    for index, customer in enumerate(customer_coords):
        if index in duplicate_indices:
            # A duplicate depot row has no meaningful polar angle (radius 0), so
            # it is held out of the sweep and re-attached afterwards to keep
            # customer coverage intact.
            continue
        angle, radius = compute_polar_features(customer, depot)
        rotated_angle = (angle - float(rotation)) % TWO_PI
        load = customer_load_score(
            customer_demands[index], max_demand, drone_payload_capacity,
            demand_weight=demand_weight, truck_only_weight=truck_only_weight,
        )
        items.append((index, rotated_angle, radius, load))

    # Checked against the rows that actually enter the sweep: a row held out as a
    # depot duplicate cannot fill a sector, so it must not count towards the
    # number of sectors that can be filled.
    if num_trucks > len(items):
        raise ValueError(
            "num_trucks must be <= number of customers to sweep "
            "(%d trucks, %d sweepable customers%s)"
            % (num_trucks, len(items),
               "" if not duplicate_indices else
               "; %d row(s) held out as depot duplicates" % len(duplicate_indices))
        )

    items.sort(key=lambda item: (item[1], item[2]))
    clusters = _split_sorted_items(items, num_trucks)

    if duplicate_indices:
        # Attach the held-out rows to the sector that starts at the current
        # sweep origin; they are at the depot, so this cannot skew the balance.
        clusters[0].extend(sorted(duplicate_indices))

    return clusters


def generate_sweep_partitions(customer_coords, depot, num_trucks,
                              drone_payload_capacity, customer_demands,
                              num_rotations=None):
    """One balanced partition per sweep rotation (section 9 / 10).

    Returns ``[(rotation_radians, partition), ...]`` in deterministic
    ``0, 2*pi/N, 2*(2*pi/N), ...`` order.  Sweeping itself is deterministic; no
    random seed is involved.
    """
    return [
        (rotation, balanced_sweep_partition(
            customer_coords, depot, num_trucks, drone_payload_capacity,
            customer_demands, rotation=rotation))
        for rotation in sweep_rotations(num_rotations)
    ]


# --------------------------------------------------------------------------
# Reporting (section 21)
# --------------------------------------------------------------------------
def minimal_arc_span_rad(angles):
    """Width of the smallest arc that contains every given angle.

    Rotation independent, and it does not care where the 0/360 degree seam
    falls: the span is ``2*pi`` minus the largest empty gap between consecutive
    angles.  For a contiguous sector this is exactly the arc the sector covers,
    whether or not the rotation happened to place it across the seam.  (It is a
    description, not a proof of continuity -- see ``contiguous`` below.)
    """
    if len(angles) < 2:
        return 0.0
    ordered = sorted(angles)
    gaps = [ordered[index + 1] - ordered[index] for index in range(len(ordered) - 1)]
    gaps.append(TWO_PI - (ordered[-1] - ordered[0]))
    return TWO_PI - max(gaps)


def describe_partition(partition, customer_coords, depot, drone_payload_capacity,
                       customer_demands, rotation=0.0):
    """Per-sector statistics used for the log and by the tests.

    Returns one dict per truck with the customer ids (1-based), the raw angular
    range, the arc the sector covers, its position range in the swept order and
    the truck-only / drone-eligible split.

    ``angle_min_deg`` / ``angle_max_deg`` are measured against the depot's +x
    axis; a sector wrapping the 0 degree seam makes that pair misleading (a 20
    degree sector can report min=2, max=342), so ``angle_span_deg`` -- the
    smallest arc containing the sector -- is the number to read.  Note that a
    contiguous sector can still cover a wide arc if it straddles a stretch with
    no customers in it.

    ``order_min`` / ``order_max`` / ``contiguous`` are the actual evidence: in
    the swept order, a sector produced by a sweep must occupy one uninterrupted
    run of positions.  ``contiguous`` is False only if something is wrong.
    """
    capacity = float(drone_payload_capacity)
    order = angular_sort(customer_coords, depot, rotation)
    position_of = {index: position for position, index in enumerate(order)}

    descriptions = []
    for truck_index, cluster in enumerate(partition):
        angles = [compute_polar_features(customer_coords[index], depot)[0]
                  for index in cluster]
        demands = [float(customer_demands[index]) for index in cluster]
        truck_only = sum(1 for demand in demands if demand > capacity + 1e-12)
        positions = sorted(position_of[index] for index in cluster
                           if index in position_of)
        contiguous = bool(positions) and positions == list(
            range(positions[0], positions[-1] + 1))
        descriptions.append({
            'truck': truck_index + 1,
            'customer_ids': [int(index) + 1 for index in cluster],
            'count': len(cluster),
            'total_demand': sum(demands),
            'truck_only': truck_only,
            'drone_eligible': len(cluster) - truck_only,
            'angle_min_deg': math.degrees(min(angles)) if angles else None,
            'angle_max_deg': math.degrees(max(angles)) if angles else None,
            'angle_span_deg': (
                math.degrees(minimal_arc_span_rad(angles)) if angles else None),
            'order_min': positions[0] if positions else None,
            'order_max': positions[-1] if positions else None,
            'order_total': len(order),
            'contiguous': contiguous,
        })
    return descriptions


def format_sweep_log(depot, customer_coords, num_trucks, drone_payload_capacity,
                     customer_demands, candidates, best, num_rotations):
    """Render the initialisation log of section 21 as a list of lines.

    ``candidates`` is ``[(rotation, partition, makespan), ...]`` and ``best`` is
    the winning :class:`SweepInitialSolution`.
    """
    lines = []
    lines.append('[SWEEP INITIALIZATION]')
    lines.append('')
    lines.append('Depot:')
    lines.append('(%.4f, %.4f)' % (float(depot[0]), float(depot[1])))
    lines.append('')
    lines.append('Customers:')
    lines.append('%d' % len(customer_coords))
    lines.append('')
    lines.append('Trucks:')
    lines.append('%d' % int(num_trucks))
    lines.append('')
    lines.append('Rotations tested:')
    lines.append('%d' % int(num_rotations))
    lines.append('')
    lines.append('Load score weights: demand=%.3f truck_only=%.3f (base 1.0)'
                 % (SWEEP_DEMAND_WEIGHT, SWEEP_TRUCK_ONLY_WEIGHT))
    lines.append('')

    for rotation, partition, makespan in candidates:
        descriptions = describe_partition(
            partition, customer_coords, depot, drone_payload_capacity, customer_demands,
            rotation=rotation)
        lines.append('Rotation %.0f deg:' % math.degrees(rotation))
        for info in descriptions:
            lines.append('  Truck %d customers = %d' % (info['truck'], info['count']))
            lines.append('  Truck %d demand = %.4f kg' % (info['truck'], info['total_demand']))
            lines.append('  Truck %d truck-only = %d' % (info['truck'], info['truck_only']))
        lines.append('  Initial makespan = %.4f s' % makespan)
        lines.append('')

    lines.append('Best rotation:')
    lines.append('%.0f deg' % math.degrees(best.rotation))
    lines.append('')
    lines.append('Best initial makespan:')
    lines.append('%.4f s' % best.makespan)
    lines.append('')
    lines.append('Best partition:')
    for info in describe_partition(best.partition, customer_coords, depot,
                                   drone_payload_capacity, customer_demands,
                                   rotation=best.rotation):
        if info['count']:
            angle_range = '[%.2f deg, %.2f deg]' % (info['angle_min_deg'], info['angle_max_deg'])
            angle_span = '%.2f deg' % info['angle_span_deg']
            sweep_run = 'positions %d..%d of %d, contiguous=%s' % (
                info['order_min'], info['order_max'], info['order_total'],
                info['contiguous'])
        else:
            angle_range = '[]'
            angle_span = '[]'
            sweep_run = '[]'
        lines.append('  Truck %d:' % info['truck'])
        lines.append('    customer ids   = %s' % (info['customer_ids'],))
        lines.append('    angle range    = %s (w.r.t. 0 deg)' % angle_range)
        lines.append('    angular span   = %s (smallest arc containing the sector)'
                     % angle_span)
        lines.append('    sweep order    = %s' % sweep_run)
        lines.append('    customer count = %d' % info['count'])
        lines.append('    total demand   = %.4f kg' % info['total_demand'])
        lines.append('    truck-only     = %d' % info['truck_only'])
        lines.append('    drone-eligible = %d' % info['drone_eligible'])
    return lines


# --------------------------------------------------------------------------
# Multi-start driver
# --------------------------------------------------------------------------
class SweepInitialSolution:
    """Result of :func:`build_best_sweep_initial_solution`."""

    def __init__(self, vehicle_routes, drone_routes, makespan, rotation,
                 partition, candidates, log_lines):
        self.vehicle_routes = vehicle_routes
        self.drone_routes = drone_routes
        self.makespan = makespan
        self.rotation = rotation
        self.rotation_degrees = math.degrees(rotation)
        self.partition = partition
        # (rotation, partition, makespan) for every rotation that was tried.
        self.candidates = candidates
        self.log_lines = log_lines

    def __iter__(self):
        # Allow ``veh, dro, makespan = build_best_sweep_initial_solution(...)``
        # so the existing three-value call sites keep working unchanged.
        return iter((self.vehicle_routes, self.drone_routes, self.makespan))


def build_best_sweep_initial_solution(customer_coords, depot, num_trucks,
                                      drone_payload_capacity, customer_demands,
                                      build_solution, num_rotations=None,
                                      rotations=None):
    """Try every sweep rotation, keep the one with the smallest makespan.

    ``build_solution(partition) -> (vehicle_routes, drone_routes, makespan)``
    must be the project's *existing* pipeline: the truck route constructor
    (nearest neighbour / cheapest / farthest insertion), then the existing drone
    sortie initialisation, then the full energy / SOC / synchronisation
    evaluation.  This driver only decides *which angular partition to feed it* --
    the reported makespan is the fully evaluated one, never a geometric score.

    Returns a :class:`SweepInitialSolution` which also unpacks as a
    ``(vehicle_routes, drone_routes, makespan)`` triple.
    """
    if rotations is None:
        rotations = sweep_rotations(num_rotations)
    rotations = list(rotations)
    if not rotations:
        raise ValueError("build_best_sweep_initial_solution needs at least one rotation")

    candidates = []
    best_record = None
    for rotation in rotations:
        partition = balanced_sweep_partition(
            customer_coords, depot, num_trucks, drone_payload_capacity,
            customer_demands, rotation=rotation)
        vehicle_routes, drone_routes, makespan = build_solution(partition)
        candidates.append((rotation, partition, makespan))
        if best_record is None or makespan < best_record[2]:
            best_record = (rotation, partition, makespan,
                           vehicle_routes, drone_routes)

    rotation, partition, makespan, vehicle_routes, drone_routes = best_record
    log_lines = format_sweep_log(
        depot, customer_coords, num_trucks, drone_payload_capacity,
        customer_demands, candidates,
        SweepInitialSolution(vehicle_routes, drone_routes, makespan, rotation,
                             partition, candidates, []),
        len(rotations))
    return SweepInitialSolution(vehicle_routes, drone_routes, makespan, rotation,
                                partition, candidates, log_lines)
