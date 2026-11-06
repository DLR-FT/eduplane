# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import random
import timeit
from dataclasses import dataclass

import numpy
import pandas
from tqdm import tqdm

from eduplane.model.ext_dubins_airplane import ExtDubinsAirplane
from eduplane.utils import logger


@dataclass
class Sample2D:
    x_g: float
    y_g: float
    chi: float


@dataclass
class Sample3D:
    x_g: float
    y_g: float
    z_g: float
    chi: float


def random_value(min_value, max_value):
    return min_value + (max_value-min_value)*random.random()


def get_samples_2D(
        xy_dist_values=[100, 200, 500],
        angles=[0, math.radians(90),
                math.radians(180),
                math.radians(270)]):
    samples = []
    for xy_dist in xy_dist_values:
        x_coordinates = [-xy_dist, 0, xy_dist]
        y_coordinates = [-xy_dist, 0, xy_dist]
        samples.extend(
            [
                Sample2D(x, y, chi) for x, y, chi in itertools.product(
                    x_coordinates, y_coordinates, angles)
            ])
    return samples


def get_samples_2D_randomized(num_samples, xy_dist_bounds=[100, 500]):
    samples = [
        Sample2D(
            random_value(xy_dist_bounds[0], xy_dist_bounds[1]),
            random_value(xy_dist_bounds[0], xy_dist_bounds[1]),
            random_value(0, 2*math.pi)) for _ in range(num_samples)
    ]
    return samples


def get_samples_3D(
        xy_dist_values=[100, 200, 500],
        z_dist_values=[0, 50, 100],
        angles=[0, math.radians(90),
                math.radians(180),
                math.radians(270)]):
    samples = []
    for xy_dist in xy_dist_values:
        for z_dist in z_dist_values:
            x_coordinates = [-xy_dist, 0, xy_dist]
            y_coordinates = [-xy_dist, 0, xy_dist]
            z_coordinates = [-z_dist, 0, z_dist]
            samples.extend(
                [
                    Sample3D(x, y, z, chi)
                    for x, y, z, chi in itertools.product(
                        x_coordinates, y_coordinates, z_coordinates, angles)
                ])
    return samples


def get_samples_3D_randomized(
        num_samples, xy_dist_bounds=[100, 500], z_dist_bounds=[-200, 200]):
    samples = [
        Sample3D(
            random_value(xy_dist_bounds[0], xy_dist_bounds[1]),
            random_value(xy_dist_bounds[0], xy_dist_bounds[1]),
            random_value(z_dist_bounds[0], z_dist_bounds[1]),
            random_value(0, 2*math.pi)) for _ in range(num_samples)
    ]
    return samples


def get_test_cases_2D(
        turn_rate=20/100,
        airspeed=20,
        wind_speed=5,
        wind_dir=math.pi*0.25,
        xy_dist=[100, 200, 500],
        path_types=["LSL", "RSR", "LSR", "RSL"],
        angles=[0, math.radians(90),
                math.radians(180),
                math.radians(270)]):

    samples = get_samples_2D(xy_dist, angles)
    start_samples = samples
    goal_samples = samples

    test_cases = []
    for start in start_samples:
        # TODO: As LSL/RSL paths may return zero-length horizontal paths, we have to filter out cases were start and
        # goal horizontal coordinates overlap - these cases should be covered by forcing CSC paths of lengths > 0.
        for goal in [g for g in goal_samples
                     if g.x_g != start.x_g and g.y_g != start.y_g]:
            for path_type in path_types:
                test_cases.append(
                    dict(
                        start_x=start.x_g,
                        start_y=start.y_g,
                        start_dir=start.chi,
                        goal_x=goal.x_g,
                        goal_y=goal.y_g,
                        goal_dir=goal.chi,
                        airspeed=airspeed,
                        turn_rate=turn_rate,
                        wind_speed=wind_speed,
                        wind_dir=wind_dir,
                        path_type=path_type,
                    ))
    return test_cases


def get_test_cases_2D_randomized(
    num_test_cases,
    turn_rate_bounds=(10/100, 30/50),
    airspeed_bounds=(10, 30),
    wind_speed_bounds=(0, 10),
    xy_dist_bounds=[100, 500],
    path_types=["LSL", "RSR", "LSR", "RSL"],
    min_air2windspeed_ratio=1.5,
):

    samples = get_samples_2D_randomized(num_test_cases, xy_dist_bounds)
    start_samples = samples
    goal_samples = samples

    test_cases = []
    while len(test_cases) < num_test_cases:
        airspeed = random_value(airspeed_bounds[0], airspeed_bounds[1])
        wind_speed = random_value(wind_speed_bounds[0], wind_speed_bounds[1])
        if airspeed < wind_speed*min_air2windspeed_ratio:
            continue
        start = random.sample(samples, 1)[0]
        goal = random.sample(samples, 1)[0]
        if start == goal:
            continue
        test_cases.append(
            dict(
                start_x=start.x_g,
                start_y=start.y_g,
                start_dir=random_value(0, 2*math.pi),
                goal_x=goal.x_g,
                goal_y=goal.y_g,
                goal_dir=random_value(0, 2*math.pi),
                airspeed=airspeed,
                turn_rate=random_value(
                    turn_rate_bounds[0], turn_rate_bounds[1]),
                wind_speed=wind_speed,
                wind_dir=random_value(0, 2*math.pi),
                path_type=random.sample(path_types, 1)[0],
            ))
    return test_cases


def get_test_cases_3D(
        turn_rate=20/100,
        airspeed=20,
        wind_speed=5,
        wind_dir=math.pi*0.25,
        xy_dist=[100, 200, 500],
        z_dist=[0, 50, 100],
        path_types=["LSL", "RSR", "LSR", "RSL"],
        angles=[0, math.radians(90),
                math.radians(180),
                math.radians(270)]):

    samples = get_samples_3D(xy_dist, z_dist, angles)
    start_samples = samples
    goal_samples = samples

    test_cases = []
    for start in start_samples:
        # TODO: As LSL/RSL paths may return zero-length horizontal paths, we have to filter out cases were start and
        # goal horizontal coordinates overlap - these cases should be covered by forcing CSC paths of lengths > 0.
        for goal in [g for g in goal_samples
                     if g.x_g != start.x_g and g.y_g != start.y_g]:
            for path_type in path_types:
                test_cases.append(
                    dict(
                        start_x=start.x_g,
                        start_y=start.y_g,
                        start_z=start.z_g,
                        start_dir=start.chi,
                        goal_x=goal.x_g,
                        goal_y=goal.y_g,
                        goal_z=goal.z_g,
                        goal_dir=goal.chi,
                        airspeed=airspeed,
                        turn_rate=turn_rate,
                        wind_speed=wind_speed,
                        wind_dir=wind_dir,
                        path_type=path_type,
                    ))
    return test_cases


def get_test_cases_3D_randomized(
    num_test_cases,
    turn_rate_bounds=(10/100, 30/50),
    airspeed_bounds=(10, 30),
    wind_speed_bounds=(0, 10),
    xy_dist_bounds=[100, 500],
    z_dist_bounds=[0, 200],
    path_types=["LSL", "RSR", "LSR", "RSL"],
    min_air2windspeed_ratio=1.5,
    min_xy2z_dist_ratio=4,
):

    samples = get_samples_3D_randomized(
        num_test_cases, xy_dist_bounds, z_dist_bounds)

    test_cases = []
    while len(test_cases) < num_test_cases:
        airspeed = random_value(airspeed_bounds[0], airspeed_bounds[1])
        wind_speed = random_value(wind_speed_bounds[0], wind_speed_bounds[1])
        if airspeed < wind_speed*min_air2windspeed_ratio:
            continue
        start = random.sample(samples, 1)[0]
        goal = random.sample(samples, 1)[0]
        if start == goal:
            continue
        xy_dist = math.sqrt(
            (goal.x_g - start.x_g)**2 + (goal.y_g - start.y_g)**2)
        if xy_dist/abs(goal.z_g - start.z_g) < min_xy2z_dist_ratio:
            continue
        test_cases.append(
            dict(
                start_x=start.x_g,
                start_y=start.y_g,
                start_z=start.z_g,
                start_dir=random_value(0, 2*math.pi),
                goal_x=goal.x_g,
                goal_y=goal.y_g,
                goal_z=goal.z_g,
                goal_dir=random_value(0, 2*math.pi),
                airspeed=airspeed,
                turn_rate=random_value(
                    turn_rate_bounds[0], turn_rate_bounds[1]),
                wind_speed=wind_speed,
                wind_dir=random_value(0, 2*math.pi),
                path_type=random.sample(path_types, 1)[0],
            ))
    return test_cases


def run_benchmark(test_cases, test_func, desc, n_times=1):
    logger.reset()
    logger.enable_rt_traces()
    runtimes = []
    print(f"Running benchmark: {desc}")
    with tqdm(total=len(test_cases), desc=desc) as pbar:
        for test_case in test_cases:
            pbar.update()
            runtimes.append(
                timeit.timeit(lambda: test_func(**test_case), number=n_times)/
                n_times)
    print(
        f"Results of benchmark in {len(test_cases)}x{n_times} runs:"
        f"\n    min: {min(runtimes):0.6f} s"
        f"\n    max: {max(runtimes):0.6f} s"
        f"\n   mean: {numpy.mean(runtimes):0.6f} s"
        f"\n median: {numpy.median(runtimes):0.6f} s")
    logger.printReport()


def collect_traces(test_cases, test_func, desc):
    """Collect runtime and value traces.
    """
    table = []
    logger.reset()
    with logger.tracing_enabled():
        with tqdm(total=len(test_cases), desc=desc) as pbar:
            for i, tc in enumerate(test_cases):
                pbar.update()
                logger.trace_id = i
                if table is not None:
                    table.append(
                        {
                            "trace_id":
                            i,
                            "path_type":
                            tc.get("path_type"),
                            "airspeed":
                            tc.get("airspeed"),
                            "wind_speed":
                            tc.get("wind_speed"),
                            "xy_dist": (
                                (tc["goal_x"] - tc["start_x"])**2 +
                                (tc["goal_y"] - tc["start_y"])**2)**0.5,
                            "z_dist": (
                                tc["goal_z"] - tc["start_z"] if "start_z" in tc
                                and "goal_z" in tc else None),
                        })
                try:
                    test_func(**tc)
                finally:
                    logger.unwind(error=True)
        return (
            logger.get_rt_traces(), logger.get_value_traces(),
            pandas.DataFrame(table))


def get_start_goal_POIs(test_case):
    """Get POI definitions for plotting start and goal positions."""
    POIs = [
        dict(
            x_g=(test_case["start_x"], test_case["start_x"]),
            y_g=(test_case["start_y"], test_case["start_y"]),
            z_g=(test_case["start_z"], test_case["start_z"]),
            # chi=test_case["start_dir"],
            label="start",
            # style="go",
            s=10,
            marker=(3, 0, math.degrees(test_case["start_dir"]) + 180),
            color="tab:cyan",
        ),
        dict(
            x_g=(test_case["goal_x"], test_case["goal_x"]),
            y_g=(test_case["goal_y"], test_case["goal_y"]),
            z_g=(test_case["goal_z"], test_case["goal_z"]),
            # chi=test_case["goal_dir"],
            label="goal",
            # style="gx",
            s=10,
            marker=(3, 0, math.degrees(test_case["goal_dir"]) + 180),
            color="tab:pink",
        ),
    ]
    if "start_z" in test_case:
        POIs[0]["z_g"] = (test_case["start_z"], test_case["start_z"])
    if "goal_z" in test_case:
        POIs[1]["z_g"] = (test_case["goal_z"], test_case["goal_z"])
    return POIs


def get_aircraft_model():
    return ExtDubinsAirplane(
        g=9.81,
        m=4,
        rho=1.225,
        S=0.98,
        C_D0=0.032,
        C_L0=0,
        C_L_min=-1.2,
        C_L_max=1.2,
        k=0.06,
        T_min=0,
        T_max=30,
        P_max=1000,
        n_min=0.2,
        n_max=4.0,
        mu_A_max=math.radians(60),
        V_A_max=50)
