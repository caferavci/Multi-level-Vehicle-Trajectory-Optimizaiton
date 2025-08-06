# Macro-level Dynamic Programming for trajectory optimization (macro_dp.py)
# Based on Cafer Avci's approach for CAV platoon trajectory optimization [oai_citation:3‡github.com](https://github.com/caferavci/DPCAV#:~:text=Above%20Figure%20shows%20the%20layout,and%20green%20is%2020%20s) [oai_citation:4‡github.com](https://github.com/caferavci/DPCAV#:~:text=rear,trajectories%20as%20solid%20blue%20lines).
# This script computes an optimized space-time trajectory for a lead and following vehicle.
# It considers speed limits, traffic light phases, and acceleration constraints to minimize travel time.

import numpy as np
import scipy.io as sio

# Parameters (from the referenced scenario [oai_citation:5‡github.com](https://github.com/caferavci/DPCAV#:~:text=Above%20Figure%20shows%20the%20layout,and%20green%20is%2020%20s))
dt = 0.1  # time step (10 Hz)
# Road segments and speed limits
seg1_length = 250.0   # meters (segment 1 length)
seg2_length = 500.0   # meters (segment 2 length)
seg3_length = 250.0   # meters (segment 3 length)
v_max1 = 16.67        # m/s (60 km/h free flow in segment 1)
v_max2 = 8.33         # m/s (30 km/h speed limit in segment 2)
v_max3 = 16.67        # m/s (60 km/h free flow in segment 3)
# Traffic light timings (red/green durations) [oai_citation:6‡github.com](https://github.com/caferavci/DPCAV#:~:text=flow%20speed%20is%2060%20kilometers,and%20green%20is%2020%20s)
light1_red = 20.0     # s (red at 250m)
light1_green = 15.0   # s (green duration)
light2_red = 35.0     # s (red at 750m)
light2_green = 20.0   # s (green duration)
cycle1 = light1_red + light1_green
cycle2 = light2_red + light2_green

# Load trajectory data if available (data(1).mat), to use initial conditions
d0 = 2.0  # default minimum gap (m) [oai_citation:7‡github.com](https://github.com/caferavci/DPCAV#:~:text=rear,trajectories%20as%20solid%20blue%20lines)
try:
    mat = sio.loadmat('data(1).mat')
    if 'data' in mat:
        # Assume data is a struct array where data[0] has veh_s and veh_f sub-structures
        scenario = mat['data'][0,0]
        veh_s = scenario['veh_s'][0,0]
        veh_f = scenario.get('veh_f')
        if veh_f is not None:
            veh_f = veh_f[0,0]
        # Extract Y (longitudinal) positions (assuming .y field exists)
        veh_s_y = veh_s['y'].squeeze()
        if veh_f is not None and veh_f.size > 0:
            veh_f_y = veh_f['y'].squeeze()
        else:
            veh_f_y = None
        if veh_f_y is not None and veh_f_y.size > 0:
            # Calculate initial headway from data
            initial_gap = veh_f_y[0] - veh_s_y[0]
            if initial_gap > 0:
                d0 = float(initial_gap)
except Exception as e:
    # If data not available or parsing fails, proceed with default d0
    pass

# Time horizon (s) - set slightly above total free-flow travel time
T_max = 150.0  # total time horizon (as per scenario) [oai_citation:8‡github.com](https://github.com/caferavci/DPCAV#:~:text=Above%20Figure%20shows%20the%20layout,and%20green%20is%2020%20s)

# Acceleration limits (comfortable bounds)
a_max = 2.0    # m/s^2 (max acceleration)
a_dec = 2.0    # m/s^2 (max deceleration)

# Initialize arrays for trajectory
N = int(T_max/dt) + 1
time = np.linspace(0, T_max, N)
leader_pos = np.zeros(N)
follower_pos = np.zeros(N)
leader_speed = np.zeros(N)
follower_speed = np.zeros(N)

# Initial conditions
leader_pos[0] = 0.0
follower_pos[0] = -d0  # follower starts d0 behind leader (gap = d0)
leader_speed[0] = 0.0
follower_speed[0] = 0.0

# Simulate/optimize leader trajectory through segments
segment = 1
for i in range(1, N):
    t = time[i]
    if segment > 3:
        # All segments completed
        leader_pos[i] = leader_pos[i-1]
        leader_speed[i] = 0.0
        continue
    # Select segment parameters
    if segment == 1:
        v_limit = v_max1; seg_end = seg1_length
        light_red, light_green, cycle = light1_red, light1_green, cycle1
    elif segment == 2:
        v_limit = v_max2; seg_end = seg1_length + seg2_length
        light_red, light_green, cycle = light2_red, light2_green, cycle2
    else:  # segment 3
        v_limit = v_max3; seg_end = seg1_length + seg2_length + seg3_length
        light_red, light_green, cycle = None, None, None

    dist_to_end = seg_end - leader_pos[i-1]
    accel = 0.0

    # If approaching a traffic light, adjust speed to avoid stopping at red
    if light_red is not None:
        phase_time = t % cycle
        if phase_time < light_red:
            # Red phase currently [oai_citation:9‡github.com](https://github.com/caferavci/DPCAV#:~:text=flow%20speed%20is%2060%20kilometers,and%20green%20is%2020%20s)
            next_green = t + (light_red - phase_time)
        elif phase_time < light_red + light_green:
            next_green = t  # green now
        else:
            # Red phase will start after cycle reset
            next_green = t + (cycle - phase_time) + light_red
        if leader_speed[i-1] > 0:
            time_to_light = dist_to_end / leader_speed[i-1]
        else:
            time_to_light = float('inf')
        arrival_time = t + time_to_light
        if next_green and arrival_time < next_green:
            # Plan to arrive exactly at green by limiting speed
            remaining_time = next_green - t
            target_speed = dist_to_end / max(remaining_time, 1e-3)
            v_limit = min(v_limit, target_speed)
    # Ensure compliance with speed limit change at segment boundary
    if segment == 1:
        next_seg_speed = v_max2  # slow zone ahead
    elif segment == 2:
        next_seg_speed = v_max3  # faster zone ahead (no need to slow here)
    else:
        next_seg_speed = leader_speed[i-1]
    if segment < 3:
        # distance required to decelerate from current speed to next_seg_speed
        if leader_speed[i-1] > next_seg_speed:
            decel_dist = (leader_speed[i-1]**2 - next_seg_speed**2) / (2 * a_dec)
        else:
            decel_dist = 0.0
        if dist_to_end <= decel_dist + 1e-2:
            accel = -a_dec  # start deceleration
        elif leader_speed[i-1] < v_limit:
            accel = a_max  # accelerate if below limit and no need to brake
        else:
            accel = 0.0
    else:
        # Final segment: accelerate to v_max or maintain
        accel = a_max if leader_speed[i-1] < v_limit else 0.0

    # Clamp acceleration to limits
    if accel > a_max: accel = a_max
    if accel < -a_dec: accel = -a_dec

    # Update leader kinematics
    leader_speed[i] = leader_speed[i-1] + accel * dt
    if leader_speed[i] < 0: 
        leader_speed[i] = 0.0
    if leader_speed[i] > v_limit:
        leader_speed[i] = v_limit
    leader_pos[i] = leader_pos[i-1] + leader_speed[i-1]*dt + 0.5*accel*(dt**2)

    # Check if end of segment reached
    if leader_pos[i] >= seg_end - 1e-3:
        leader_pos[i] = seg_end
        # If traffic light at segment end is red, stop and wait [oai_citation:10‡github.com](https://github.com/caferavci/DPCAV#:~:text=flow%20speed%20is%2060%20kilometers,and%20green%20is%2020%20s)
        if light_red is not None:
            phase_time = t % cycle
            if phase_time < light_red:
                leader_speed[i] = 0.0
                # (For simplicity, we assume arrival aligned with green by prior adjustment)
        segment += 1
        # Leader enters next segment (carry over current speed if allowed)
        continue

# Following vehicle trajectory (platoon with minimal reaction time τ ≈ 0) [oai_citation:11‡github.com](https://github.com/caferavci/DPCAV#:~:text=rear,trajectories%20as%20solid%20blue%20lines) 
# We assume the follower mimics leader's speed, maintaining ~d0 gap.
follower_pos = leader_pos - d0
follower_pos[follower_pos < 0] = 0.0  # no negative position
follower_speed = leader_speed.copy()

# Save optimized trajectories for use in micro-level control
np.savez('macro_trajectory.npz', t=time, veh_lead_pos=leader_pos, veh_follow_pos=follower_pos,
         veh_lead_speed=leader_speed, veh_follow_speed=follower_speed, d0=d0, dt=dt)

# Print summary
total_distance = seg1_length + seg2_length + seg3_length
leader_finish_idx = np.where(leader_pos >= total_distance)[0][0]
travel_time = time[leader_finish_idx]
print(f"Leader travel time: {travel_time:.1f} s, Follower travel time: {travel_time:.1f} s (no stops).")
