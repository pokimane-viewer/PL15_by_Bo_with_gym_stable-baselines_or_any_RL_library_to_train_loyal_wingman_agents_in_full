"""
Full Modular PL-15 Simulation with CuPy/Numpy Support (CuPy 3.12+), Minimal Comments, 2D Visualization
"""

import math, random, sys
try:
    import cupy as cp
    if int(cp.__version__.split('.')[1]) < 12:
        raise ImportError("Requires CuPy 3.12+")
    xp = cp
except ImportError:
    import numpy as np
    xp = np

import simpy
try:
    import simpy.rt as rt
    HAS_RT = True
except ImportError:
    HAS_RT = False

try:
    import gym
    from gym import spaces
except ImportError:
    gym = None

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

C = 299792458.0
k_B = 1.380649e-23
T0 = 290.0

class RadarSeeker:
    def __init__(self, freq_hz, power_w, gain_tx=1.0, gain_rx=1.0, noise_fig=1.0, 
                 bandwidth_hz=1e6, min_detect_snr=10.0, rf_env=None, ecm=None):
        self.freq_hz = freq_hz
        self.wavelength = C / freq_hz
        self.power = power_w
        self.gain_tx = gain_tx
        self.gain_rx = gain_rx
        self.noise_fig = noise_fig
        self.bandwidth = bandwidth_hz
        self.min_detect_snr = min_detect_snr
        self.rf_env = rf_env
        self.ecm = ecm
        self.thermal_noise_power = k_B * T0 * self.bandwidth
        self.noise_power = self.thermal_noise_power * self.noise_fig

    def sense(self, missile_pos, missile_vel, target_pos, target_rcs):
        diff = target_pos - missile_pos
        R = xp.linalg.norm(diff)
        R = max(R, 1e-8)
        Pr = (self.power * self.gain_tx * self.gain_rx * (self.wavelength**2) *
              target_rcs) / (((4*math.pi)**3) * (R**4))
        if self.rf_env:
            Pr *= self.rf_env.get_path_loss_factor(R)
        Pj = 0.0
        if self.ecm:
            Pj = self.ecm.get_jamming_power(R, self.gain_rx, self.wavelength)
        total_noise = self.noise_power + Pj
        snr = Pr / (total_noise + 1e-30)
        # Doppler assumed zero unless velocities are known externally
        return {'power': Pr, 'snr': snr, 'doppler': 0.0, 'distance': R}

    def detect_target(self, mp, mv, tp, trcs):
        signal = self.sense(mp, mv, tp, trcs)
        if signal['snr'] >= self.min_detect_snr:
            return {'range': signal['distance'], 'doppler': signal['doppler'], 'snr': signal['snr']}
        return None

class SignalProcessing:
    def __init__(self, detection_threshold=10.0):
        self.detection_threshold = detection_threshold
    def process(self, radar_signal):
        if radar_signal['snr'] >= self.detection_threshold:
            return {
                'range': radar_signal['distance'],
                'velocity': radar_signal['doppler'],
                'snr': radar_signal['snr']
            }
        return None

class Target:
    def __init__(self, position, velocity, rcs_mean=5.0, rcs_model="Swerling1", maneuver=None):
        self.position = xp.array(position, dtype=float)
        self.velocity = xp.array(velocity, dtype=float)
        self.rcs_mean = rcs_mean
        self.rcs_model = rcs_model
        self.maneuver = maneuver

    def update(self, dt):
        if callable(self.maneuver):
            self.maneuver(self, dt)
        self.position += self.velocity * dt

    def get_rcs(self):
        if not self.rcs_model: return self.rcs_mean
        m = self.rcs_model.lower()
        if m.startswith("swerling1") or m.startswith("swerling2"):
            return float(xp.random.exponential(self.rcs_mean))
        elif m.startswith("swerling3") or m.startswith("swerling4"):
            shape = 2.0
            scale = self.rcs_mean / shape
            return float(xp.random.gamma(shape, scale))
        return self.rcs_mean

class RFEnvironment:
    def __init__(self, clutter_power=0.0):
        self.clutter_power = clutter_power
    def get_path_loss_factor(self, distance):
        return 1.0
    def get_clutter_noise(self):
        return self.clutter_power

class ECM:
    def __init__(self, eirp=1000.0):
        self.eirp = eirp
    def get_jamming_power(self, distance, radar_gain_rx, radar_wavelength):
        distance = max(distance, 1e-8)
        pj = (self.eirp * radar_gain_rx * (radar_wavelength**2)) / (((4*math.pi)**2) * (distance**2))
        return pj

class SensorFusion:
    def __init__(self):
        self.target_pos_est = None
        self.target_vel_est = None
        self._initialized = False
        self._last_pos = None

    def initialize(self, tpos, tvel):
        self.target_pos_est = xp.array(tpos, dtype=float)
        self.target_vel_est = xp.array(tvel, dtype=float)
        self._initialized = True
        self._last_pos = xp.array(tpos, dtype=float)

    def update(self, detection, missile_pos, dt):
        if detection is not None:
            R = detection['range']
            if not self._initialized:
                direction = xp.array([1.0, 0.0])
                self.target_pos_est = missile_pos + direction*R
                self.target_vel_est = xp.zeros_like(direction)
                self._initialized = True
                self._last_pos = xp.copy(self.target_pos_est)
            else:
                rel = self.target_pos_est - missile_pos
                d = xp.linalg.norm(rel)
                direction = rel / (d+1e-9)
                self.target_pos_est = missile_pos + direction * R
                if self._last_pos is not None:
                    dp = self.target_pos_est - self._last_pos
                    self.target_vel_est = dp / dt
                self._last_pos = xp.copy(self.target_pos_est)
        else:
            if self._initialized and self.target_vel_est is not None:
                self.target_pos_est = self.target_pos_est + self.target_vel_est * dt
                self._last_pos = xp.copy(self.target_pos_est)

class Missile:
    def __init__(self, position, velocity, max_turn_acc=30*9.81):
        self.position = xp.array(position, dtype=float)
        self.velocity = xp.array(velocity, dtype=float)
        self.max_turn_acc = max_turn_acc
    def update(self, accel_command, dt):
        a_lat = float(xp.clip(accel_command if accel_command else 0.0,
                              -self.max_turn_acc, self.max_turn_acc))
        vx, vy = float(self.velocity[0]), float(self.velocity[1])
        speed = math.hypot(vx, vy)
        if speed < 1e-3: return
        heading = math.atan2(vy, vx)
        ang_rate = a_lat / speed
        new_heading = heading + ang_rate*dt
        new_speed = speed
        self.velocity = xp.array([new_speed*math.cos(new_heading),
                                  new_speed*math.sin(new_heading)], dtype=float)
        self.position += self.velocity*dt

class Guidance:
    def __init__(self, N=4):
        self.N = N
        self.last_LOS_angle = None
    def compute_command(self, mp, mv, tp, tv, dt):
        R_vec = tp - mp
        R = float(xp.linalg.norm(R_vec))
        R = max(R, 1e-6)
        u_LOS = R_vec / R
        LOS_angle = math.atan2(float(R_vec[1]), float(R_vec[0]))
        if self.last_LOS_angle is None: LOS_rate = 0.0
        else:
            adiff = LOS_angle - self.last_LOS_angle
            adiff = (adiff + math.pi) % (2*math.pi) - math.pi
            LOS_rate = adiff / dt
        self.last_LOS_angle = LOS_angle
        rel_vel = tv - mv
        range_rate = float(xp.dot(rel_vel, u_LOS))
        closing_speed = -range_rate
        if closing_speed < 0: closing_speed = 0
        a_command = self.N * closing_speed * LOS_rate
        return a_command

class PL15Env(gym.Env if gym else object):
    metadata = {'render.modes': []}
    def __init__(self, config=None):
        cfg = config if config else {}
        self.dt = cfg.get('dt', 0.1)
        self.real_time = cfg.get('real_time', False)
        self.use_internal_guidance = cfg.get('use_internal_guidance', False)
        self.max_time = cfg.get('max_time', 60.0)
        if self.real_time and HAS_RT:
            self.sim_env = rt.RealtimeEnvironment(factor=1.0)
        else:
            self.sim_env = simpy.Environment()
        mspeed = cfg.get('missile_speed', 800.0)
        tspeed = cfg.get('target_speed', None)
        if tspeed is None: tspeed = random.uniform(200.0, 300.0)
        rcs_m = cfg.get('target_rcs_mean', 5.0)
        dist = random.uniform(10000.0, 20000.0)
        bearing = random.uniform(-math.pi/3, math.pi/3)
        mpos = xp.array([0.0, 0.0])
        mvel = xp.array([mspeed, 0.0])
        tpos = dist * xp.array([math.cos(bearing), math.sin(bearing)])
        thead = random.uniform(0, 2*math.pi)
        tvel = tspeed * xp.array([math.cos(thead), math.sin(thead)])
        self.missile = Missile(mpos, mvel)
        self.target = Target(tpos, tvel, rcs_mean=rcs_m, rcs_model="Swerling1")
        cpow = cfg.get('clutter_power', 0.0)
        self.rf_env = RFEnvironment(clutter_power=cpow)
        ecmp = cfg.get('ecm_power', 0.0)
        self.ecm = ECM(eirp=ecmp) if ecmp and ecmp>0 else None
        rfreq = 10e9
        rpower = 5000.0
        self.radar = RadarSeeker(freq_hz=rfreq, power_w=rpower, gain_tx=1.0, gain_rx=1.0, 
                                 noise_fig=1.0, bandwidth_hz=1e6, min_detect_snr=10.0,
                                 rf_env=self.rf_env, ecm=self.ecm)
        self.signal_processor = SignalProcessing(detection_threshold=self.radar.min_detect_snr)
        self.guidance = Guidance(N=4)
        self.sensor_fusion = SensorFusion()
        self.sensor_fusion.initialize(self.target.position, self.target.velocity)
        self.time = 0.0
        self.done = False
        if gym:
            high_val = xp.finfo(xp.float32).max
            self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(4,), dtype=xp.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=xp.float32)
        else:
            self.observation_space = None
            self.action_space = None
        self.missile_history = []
        self.target_history = []

    def step(self, action):
        if self.done: return self._get_observation(), 0.0, True, {}
        ev = self.sim_env.timeout(self.dt)
        self.sim_env.run(ev)
        self.time += self.dt
        self.target.update(self.dt)
        sig = self.target.get_rcs()
        radar_sig = self.radar.sense(self.missile.position, self.missile.velocity, self.target.position, sig)
        detection = self.signal_processor.process(radar_sig)
        self.sensor_fusion.update(detection, self.missile.position, self.dt)
        if self.sensor_fusion.target_pos_est is not None:
            tp_est = self.sensor_fusion.target_pos_est
            tv_est = self.sensor_fusion.target_vel_est if self.sensor_fusion.target_vel_est is not None else xp.array([0.0, 0.0])
        else:
            tp_est = self.target.position
            tv_est = self.target.velocity
        if self.use_internal_guidance:
            cmd = self.guidance.compute_command(self.missile.position, self.missile.velocity, tp_est, tv_est, self.dt)
        else:
            a_norm = float(action[0]) if isinstance(action, (list, xp.ndarray)) else float(action)
            a_norm = max(-1.0, min(1.0, a_norm))
            cmd = a_norm * self.missile.max_turn_acc
        self.missile.update(cmd, self.dt)
        rel_vec = self.target.position - self.missile.position
        miss_dist = float(xp.linalg.norm(rel_vec))
        done = False
        reward = 0.0
        if miss_dist < 50.0:
            done = True
            reward = 100.0
        if self.time >= self.max_time:
            done = True
        if done and reward == 0.0:
            reward = -100.0
        self.done = done
        obs = self._get_observation()
        info = {'miss_distance': miss_dist, 'target_detected': detection is not None}
        self.missile_history.append((float(self.missile.position[0]), float(self.missile.position[1])))
        self.target_history.append((float(self.target.position[0]), float(self.target.position[1])))
        return obs, reward, done, info

    def _get_observation(self):
        if self.sensor_fusion.target_pos_est is not None:
            rp = self.sensor_fusion.target_pos_est - self.missile.position
            rv = (self.sensor_fusion.target_vel_est - self.missile.velocity
                  if self.sensor_fusion.target_vel_est is not None else xp.array([0.0,0.0]))
        else:
            rp = self.target.position - self.missile.position
            rv = self.target.velocity - self.missile.velocity
        obs = xp.array([rp[0], rp[1], rv[0], rv[1]], dtype=xp.float32)
        return obs

    def reset(self):
        new_env = PL15Env()
        self.__dict__.update(new_env.__dict__)
        return self._get_observation()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def plot_trajectory(self):
        if not HAS_MPL:
            print("matplotlib not available.")
            return
        mx = [p[0] for p in self.missile_history]
        my = [p[1] for p in self.missile_history]
        tx = [p[0] for p in self.target_history]
        ty = [p[1] for p in self.target_history]
        plt.figure()
        plt.plot(mx, my, 'r-', label="Missile")
        plt.plot(tx, ty, 'b-', label="Target")
        plt.scatter([mx[0]], [my[0]], c='r', marker='o', label="Missile Start")
        plt.scatter([tx[0]], [ty[0]], c='b', marker='o', label="Target Start")
        plt.legend()
        plt.title("PL-15 Engagement Trajectory")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis('equal')
        plt.grid(True)
        plt.show()

"""
Usage Example (Pseudocode):

env = PL15Env(config={'dt':0.05, 'real_time':False, 'ecm_power':2000.0, 'clutter_power':10.0})
obs = env.reset()
done = False
while not done:
    action = [0.0]  # or from your RL policy
    obs, reward, done, info = env.step(action)
env.plot_trajectory()

If CuPy 3.12+ is installed, xp uses GPU arrays. Otherwise falls back to NumPy.
"""
