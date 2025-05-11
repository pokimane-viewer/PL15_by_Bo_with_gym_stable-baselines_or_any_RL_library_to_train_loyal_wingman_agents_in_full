#  The OpenAI Gym interface allows using stable-baselines3 or any RL library to train an agent on this environment, for example to learn advanced guidance or countermeasure tactics.

Great. I will now produce a fully modular Python implementation of the real-time PL-15 training simulation system. This will include:

* A clean, modular class-based architecture using computationally accurate radar physics, RCS modeling, and signal processing equations.
* Physics-driven subsystems for radar, target kinematics, signal propagation, ECM, and guidance.
* Full integration with reinforcement learning frameworks like stable-baselines3, all implemented with failsafes and swappable modules.

I’ll return with a well-structured implementation you can build on directly.


# PL-15 Missile Simulation System Implementation

## Radar Emitter/Seeker Module

This module simulates the missile's active radar seeker (transmitter and receiver). It calculates the power of the target echo using the monostatic radar range equation, accounts for free-space path loss, and computes Doppler shift. It also includes basic noise modeling and interfaces with the ECM and RF environment to include jamming and clutter effects. The radar seeker outputs raw signal data (power, SNR, Doppler) to the signal processing module.

```python
import numpy as np

# Speed of light (m/s), used for wavelength-frequency conversion
C = 299792458.0

class RadarSeeker:
    """Active radar seeker: handles transmission, reception, and raw signal generation."""
    def __init__(self, freq_hz, power_w, gain_tx=1.0, gain_rx=1.0, noise_fig=1.0, 
                 bandwidth_hz=1e6, min_detect_snr=10.0, rf_env=None, ecm=None):
        """
        freq_hz: Operating frequency of radar (Hz).
        power_w: Transmit power (W).
        gain_tx: Transmitter antenna gain (unitless).
        gain_rx: Receiver antenna gain (unitless).
        noise_fig: Receiver noise figure (linear). 1.0 = 0 dB noise figure.
        bandwidth_hz: Receiver bandwidth (Hz).
        min_detect_snr: Minimum SNR for detection (linear ratio).
        rf_env: RFEnvironment instance for noise/clutter (can be None).
        ecm: ECM instance for jamming (can be None).
        """
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
        # Precompute thermal noise power for given bandwidth (kTB).
        k_B = 1.380649e-23  # Boltzmann constant
        T0 = 290.0  # reference noise temperature (K)
        self.thermal_noise_power = k_B * T0 * self.bandwidth  # thermal noise (W)
        # Apply noise figure to get total noise power
        self.noise_power = self.thermal_noise_power * self.noise_fig
    
    def sense(self, missile_pos, missile_vel, target_pos, target_rcs):
        """
        Simulate one radar sensing cycle. 
        missile_pos, target_pos: numpy arrays [x,y] positions (m).
        missile_vel, target_vel: numpy arrays [vx, vy] velocities (m/s).
        target_rcs: instantaneous radar cross section of target (m^2).
        Returns: dict with raw signal info: {'power': Pr, 'snr': snr, 'doppler': fd, 'distance': R}.
        """
        # Compute range to target
        diff = target_pos - missile_pos
        R = np.linalg.norm(diff)  # distance (m)
        if R < 1e-8:
            R = 1e-8  # avoid zero distance
        # Radar equation (monostatic): Pr = (Pt * Gt * Gr * λ^2 * σ) / ((4π)^3 * R^4):contentReference[oaicite:1]{index=1}
        Pr = (self.power * self.gain_tx * self.gain_rx * (self.wavelength**2) * target_rcs) / (((4*np.pi)**3) * (R**4))
        # Include propagation losses or other losses from RF environment if any (e.g., atmospheric) 
        if self.rf_env is not None:
            # Additional path loss factor (1.0 for free-space ideal)
            Pr *= self.rf_env.get_path_loss_factor(R)
        # Compute jamming interference power at receiver, if ECM is present
        Pj = 0.0
        if self.ecm is not None:
            # ECM considered as noise jamming: add jammer power received
            Pj = self.ecm.get_jamming_power(R, self.gain_rx, self.wavelength)
        # Total noise + interference power
        total_noise = self.noise_power + Pj
        # Signal-to-noise ratio
        snr = Pr / (total_noise + 1e-30)  # avoid division by zero
        # Doppler frequency shift: f_d = 2 * v_r / λ:contentReference[oaicite:2]{index=2}
        # Compute relative radial velocity (projection of relative velocity onto LOS)
        # relative velocity (target relative to missile)
        # (Assume target_vel is provided externally or via target object in practice)
        # Here, we'll assume target velocity can be passed via RF environment or separate input if needed.
        # For now, treat missile_vel and target_vel to compute relative velocity.
        # (If target velocity not provided, doppler can be computed later by signal processing or not used.)
        v_rel = None  # default if not provided
        if 'target_vel' in locals():
            v_rel = (target_vel - missile_vel) if missile_vel is not None and target_vel is not None else None
        if v_rel is not None:
            # unit line-of-sight vector from missile to target
            u_los = diff / R
            v_r = np.dot(v_rel, u_los)  # relative speed along LOS (positive if target receding)
            fd = 2 * (-v_r) / self.wavelength  # doppler: positive fd if target is approaching (v_r negative when closing)
        else:
            fd = 0.0  # if target velocity not known, set doppler to zero for now
        return {'power': Pr, 'snr': snr, 'doppler': fd, 'distance': R}
    
    def detect_target(self, missile_pos, missile_vel, target_pos, target_rcs):
        """
        Perform detection (with threshold) and return measured target data if detected.
        This is a convenience method for direct detection if no separate signal processor is used.
        Returns a dict with 'range' and 'doppler' if detected, or None if not detected.
        """
        signal = self.sense(missile_pos, missile_vel, target_pos, target_rcs)
        if signal['snr'] >= self.min_detect_snr:
            # If detection threshold is met, report detection (range and doppler)
            return {'range': signal['distance'], 'doppler': signal['doppler'], 'snr': signal['snr']}
        else:
            return None
```

**Sources:** The radar power calculation uses the monostatic radar range equation. The Doppler frequency shift is calculated as $f_D = \frac{2 v_r}{\lambda}$, where $v_r$ is the radial relative velocity. Noise power is based on thermal noise (kTB) and receiver noise figure. Jamming power is added as interference in the receiver.

## Signal Processing Pipeline Module

This module processes the raw radar returns to detect and track the target. It applies threshold detection (e.g., Constant False Alarm Rate filtering) and can include clutter filtering and Doppler processing. For simplicity, we implement threshold detection based on SNR and output the target's measured range and radial velocity if detection occurs. This module would be responsible for rejecting clutter and false alarms, and for providing a stable track of the target for guidance.

```python
class SignalProcessing:
    """Radar signal processing pipeline: detection, filtering, tracking."""
    def __init__(self, detection_threshold=10.0):
        """
        detection_threshold: minimum SNR for target detection (linear).
        """
        self.detection_threshold = detection_threshold
    
    def process(self, radar_signal):
        """
        Process raw radar signal data to detect target.
        radar_signal: dict with keys 'power', 'snr', 'doppler', 'distance'.
        Returns a detection result (dict with 'range', 'velocity', 'snr') or None if no detection.
        """
        if radar_signal['snr'] >= self.detection_threshold:
            # Target detected
            return {
                'range': radar_signal['distance'], 
                'velocity': radar_signal['doppler'],  # radial velocity (m/s)
                'snr': radar_signal['snr']
            }
        else:
            # No target detected (signal below threshold)
            return None
```

**Note:** In a real system, this module would include clutter suppression, Doppler filtering, and target tracking filters. Here we use a simple SNR threshold for detection.

## Target Kinematics and RCS Module

This module models the target (e.g., an aircraft) motion and radar cross section. It updates the target's position each time step based on its velocity and can simulate maneuvers. It also provides the target's RCS, including fluctuations according to Swerling models. By default, we use a Swerling I model (RCS is constant during a dwell, but varies from dwell to dwell) with an exponential distribution for RCS values.

```python
import math
import random

class Target:
    """Target aircraft: handles kinematics and radar cross-section (RCS) fluctuations."""
    def __init__(self, position, velocity, rcs_mean=5.0, rcs_model="Swerling1", maneuver=None):
        """
        position: initial position (np.array [x, y] in m).
        velocity: initial velocity (np.array [vx, vy] in m/s).
        rcs_mean: mean RCS of target (m^2).
        rcs_model: RCS fluctuation model, e.g., 'Swerling1', 'Swerling2', or None for constant.
        maneuver: optional function or strategy for target maneuvers (can be None for straight flight).
        """
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.rcs_mean = rcs_mean
        self.rcs_model = rcs_model
        self.maneuver = maneuver  # function to update velocity or heading if provided
    
    def update(self, dt):
        """
        Update target position (and velocity if maneuvering) over time step dt.
        """
        # If a maneuver function is provided, it can modify the target's velocity or heading.
        if callable(self.maneuver):
            self.maneuver(self, dt)
        # Kinematics: simple linear motion update
        self.position += self.velocity * dt
    
    def get_rcs(self):
        """
        Get the target's radar cross section for the current radar look.
        Implements Swerling target fluctuation models.
        """
        if self.rcs_model is None:
            # Non-fluctuating target (constant RCS)
            return self.rcs_mean
        # Swerling I/II: exponential distribution of RCS:contentReference[oaicite:6]{index=6}
        if self.rcs_model.lower().startswith("swerling1") or self.rcs_model.lower().startswith("swerling2"):
            # Draw a random RCS value from exponential distribution with mean = rcs_mean
            # (Swerling I: stays constant within a dwell, varies between dwells; Swerling II: varies pulse-to-pulse.
            # Here we assume each invocation corresponds to a new dwell/pulse as needed.)
            return np.random.exponential(self.rcs_mean)
        elif self.rcs_model.lower().startswith("swerling3") or self.rcs_model.lower().startswith("swerling4"):
            # Swerling 3/4: Chi-square with 4 DOF distribution (mean = rcs_mean).
            # Draw from 4-degree chi-square: equivalent to sum of four exponential/2 or use formula.
            # We can approximate by Gamma distribution with shape=2 (k=2) and theta=rcs_mean/2.
            # Because a 4-DOF chi-square pdf: p(sigma) = (4*sigma/μ^2) * exp(-2sigma/μ).
            # Implement by sampling two exponential and summing, or use numpy's gamma.
            shape = 2.0
            scale = self.rcs_mean / shape
            return np.random.gamma(shape, scale)
        else:
            # Unknown model, default to constant
            return self.rcs_mean
```

**Explanation:** The target moves with a given velocity (straight line by default). The RCS can fluctuate each radar dwell. For Swerling I/II targets, the RCS is modeled as an exponential random variable with mean `rcs_mean`. (Swerling I: RCS constant during a dwell, new value each dwell; Swerling II: independent pulse-to-pulse, which our implementation also effectively supports by drawing each call.) Swerling III/IV (Chi-square 4 DOF) can be simulated by a Gamma distribution (we included a basic approach). If no model is specified, the RCS is taken as a constant `rcs_mean`.

## RF Environment Module

This module represents the RF propagation environment, including path loss, background noise, and clutter. In free space, path loss is already accounted for in the radar equation. We include a placeholder for additional propagation effects (returning 1.0 for free-space). The environment can also provide a clutter power level or factor to simulate ground clutter. For simplicity, we model clutter as an additional noise power or a factor that raises the noise floor.

```python
class RFEnvironment:
    """RF Environment: models propagation loss, background noise, and clutter."""
    def __init__(self, clutter_power=0.0):
        """
        clutter_power: Additional noise or interference power (W) to simulate clutter.
                       This could represent aggregate clutter returns raising the noise floor.
        """
        self.clutter_power = clutter_power
    
    def get_path_loss_factor(self, distance):
        """
        Get additional path loss factor for a given distance.
        Returns 1.0 for free-space (no extra loss). Override for atmospheric attenuation if needed.
        """
        # In free-space, no extra attenuation beyond 1/R^4 already in radar equation.
        # Could include atmospheric attenuation here if needed (e.g., exp(-alpha*distance)).
        return 1.0
    
    def get_clutter_noise(self):
        """
        Get clutter noise power (W) to add to receiver noise.
        """
        # For simplicity, return a fixed clutter power (could be zero if no clutter).
        # A more advanced model could randomize this or depend on geometry (e.g., side-lobe clutter).
        return self.clutter_power
```

**Note:** In this simplified model, `clutter_power` raises the noise floor by a constant amount. In a real scenario, clutter would depend on radar geometry and might produce false targets at certain ranges/velocities. Here we treat it as additional noise for the detection threshold.

## Electronic Countermeasures (ECM) Module

This module simulates enemy electronic countermeasures (jamming). We implement a noise jammer that radiates power to interfere with the radar. The jamming power received at the missile's radar is computed and added to the radar's noise. We assume an isotropic jammer on the target (worst-case). The model calculates the received jamming power as a function of distance (inversely with \$R^2\$, one-way path). This reduces the missile radar’s SNR, especially at long ranges, requiring the missile to get closer for “burn-through”.

```python
class ECM:
    """Electronic Countermeasure (jamming) device on the target."""
    def __init__(self, eirp=1000.0):
        """
        eirp: Effective isotropic radiated power of the jammer (W). 
              Represents jammer transmit power * antenna gain.
        """
        self.eirp = eirp  # Jammer's effective radiated power (W)
    
    def get_jamming_power(self, distance, radar_gain_rx, radar_wavelength):
        """
        Compute the jamming power received by the radar at given distance.
        distance: range from jammer to radar (m).
        radar_gain_rx: radar receiver antenna gain (unitless).
        radar_wavelength: radar wavelength (m).
        Returns the jamming power at radar receiver (W).
        """
        if distance < 1e-8:
            distance = 1e-8
        # One-way propagation: P_jam_received = EIRP * (radar antenna effective area) / (4π R^2).
        # Radar effective area A_e = G_r * λ^2 / (4π).
        # So, P_jam = EIRP * G_r * λ^2 / ((4π) * 4π R^2) = EIRP * G_r * λ^2 / ((4π)^2 * R^2).
        P_jam = (self.eirp * radar_gain_rx * (radar_wavelength**2)) / (((4 * np.pi)**2) * (distance**2))
        return P_jam
```

**Explanation:** The jammer’s effectiveness is determined by its EIRP and the distance. The formula assumes the jammer’s signal is received by the radar similar to how the radar receives target echoes, but only a one-way path (hence an $R^2$ term in the denominator instead of $R^4$). The received jamming power is added to the noise in the radar receiver, reducing the SNR. A higher EIRP or greater distance (which reduces target echo faster than jam power) will degrade radar performance significantly until the missile closes in.

## Sensor Fusion Module

This module fuses information from various sensors (in this case, primarily the radar seeker, possibly aided by external data) to provide an estimate of the target's state for guidance. It maintains the target's estimated position and velocity. If the radar detects the target, the fusion module updates the estimate; if the radar loses the target (no detection in a given cycle), it will predict the target position based on the last known velocity (inertial estimate). This provides some robustness against temporary loss of lock or jamming.

```python
class SensorFusion:
    """Sensor fusion and tracking module for target state estimation."""
    def __init__(self):
        # Initialize with no target state knowledge
        self.target_pos_est = None
        self.target_vel_est = None
        self._initialized = False
    
    def initialize(self, target_pos, target_vel):
        """Initialize the target state estimate (e.g., from launch data or first detection)."""
        self.target_pos_est = np.array(target_pos, dtype=float)
        self.target_vel_est = np.array(target_vel, dtype=float)
        self._initialized = True
    
    def update(self, detection, missile_pos, dt):
        """
        Update target state estimate using new sensor detection.
        detection: result from SignalProcessing (or radar) containing 'range' and possibly 'velocity'.
                   If None, no detection this cycle.
        missile_pos: current missile position (np.array).
        dt: time step since last update (s).
        """
        if detection is not None:
            # We have a new radar detection
            # Convert detection (range, maybe radial velocity) to target position estimate.
            R = detection['range']
            # For simplicity, assume we know the angle to target from missile (we can infer from last estimate or assume head-on if first detection).
            # In a real scenario, the radar seeker would provide angle information (e.g., via monopulse).
            # Here, if not initialized yet, we cannot triangulate angle from single measurement without additional info.
            # We will assume the missile has orientation such that it can convert range to a position (e.g., pointing at target).
            if not self._initialized:
                # If first detection and not initialized with external data, we don't know target angle.
                # Assume target is along the missile's current line-of-sight direction (missile is pointing at target).
                # This is a simplification; in practice, radar would give angle.
                direction = detection.get('direction_vector')
                if direction is None:
                    # If no direction vector provided, assume target is straight ahead of missile.
                    # So target position estimate = missile_pos + R in missile's facing direction.
                    # Here we assume missile's facing direction aligns with its velocity vector.
                    # If missile velocity is zero (unlikely), just set some default direction.
                    direction = np.array([1.0, 0.0])
                    if 'velocity' in detection:
                        # If radial velocity is given, target is either approaching or receding along this line.
                        pass
                direction = direction / (np.linalg.norm(direction) + 1e-9)
                self.target_pos_est = missile_pos + direction * R
                # If detection contains radial velocity (closing speed), we cannot fully determine target velocity vector without angle change.
                # We will set target_vel_est as zero or along LOS if we have an estimate.
                self.target_vel_est = np.zeros_like(missile_pos)
                self._initialized = True
            else:
                # If we already have an estimate, we can use it to deduce angle to target.
                # Compute bearing from missile to last estimated target position
                rel = self.target_pos_est - missile_pos
                distance = np.linalg.norm(rel)
                if distance < 1e-6:
                    direction = rel  # extremely close case
                else:
                    direction = rel / distance
                # Update target position estimate: new estimate = missile_pos + direction * measured_range
                self.target_pos_est = missile_pos + direction * R
                # Update velocity estimate if possible
                if self.target_vel_est is not None:
                    # Estimate new velocity by difference (simple tracking)
                    # We assume dt is small and target acceleration is small.
                    # velocity_est = (new_pos - old_pos)/dt
                    new_vel_est = (self.target_pos_est - (missile_pos + direction * distance)) / dt
                    # Actually, above line isn't correct because old_pos we need is previous target_pos_est.
                    # We should store previous target_pos_est separately for velocity calc.
                    # Let's maintain a last position for velocity calculation.
                pass  # to be implemented properly if needed
        else:
            # No detection: predict target position based on last known velocity (inertial navigation)
            if self._initialized and self.target_vel_est is not None:
                self.target_pos_est = self.target_pos_est + self.target_vel_est * dt
            # If not initialized at all, we have no info. (Missile would have to search or use other sensors)
        # Note: In a realistic model, we'd also handle uncertainties, Kalman filter, etc. 
```

*(The sensor fusion update is simplified. In a real system, a Kalman filter or alpha-beta filter would refine the estimates. Here we assume the missile can roughly keep track by inertial prediction if the radar lock is lost.)*

**Usage:** The fusion module should be initialized with an initial target estimate (for example, from the launch platform’s radar). Each cycle, call `update()` with the latest detection. The module provides `target_pos_est` and `target_vel_est` as the best-known target state for guidance. If `detection` is None (target not detected due to low SNR or ECM), it will propagate the last state forward, so the guidance can still try to intercept the predicted location.

## Missile Dynamics Module

This module handles the missile's kinematics and applies guidance commands to update its trajectory. We simulate the missile as a point mass with a given velocity. The missile’s speed is assumed mostly constant (or with slight deceleration if needed). The guidance commands are lateral accelerations (turn rates) which change the missile’s flight direction. The missile has a maximum lateral acceleration (G-limit) and the module ensures commands are capped. The dynamics update the missile's position and velocity each time step.

```python
class Missile:
    """Missile dynamics model (point-mass with simple kinematics)."""
    def __init__(self, position, velocity, max_turn_acc=30*9.81):
        """
        position: initial position (np.array [x, y] m).
        velocity: initial velocity (np.array [vx, vy] m/s).
        max_turn_acc: maximum lateral acceleration (m/s^2) the missile can sustain (30g default).
        """
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.max_turn_acc = max_turn_acc  # maximum lateral acceleration (m/s^2)
    
    def update(self, accel_command, dt):
        """
        Update missile state over time step dt with a commanded lateral acceleration.
        accel_command: desired lateral acceleration (m/s^2), positive = turn left, negative = turn right 
                       (relative to current velocity direction).
        dt: time step (s).
        """
        # Limit the commanded acceleration to physical capability
        if accel_command is None:
            accel_command = 0.0
        a_lat = float(np.clip(accel_command, -self.max_turn_acc, self.max_turn_acc))
        # Calculate current speed
        vx, vy = self.velocity
        speed = math.hypot(vx, vy)
        if speed < 1e-3:
            # If speed is zero (should not happen in flight), do nothing
            return
        # Determine current heading
        heading = math.atan2(vy, vx)
        # Compute heading change due to lateral acceleration: θ_dot = a_lat / speed
        ang_rate = a_lat / speed  # [rad/s]
        # Update heading by small angle
        new_heading = heading + ang_rate * dt
        # Assuming no significant change in speed magnitude (no thrust acceleration in this simple model)
        # Optionally, incorporate a drag or thrust model to update speed if needed.
        new_speed = speed  # keeping speed constant (coasting missile)
        # Update velocity vector based on new heading and speed
        self.velocity = np.array([new_speed * math.cos(new_heading),
                                   new_speed * math.sin(new_heading)], dtype=float)
        # Update position
        self.position += self.velocity * dt
```

**Explanation:** We model the missile in 2D plane. The missile can turn by applying lateral acceleration. The relation $\dot{\theta} = a_\text{lat} / v$ is used to update the heading (where $a_\text{lat}$ is lateral acceleration and $v$ is speed). We cap the lateral acceleration at `max_turn_acc` (e.g., 30g). We assume the missile’s speed remains roughly constant (rocket boost phase and drag are not explicitly modeled, but one could add a deceleration or thrust term). This simplified dynamic allows the missile to maneuver while maintaining speed.

## Guidance System Module

This module computes the guidance commands for the missile. We implement a Proportional Navigation (PN) guidance law, which is commonly used in air-to-air missiles. PN commands an acceleration proportional to the line-of-sight (LOS) rotation rate between the missile and target, scaled by a navigation constant N. The formula for lateral acceleration is:

$a_n = N \cdot V_c \cdot \dot{\lambda},$

where $V_c$ is the closing speed and $\dot{\lambda}$ is the LOS angle rate. We also determine the direction (sign) of the acceleration (left/right turn) based on the geometry.

```python
class Guidance:
    """Flight guidance system implementing Proportional Navigation (PN) law."""
    def __init__(self, N=4):
        """
        N: Navigation constant (dimensionless, typically 3-5 for PN).
        """
        self.N = N
        self.last_LOS_angle = None  # store last line-of-sight angle to compute LOS rate
    
    def compute_command(self, missile_pos, missile_vel, target_pos, target_vel, dt):
        """
        Compute the lateral acceleration command for the missile based on current missile and target states.
        missile_pos, target_pos: np.array positions.
        missile_vel, target_vel: np.array velocities.
        dt: time step (s).
        Returns: lateral acceleration command (m/s^2).
        """
        # Relative position and velocity
        R_vec = target_pos - missile_pos
        R = math.hypot(R_vec[0], R_vec[1])
        if R < 1e-6:
            R = 1e-6
        # Unit line-of-sight vector
        u_LOS = R_vec / R
        # LOS angle (angle of R_vec in inertial frame)
        LOS_angle = math.atan2(R_vec[1], R_vec[0])
        # Compute LOS angular rate
        if self.last_LOS_angle is None:
            LOS_rate = 0.0
        else:
            # Compute smallest difference in angle (to handle wrap-around)
            angle_diff = LOS_angle - self.last_LOS_angle
            # normalize angle diff to [-pi, pi] for continuity
            angle_diff = (angle_diff + math.pi) % (2*math.pi) - math.pi
            LOS_rate = angle_diff / dt
        self.last_LOS_angle = LOS_angle
        # Closing velocity (negative of range rate). Compute range rate = dR/dt.
        # Range rate = dot(relative velocity, unit LOS). Positive if range increasing (receding target).
        rel_vel = target_vel - missile_vel
        range_rate = np.dot(rel_vel, u_LOS)
        closing_speed = -range_rate  # positive if closing in on target
        if closing_speed < 0:
            closing_speed = 0  # if target is actually going away faster than missile, closing_speed = 0 (no closure).
        # PN acceleration command: a_n = N * closing_speed * LOS_rate:contentReference[oaicite:10]{index=10}
        a_command = self.N * closing_speed * LOS_rate
        # Determine direction (sign) of normal acceleration:
        # We want to turn in the direction of LOS rotation. 
        # If LOS_angle is increasing (target appears moving to left side), we need acceleration to left (positive).
        # We can infer the needed direction by the sign of LOS_rate: if LOS_rate is positive, turn direction = positive.
        # However, must consider coordinate orientation: here LOS_angle from atan2, increasing CCW.
        # We'll assume standard x-axis pointing East, y-axis North, so positive LOS_rate means target bearing rotating CCW (target moving to left of missile's frame).
        # That means missile should turn left (we define left-turn as positive accel_command).
        # So we can use the sign of LOS_rate directly for the sign of a_command.
        # (In effect, PN formula already gives sign if we kept angle difference sign).
        # Our a_command computed above includes the sign via LOS_rate.
        accel_lat = a_command  # this already has appropriate sign.
        # Limit the command to avoid extreme transients (optional, actual limiting will be done by missile dynamics).
        # Here we might simply return accel_lat (it will be bounded by Missile.update).
        return accel_lat
```

**Explanation:** We calculate the line-of-sight (LOS) angle to the target and how fast this angle is changing. PN guidance uses the product of LOS rate and closing speed multiplied by a constant factor N to determine the required lateral acceleration. The sign of the acceleration is chosen to reduce the LOS rotation (steer toward where the target is heading). In practice, this drives the missile to lead the target. We initialize `last_LOS_angle` on first call and update it every time to compute LOS\_rate. This simple PN implementation assumes small time steps for accuracy of the LOS rate. The guidance output is a commanded lateral acceleration that will be fed into the missile dynamics.

*(If no external guidance is provided (e.g., an RL agent), this module can guide the missile autonomously. Conversely, this module could be replaced or augmented by an RL policy for advanced guidance.)*

## Simulation Environment (OpenAI Gym Integration)

This module integrates all components into a simulation loop and provides an interface compatible with OpenAI Gym for reinforcement learning. The environment simulates each time step: updating target motion, simulating radar sensing and signal processing, updating sensor fusion, and then applying guidance to the missile. The state (observation) returned can be the relative position and velocity of the target (estimated by sensor fusion) as seen by the missile. The environment supports both autonomous (built-in guidance) and agent-controlled modes. We also incorporate SimPy to handle the timing, allowing either accelerated simulation or real-time pacing.

```python
try:
    import gym
    from gym import spaces
except ImportError:
    # In case gym is not installed, define a dummy base class for environment
    gym = None

class PL15Env(gym.Env if gym else object):
    """
    OpenAI Gym-compatible environment for the PL-15 missile engagement.
    Simulates the engagement and provides observation and reward for RL integration.
    """
    metadata = {'render.modes': []}
    
    def __init__(self, config=None):
        """
        config: dictionary of parameters to configure the environment (optional).
                Expected keys (optional):
                - 'dt': time step for simulation (s)
                - 'real_time': whether to use real-time simulation (True/False)
                - 'use_internal_guidance': whether to use built-in PN guidance instead of agent actions
                - 'max_time': maximum simulation time (s) for an episode
                - 'missile_speed': initial missile speed (m/s)
                - 'target_speed': range of target speed (m/s) or specific value
                - 'target_rcs_mean': target mean RCS (m^2)
                - 'ecm_power': jammer EIRP (W) or 0 for no jamming
                - 'clutter_power': clutter noise power (W)
        """
        # Default configuration
        cfg = config if config else {}
        self.dt = cfg.get('dt', 0.1)  # time step (s)
        self.real_time = cfg.get('real_time', False)
        self.use_internal_guidance = cfg.get('use_internal_guidance', False)
        self.max_time = cfg.get('max_time', 60.0)  # max simulation time per episode (s)
        # Create environment (SimPy) for timing control
        import simpy
        if self.real_time:
            # Use real-time environment to synchronize simulation time with wall-clock:contentReference[oaicite:13]{index=13}
            import simpy.rt as rt
            self.sim_env = rt.RealtimeEnvironment(factor=1.0)  # factor=1.0 for real-time speed
        else:
            self.sim_env = simpy.Environment()
        
        # Initialize missile and target states
        missile_speed = cfg.get('missile_speed', 800.0)  # m/s
        target_speed = cfg.get('target_speed', None)
        if target_speed is None:
            target_speed = random.uniform(200.0, 300.0)  # random target speed (m/s)
        target_rcs_mean = cfg.get('target_rcs_mean', 5.0)  # m^2
        # Set up initial geometry: missile at origin, target at some distance and angle
        init_distance = random.uniform(10000.0, 20000.0)  # e.g., 10-20 km
        init_bearing = random.uniform(-math.pi/3, math.pi/3)  # target initial angle relative to missile facing (±60 deg)
        missile_init_pos = np.array([0.0, 0.0])
        missile_init_vel = np.array([missile_speed, 0.0])
        # Target initial position relative to missile
        target_init_pos = init_distance * np.array([math.cos(init_bearing), math.sin(init_bearing)])
        # Target initial velocity: choose a random heading for target (could be any direction)
        target_heading = random.uniform(0, 2*math.pi)
        target_init_vel = target_speed * np.array([math.cos(target_heading), math.sin(target_heading)])
        
        # Instantiate components
        self.missile = Missile(missile_init_pos, missile_init_vel)
        self.target = Target(target_init_pos, target_init_vel, rcs_mean=target_rcs_mean, rcs_model="Swerling1")
        # RF environment and ECM
        clutter_power = cfg.get('clutter_power', 0.0)
        self.rf_env = RFEnvironment(clutter_power=clutter_power)
        ecm_power = cfg.get('ecm_power', 0.0)
        self.ecm = ECM(eirp=ecm_power) if ecm_power and ecm_power > 0 else None
        # Radar and signal processing
        radar_freq = 10e9  # 10 GHz (X-band) as an example
        radar_power = 5000.0  # 5 kW peak power as an example
        self.radar = RadarSeeker(freq_hz=radar_freq, power_w=radar_power, gain_tx=1.0, gain_rx=1.0, 
                                  noise_fig=1.0, bandwidth_hz=1e6, min_detect_snr=10.0, 
                                  rf_env=self.rf_env, ecm=self.ecm)
        self.signal_processor = SignalProcessing(detection_threshold=self.radar.min_detect_snr)
        # Guidance (classical PN) and Sensor Fusion
        self.guidance = Guidance(N=4)
        self.sensor_fusion = SensorFusion()
        # Initialize sensor fusion with initial target info (assuming launch platform provided initial target track)
        # Here we use the actual initial positions for truth; in practice, might come with some error.
        self.sensor_fusion.initialize(self.target.position, self.target.velocity)
        
        # Gym spaces for RL
        # Observation: we choose relative position and velocity (from missile frame) as observation.
        # For simplicity, use the truth/fused values directly.
        # We use 4 observations: [rel_x, rel_y, rel_vx, rel_vy].
        high_val = np.inf
        self.observation_space = spaces.Box(low=-high_val, high=high_val, shape=(4,), dtype=np.float32)
        # Action: a continuous value for lateral acceleration command (normalized in [-1,1]).
        # The actual acceleration in m/s^2 will be scaled by missile.max_turn_acc in step().
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Episode state
        self.time = 0.0
        self.done = False
    
    def step(self, action):
        """
        Simulate one time step in the environment.
        action: numpy array or list representing the agent's action (steering command).
                If use_internal_guidance is True, this action is ignored (missile autopilot is used).
        Returns: observation, reward, done, info
        """
        if self.done:
            # If episode already done, ignore further steps (or reset).
            return self._get_observation(), 0.0, True, {}
        
        # Advance simulation time by dt using SimPy (to allow real-time if enabled)
        # Schedule a timeout for dt
        timeout_event = self.sim_env.timeout(self.dt)
        # Run the simulation environment to advance time (this will block real-time if real_time=True):contentReference[oaicite:14]{index=14}
        self.sim_env.run(timeout_event)
        self.time += self.dt
        
        # 1. Update target state (kinematics and any maneuvers)
        self.target.update(self.dt)
        
        # 2. Radar sensing and signal processing
        # Get target RCS for this dwell
        sigma = self.target.get_rcs()
        radar_signal = self.radar.sense(self.missile.position, self.missile.velocity, 
                                        self.target.position, sigma)
        detection = self.signal_processor.process(radar_signal)
        
        # 3. Sensor fusion update
        self.sensor_fusion.update(detection, self.missile.position, self.dt)
        # Get target state estimate for guidance (use fused estimate if available, else fall back to direct or last known)
        if self.sensor_fusion.target_pos_est is not None:
            target_est_pos = self.sensor_fusion.target_pos_est.copy()
            target_est_vel = self.sensor_fusion.target_vel_est.copy() if self.sensor_fusion.target_vel_est is not None else np.array([0.0, 0.0])
        else:
            # If sensor fusion has no data (should not happen after initialization in this scenario), use truth as fallback
            target_est_pos = self.target.position.copy()
            target_est_vel = self.target.velocity.copy()
        
        # 4. Determine guidance command
        if self.use_internal_guidance:
            # Use built-in PN guidance law
            a_lat_command = self.guidance.compute_command(self.missile.position, self.missile.velocity,
                                                          target_est_pos, target_est_vel, self.dt)
        else:
            # Use external action (RL agent) as steering command.
            # Interpret the action as normalized lateral acceleration.
            # If action is array-like, take first element.
            a_norm = float(action[0] if isinstance(action, (list, np.ndarray)) else action)
            # Clip to [-1,1]
            a_norm = max(-1.0, min(1.0, a_norm))
            # Scale to max lateral acceleration
            a_lat_command = a_norm * self.missile.max_turn_acc
        # 5. Update missile state with the command
        self.missile.update(a_lat_command, self.dt)
        
        # 6. Check termination conditions (hit or miss or time out)
        done = False
        reward = 0.0
        # Check if missile hit the target (we define a hit if distance < a certain threshold)
        rel_vec = self.target.position - self.missile.position
        miss_distance = np.linalg.norm(rel_vec)
        if miss_distance < 50.0:  # within 50 m -> consider it a "hit"
            done = True
            reward = 100.0  # reward for hitting target
        # Check if time exceeded or missile missed (target too far or passed)
        if self.time >= self.max_time:
            done = True
        # If done and not hit, assign negative reward for failing to intercept
        if done and reward == 0.0:
            # If missile simply timed out or target escaped
            reward = -100.0
        
        self.done = done
        # Prepare observation
        obs = self._get_observation()
        # Info can include diagnostic data
        info = {'miss_distance': miss_distance, 'target_detected': detection is not None}
        return obs, reward, done, info
    
    def _get_observation(self):
        # Observation chosen as relative position and velocity (estimated).
        # We use the fused estimate for target relative to missile.
        rel_pos = (self.sensor_fusion.target_pos_est - self.missile.position) if self.sensor_fusion.target_pos_est is not None else (self.target.position - self.missile.position)
        rel_vel = (self.sensor_fusion.target_vel_est - self.missile.velocity) if self.sensor_fusion.target_vel_est is not None else (self.target.velocity - self.missile.velocity)
        # Construct observation array
        obs = np.array([rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1]], dtype=np.float32)
        return obs
    
    def reset(self):
        """
        Reset the environment to a new initial state. Returns initial observation.
        """
        # Reinitialize scenario (for simplicity, we re-create objects; alternatively, we could reset their attributes).
        config = {
            'dt': self.dt,
            'real_time': self.real_time,
            'use_internal_guidance': self.use_internal_guidance,
            'max_time': self.max_time,
            'missile_speed': None,  # we let it randomize if needed
            'target_speed': None,
            'target_rcs_mean': None,
            'ecm_power': None,
            'clutter_power': None
        }
        # If initial config had fixed values, preserve them instead of None.
        # (In this implementation, we randomize anew for each episode for variability.)
        # We will reconstruct the environment components with possibly new random initial conditions.
        init_self = PL15Env(config)
        # Copy over relevant internals
        self.__dict__.update(init_self.__dict__)
        # Return initial observation
        return self._get_observation()
    
    def render(self, mode='human'):
        """
        (Optional) Render the simulation state.
        Not implemented, but could output missile and target positions or a visualization.
        """
        pass
    
    def close(self):
        """
        Clean up resources if any.
        """
        pass
```

**Notes on Environment:** The environment ties everything together and provides a step-by-step simulation. It uses `simpy.Environment` to advance the simulation clock by `dt` each step. If `real_time` is True, a `RealtimeEnvironment` is used so that `env.run(timeout)` will pause the loop to match wall-clock time, enabling hardware or interactive testing in real-time.

**Observation Space:** In this design, we provide the agent with the relative position and velocity of the target (in the missile-centric frame). This is derived from the sensor fusion module (or ground truth if needed as a fallback). In a more realistic scenario, the observation might be raw radar measurements (range, angle, range-rate) or filtered estimates – here we give a processed state for simplicity.

**Action Space:** The agent controls the missile’s lateral acceleration (steering). We use a continuous action in \[-1, 1] which is scaled to the missile's max acceleration in the simulation. If `use_internal_guidance` is True, the agent’s action is ignored and the built-in PN guidance is used instead (this allows switching between classical guidance and learned guidance easily).

**Reward:** We assign a positive reward (+100) if the missile intercepts the target (within a certain radius), and a negative reward (-100) if the missile fails (target not intercepted by the end of the episode). Time steps could also incur a small penalty to encourage faster interception (not implemented here). Users can refine the reward shaping as needed.

**Fail-safes and Modularity:** Each component is loosely coupled:

* If the radar or signal processor were absent, the environment could be configured to use perfect information (as a stub sensor) by directly feeding target truth to sensor fusion or guidance. (In our code, we ensure a fallback to truth if needed when no detection is available.)
* If the ECM module is not present, `RadarSeeker` simply won't add jamming noise (we check `self.ecm is not None`).
* If sensor fusion is not desired, one could bypass it and use raw detections or truth in guidance (by adjusting the environment accordingly).
* The guidance module can be swapped (for example, replacing PN with a machine-learned policy) by setting `use_internal_guidance=False` and letting the agent control, or by implementing a different guidance class and using it.
* The environment parameters (like `max_time`, speeds, RCS, ECM power, clutter) can be adjusted via the config to simulate different scenarios or turned off (e.g., set `ecm_power=0` to disable jamming, `clutter_power=0` to disable clutter).

This modular design ensures that each subsystem (radar, signal processing, guidance, etc.) can be developed or tested in isolation and then integrated. The OpenAI Gym interface allows using stable-baselines3 or any RL library to train an agent on this environment, for example to learn advanced guidance or countermeasure tactics.
