import time
import random
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PHARMACOKINETIC / PHARMACODYNAMIC MODEL
# ==========================================
class PatientSimulator:
    """
    Simulates a single-compartment PK/PD model for drug concentration (e.g., insulin/anesthetic)
    and patient response (e.g., blood glucose or mean arterial pressure).
    """
    def __init__(self, initial_concentration=100.0):
        self.concentration = initial_concentration
        self.history_concentration = [initial_concentration]
        self.history_dose = [0.0]

    def step(self, infusion_rate, elimination_rate=0.05):
        # Simple PK differential update: dC/dt = Dose - Elimination * C
        self.concentration += infusion_rate - (self.elimination_rate_func() * self.concentration)
        self.history_concentration.append(self.concentration)
        self.history_dose.append(infusion_rate)
        return self.concentration

    def elimination_rate_func(self):
        return 0.08  # Metabolic clearance constant


# ==========================================
# 2. CONTROLLERS & REAL-TIME DECISION GUARD
# ==========================================
class AdvancedAIController:
    """
    Unverified Deep Learning / DRL model that calculates optimal dynamic dosage.
    Subject to non-deterministic execution time and potential latency spikes.
    """
    def compute_dosage(self, current_concentration, simulate_lag=False):
        start_time = time.time()
        
        # Simulate AI neural network computation
        base_computation_delay = 0.002 # 2ms normal
        if simulate_lag:
            base_computation_delay = 0.018 # 18ms lag (violates WCET budget of 10ms)
        
        time.sleep(base_computation_delay)
        
        # DRL dosage policy mock: target concentration is 80.0
        error = current_concentration - 80.0
        suggested_dose = max(0.0, 5.0 - (0.1 * error))
        
        exec_time = time.time() - start_time
        return suggested_dose, exec_time


class BaselinePIDController:
    """
    Deterministic, formally verified PID controller providing safe fallback control.
    Guaranteed low and bounded WCET.
    """
    def __init__(self):
        self.prev_error = 0.0

    def compute_dosage(self, current_concentration):
        start_time = time.time()
        error = current_concentration - 80.0
        # Simple proportional-derivative control law
        safe_dose = max(0.0, 3.0 - (0.05 * error))
        exec_time = time.time() - start_time
        return safe_dose, exec_time


class RealTimeSimplexGuard:
    """
    The Real-Time Decision Module (Guard). Enforces strict WCET bounds and 
    pharmacokinetic safety envelopes every periodic cycle.
    """
    def __init__(self, wcet_budget=0.010, safe_min=1.0, safe_max=8.0):
        self.wcet_budget = wcet_budget  # 10ms deadline budget
        self.safe_min = safe_min
        self.safe_max = safe_max
        self.fallback_count = 0

    def evaluate_and_dispatch(self, ai_dose, ai_exec_time, pid_controller, current_concentration):
        # Check 1: Timing Condition (WCET Deadline Enforcement)
        timing_violation = ai_exec_time > self.wcet_budget

        # Check 2: State-Space Safety Envelope Condition
        safety_violation = not (self.safe_min <= ai_dose <= self.safe_max)

        if timing_violation or safety_violation:
            self.fallback_count += 1
            # Instant failover to Baseline PID Controller
            pid_dose, pid_exec_time = pid_controller.compute_dosage(current_concentration)
            return pid_dose, "FALLBACK (PID)", timing_violation, safety_violation
        else:
            return ai_dose, "AI_CONTROLLER", False, False


# ==========================================
# 3. REAL-TIME SCHEDULING & SIMULATION LOOP
# ==========================================
def run_realtime_simulation(total_cycles=50):
    patient = PatientSimulator(initial_concentration=110.0)
    ai_controller = AdvancedAIController()
    pid_controller = BaselinePIDController()
    simplex_guard = RealTimeSimplexGuard(wcet_budget=0.010, safe_min=1.0, safe_max=7.5)

    timeline = []
    concentrations = []
    actuated_doses = []
    controller_sources = []
    faults_injected = []

    print(f"{'Cycle':<6} | {'Source':<18} | {'Exec Time (ms)':<15} | {'Dose':<8} | {'Concentration':<15} | {'Notes'}")
    print("-" * 80)

    for cycle in range(1, total_cycles + 1):
        # Inject artificial CPU lag/cache stall on specific cycles to stress-test the Guard
        simulate_lag = True if cycle in [12, 28, 41] else False

        # Step 1: Execute Advanced AI Controller
        ai_dose, ai_exec_time = ai_controller.compute_dosage(patient.concentration, simulate_lag=simulate_lag)

        # Step 2: Real-Time Guard Interception & Evaluation
        final_dose, source, t_violation, s_violation = simplex_guard.evaluate_and_dispatch(
            ai_dose, ai_exec_time, pid_controller, patient.concentration
        )

        # Step 3: Actuation & Plant Update
        new_concentration = patient.step(final_dose)

        # Logging metrics
        timeline.append(cycle)
        concentrations.append(new_concentration)
        actuated_doses.append(final_dose)
        controller_sources.append(source)
        faults_injected.append(simulate_lag)

        note = "NORMAL"
        if t_violation:
            note = "TIMEOUT (>10ms WCET)"
        elif s_violation:
            note = "UNSAFE DOSAGE"

        print(f"{cycle:<6} | {source:<18} | {ai_exec_time*1000:<15.2f} | {final_dose:<8.2f} | {new_concentration:<15.2f} | {note}")

    print("-" * 80)
    print(f"Total Failover Events Triggered by Guard: {simplex_guard.fallback_count}")
    return timeline, concentrations, actuated_doses, controller_sources, faults_injected


# ==========================================
# 4. VISUALIZATION SCRIPT (MATPLOTLIB)
# ==========================================
def plot_simulation_results(timeline, concentrations, actuated_doses, controller_sources, faults_injected):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot Patient Concentration & Safety Bounds
    ax1.plot(timeline, concentrations, label='Patient Blood Concentration', color='b', lw=2)
    ax1.axhline(y=80.0, color='g', linestyle='--', label='Target Setpoint (80 mg/L)')
    ax1.axhline(y=115.0, color='r', linestyle=':', label='Upper Safety Limit')
    ax1.axhline(y=45.0, color='r', linestyle=':', label='Lower Safety Limit')
    
    # Highlight Failover Events
    for i, source in enumerate(controller_sources):
        if "FALLBACK" in source:
            ax1.axvline(x=timeline[i], color='orange', alpha=0.4, label='Simplex Failover Switch' if i == 1 else "")

    ax1.set_ylabel('Drug Concentration (mg/L)')
    ax1.set_title('Real-Time Simplex Architecture: Patient Physiological Response & Runtime Assurance')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # Plot Actuated Doses & Controller Source
    colors = ['tab:blue' if src == 'AI_CONTROLLER' else 'tab:orange' for src in controller_sources]
    ax2.bar(timeline, actuated_doses, color=colors, alpha=0.8, label='Actuated Dose (Blue=AI, Orange=PID Fallback)')
    
    for i, faulted in enumerate(faults_injected):
        if faulted:
            ax2.annotate('CPU Stall / Lag', (timeline[i], actuated_doses[i]),
                         textcoords="offset points", xytext=(0,10), ha='center',
                         arrowprops=dict(arrowstyle='->', color='red'), color='red', fontsize=9)

    ax2.set_xlabel('Real-Time Execution Cycles (Periodic Interval T)')
    ax2.set_ylabel('Infusion Dose Rate')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('simplex_medical_simulation.png', dpi=300)
    print("\n[INFO] Simulation plot successfully saved as 'simplex_medical_simulation.png'.")
    plt.show()

if __name__ == "__main__":
    t, c, d, s, f = run_realtime_simulation(total_cycles=50)
    plot_simulation_results(t, c, d, s, f)