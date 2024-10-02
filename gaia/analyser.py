import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal

CUT_FREQ = 10
SAMPLE_FREQ = 500
FORCE_PLATE_LENGTH = 0.6  # in meters
NUM_FORCE_PLATES = 4
REPORT_KEYS = {
    "total_time": "Total Time [s]",
    "number_of_steps": "Number of Steps",
    "sym_idx_fx": "Symmetry Index Fx",
    "sym_idx_fy": "Symmetry Index Fy",
    "sym_idx_fz": "Symmetry Index Fz",
    "integral_r": "Integral Fy (R) [N·s]",
    "integral_l": "Integral Fy (L) [N·s]",
    "mech_energ_r": "Mechanical Energy Expenditure (R) [J]",
    "mech_energ_l": "Mechanical Energy Expenditure (L) [J]",
    "single_support_r": "Single Support Phase (R) [s]",
    "single_support_l": "Single Support Phase (L) [s]",
    "double_support": "Double Support Phase [s]",
    "velociry_ms": "Average Velocity [m/s]",
    "velociry_kmh": "Average Velocity [km/h]",
    "step_freq": "Step Frequency [step/s]",
    "step_time_num": "Step Time (Number of Steps Based) [s]",
    "step_time_sup": "Step Time (Support Phase Based) [s]",
}


def remove_nan_values(values):
    try:
        if math.isnan(float(values[4])) and math.isnan(float(values[7])):
            return False
    except ValueError:
        pass
    return True


def remove_zero(data):
    data = [i for i in data if i != 0]
    return data


def get_average(value):
    value = sum(value) / len(value)
    return value


def load_data(file_path):
    """Load the data from a text file."""
    data = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            values = line.split()
            if len(values) == 8:
                if remove_nan_values(values):
                    data.append(values)

    # Convert to DataFrame
    df = pd.DataFrame(
        data, columns=["Frame", "Time", "Fx1", "Fy1", "Fz1", "Fx2", "Fy2", "Fz2"]
    )
    df = df.apply(
        pd.to_numeric, errors="coerce"
    )  # Convert all columns to numeric, setting errors to NaN

    # Apply butterworth lowpass filter
    df["Fx1"] = butter_lowpass_filter(df["Fx1"], CUT_FREQ, SAMPLE_FREQ)
    df["Fx2"] = butter_lowpass_filter(df["Fx2"], CUT_FREQ, SAMPLE_FREQ)
    df["Fy1"] = butter_lowpass_filter(df["Fy1"], CUT_FREQ, SAMPLE_FREQ)
    df["Fy2"] = butter_lowpass_filter(df["Fy2"], CUT_FREQ, SAMPLE_FREQ)
    df["Fz1"] = butter_lowpass_filter(df["Fz1"], CUT_FREQ, SAMPLE_FREQ)
    df["Fz2"] = butter_lowpass_filter(df["Fz2"], CUT_FREQ, SAMPLE_FREQ)
    return df


def plot_raw_data(raw_data):
    """Load and plot raw force data for a given file."""

    # Visualize the raw data
    plt.figure(figsize=(15, 10))

    # Plot raw forces in the X direction
    plt.subplot(3, 1, 1)
    plt.plot(raw_data["Time"], raw_data["Fx1"], label="Fx1 (Right Foot)")
    plt.plot(raw_data["Time"], raw_data["Fx2"], label="Fx2 (Left Foot)")
    plt.title(f"Raw Force in X Direction - {file_path}")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend()

    # Plot raw forces in the Y direction
    plt.subplot(3, 1, 2)
    plt.plot(raw_data["Time"], raw_data["Fy1"], label="Fy1 (Right Foot)")
    plt.plot(raw_data["Time"], raw_data["Fy2"], label="Fy2 (Left Foot)")
    plt.title(f"Raw Force in Y Direction - {file_path}")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend()

    # Plot raw forces in the Z direction
    plt.subplot(3, 1, 3)
    plt.plot(raw_data["Time"], raw_data["Fz1"], label="Fz1 (Right Foot)")
    plt.plot(raw_data["Time"], raw_data["Fz2"], label="Fz2 (Left Foot)")
    plt.title(f"Raw Force in Z Direction - {file_path}")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_y_direction(raw_data):
    """Plot force data in the Y direction for a given file."""
    # Visualize the force in the Y direction
    plt.figure(figsize=(10, 6))

    plt.plot(raw_data["Time"], raw_data["Fy1"], label="Fy1 (Right Foot)")
    plt.plot(raw_data["Time"], raw_data["Fy2"], label="Fy2 (Left Foot)")
    plt.title(f"Force in Y Direction - {file_path}")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    # plt.savefig("/content/y_direction.png")
    plt.legend()

    plt.tight_layout()
    plt.show()


def get_support_phase(raw, r_foot, l_foot):
    step_time_right = []
    step_time_left = []
    step_time_both = []

    step_right = step_left = step_both = 0

    for _, row in raw.iterrows():
        right_nan = math.isnan(row[r_foot])
        left_nan = math.isnan(row[l_foot])

        if not right_nan and left_nan:
            if step_right == len(step_time_right):
                step_time_right.append(0)
            step_time_right[step_right] += 1

            if step_left < len(step_time_left) and step_time_left[step_left] != 0:
                step_left += 1
            if step_both < len(step_time_both) and step_time_both[step_both] != 0:
                step_both += 1

        elif right_nan and not left_nan:
            if step_left == len(step_time_left):
                step_time_left.append(0)
            step_time_left[step_left] += 1

            if step_right < len(step_time_right) and step_time_right[step_right] != 0:
                step_right += 1
            if step_both < len(step_time_both) and step_time_both[step_both] != 0:
                step_both += 1

        elif not right_nan and not left_nan:
            if step_both == len(step_time_both):
                step_time_both.append(0)
            step_time_both[step_both] += 1

            if step_right < len(step_time_right) and step_time_right[step_right] != 0:
                step_right += 1
            if step_left < len(step_time_left) and step_time_left[step_left] != 0:
                step_left += 1

    step_time_right = [time / SAMPLE_FREQ for time in remove_zero(step_time_right)]
    step_time_left = [time / SAMPLE_FREQ for time in remove_zero(step_time_left)]
    step_time_both = [time / SAMPLE_FREQ for time in remove_zero(step_time_both)]

    return step_time_right, step_time_left, step_time_both


def get_velocity(data):
    distance = NUM_FORCE_PLATES * FORCE_PLATE_LENGTH
    velocity = distance / data["total_time"]
    return velocity


def get_integrals(raw_data):
    """Calculate the integrals of the individual force-time curves in the Y direction for both feet."""
    # Remove rows with NaN values
    raw_data = raw_data.dropna(subset=["Fy1", "Fy2"])

    # Calculate the integral using the trapezoidal rule for each curve
    integral_Fy1 = np.trapz(raw_data["Fy1"], raw_data["Time"])
    integral_Fy2 = np.trapz(raw_data["Fy2"], raw_data["Time"])

    return integral_Fy1, integral_Fy2


def get_mechanical_energy(raw_data):
    """Estimate the mechanical energy expenditure based on the force and movement data."""
    # Remove rows with NaN values
    raw_data = raw_data.dropna()

    # Calculate the time differences
    dt = np.diff(raw_data["Time"])
    dt = np.insert(dt, 0, dt[0])  # Insert the first time difference to match the length

    # Calculate velocities
    vx1 = np.cumsum(raw_data["Fx1"] * dt)
    vy1 = np.cumsum(raw_data["Fy1"] * dt)
    vz1 = np.cumsum(raw_data["Fz1"] * dt)

    vx2 = np.cumsum(raw_data["Fx2"] * dt)
    vy2 = np.cumsum(raw_data["Fy2"] * dt)
    vz2 = np.cumsum(raw_data["Fz2"] * dt)

    # Calculate displacements
    dx1 = np.cumsum(vx1 * dt)
    dy1 = np.cumsum(vy1 * dt)
    dz1 = np.cumsum(vz1 * dt)

    dx2 = np.cumsum(vx2 * dt)
    dy2 = np.cumsum(vy2 * dt)
    dz2 = np.cumsum(vz2 * dt)

    # Calculate the work done by each force component
    work_x1 = np.trapz(raw_data["Fx1"] * dx1, raw_data["Time"])
    work_y1 = np.trapz(raw_data["Fy1"] * dy1, raw_data["Time"])
    work_z1 = np.trapz(raw_data["Fz1"] * dz1, raw_data["Time"])

    work_x2 = np.trapz(raw_data["Fx2"] * dx2, raw_data["Time"])
    work_y2 = np.trapz(raw_data["Fy2"] * dy2, raw_data["Time"])
    work_z2 = np.trapz(raw_data["Fz2"] * dz2, raw_data["Time"])

    # Total mechanical energy expenditure
    total_energy_right_foot = work_x1 + work_y1 + work_z1
    total_energy_left_foot = work_x2 + work_y2 + work_z2

    return total_energy_right_foot, total_energy_left_foot


def get_step_time_num(total_time, number_of_steps):
    """Estimate the step time based on the total time and number of steps."""
    step_time = total_time / number_of_steps
    return step_time


def get_step_time_sup(single_right, single_left, double):
    """Estimate the step time based on the support phase data."""
    average_right = get_average(single_right)
    average_left = get_average(single_left)
    average_double = get_average(double)
    step_time = (average_right + average_left + average_double) / 2
    return step_time


def get_step_frequency(number_of_steps, total_time):
    """Estimate the step frequency in steps per second (Hz)."""
    step_frequency = number_of_steps / total_time
    return step_frequency


def get_number_of_steps(single_support_right, single_support_left):
    """ ""Calculate the number of steps taken based on support phases."""
    return len(single_support_right) + len(single_support_left)


def get_gait_symmetry(raw_data):
    """Calculate gait symmetry based on peak forces."""
    peaks = {
        "Fx1": raw_data["Fx1"].max(),
        "Fy1": raw_data["Fy1"].max(),
        "Fz1": raw_data["Fz1"].max(),
        "Fx2": raw_data["Fx2"].max(),
        "Fy2": raw_data["Fy2"].max(),
        "Fz2": raw_data["Fz2"].max(),
    }

    symmetry_indices = {
        "sym_idx_fx": (
            2 * abs(peaks["Fx1"] - peaks["Fx2"]) / (peaks["Fx1"] + peaks["Fx2"])
        ),
        "sym_idx_fy": (
            2 * abs(peaks["Fy1"] - peaks["Fy2"]) / (peaks["Fy1"] + peaks["Fy2"])
        ),
        "sym_idx_fz": (
            2 * abs(peaks["Fz1"] - peaks["Fz2"]) / (peaks["Fz1"] + peaks["Fz2"])
        ),
    }

    return symmetry_indices


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    # Identificar segmentos válidos (não NaN)
    isnan = np.isnan(data)
    filtered_data = np.copy(
        data
    )  # Criar uma cópia dos dados para preencher com dados filtrados
    if not isnan.any():  # Se não houver NaNs, filtrar todos os dados
        filtered_data = signal.filtfilt(b, a, data)
    else:
        segments = np.where(np.diff(isnan.astype(int)) != 0)[0] + 1
        segments = np.insert(segments, 0, 0)
        segments = np.append(segments, len(data))
        for start, end in zip(segments[:-1], segments[1:]):
            if not isnan[start:end].any():
                filtered_data[start:end] = signal.filtfilt(b, a, data[start:end])
    return filtered_data


def export_report(data):
    # Definir o caminho do arquivo CSV
    report = "/content/report.csv"

    # Verificar se o arquivo já existe
    if not os.path.isfile(report):
        data.to_csv(report, index=False)
    else:
        data.to_csv(report, mode="a", header=False, index=False)


def print_report(data):
    data = data.drop("name", axis=1)
    for key, value in data.items():
        print(f"{REPORT_KEYS[key]}: {float(value[0]):.2f}")


def analyze_file(file_path):
    """Analyze the force data for a given file."""
    # Initialize report data
    report_data = pd.DataFrame([file_path], columns=["name"])

    # Load data
    raw_data = load_data(file_path)

    # Plot the forces
    plot_raw_data(raw_data)

    # Plot the force in the Y direction only
    plot_y_direction(raw_data)

    # Drop rows where 'Time' is NaN
    time = raw_data.dropna(subset=["Time"])

    # Calculate the total time
    total_time = time["Time"].iloc[-1] - time["Time"].iloc[0]  # in seconds
    report_data["total_time"] = total_time

    # Calculate and display gait symmetry
    symmetry_indices = get_gait_symmetry(raw_data)
    for key, value in symmetry_indices.items():
        report_data[key] = value

    # Calculate and print the integrals for the Y direction
    integral_r, integral_l = get_integrals(raw_data)
    report_data["integral_r"] = integral_r
    report_data["integral_l"] = integral_l

    # Estimate the mechanical energy expenditure
    mech_energ_r, mech_energ_l = get_mechanical_energy(raw_data)
    report_data["mech_energ_r"] = mech_energ_r
    report_data["mech_energ_l"] = mech_energ_l

    # Estimate Support Phase
    single_support_right_list, single_support_left_list, double_support_list = (
        get_support_phase(raw_data, "Fy1", "Fy2")
    )

    report_data["single_support_r"] = get_average(single_support_right_list)
    report_data["single_support_l"] = get_average(single_support_left_list)
    report_data["double_support"] = get_average(double_support_list)

    # Number of steps taken
    number_of_steps = get_number_of_steps(
        single_support_right_list, single_support_left_list
    )
    report_data["number_of_steps"] = number_of_steps

    # Calculate the step time
    step_time_num = get_step_time_num(total_time, number_of_steps)
    report_data["step_time_num"] = step_time_num
    step_time_sup = get_step_time_sup(
        single_support_right_list, single_support_left_list, double_support_list
    )
    report_data["step_time_sup"] = step_time_sup

    # Estimate Velocity Based on Support Phases
    report_data["velociry_ms"] = get_velocity(report_data)
    report_data["velociry_kmh"] = report_data["velociry_ms"] * 3.6

    # Estimate the step frequency
    step_frequency = get_step_frequency(number_of_steps, total_time)
    report_data["step_freq"] = step_frequency

    # Export the report
    export_report(report_data)

    # Print the report
    print_report(report_data)


# Running the analysis on the provided file
file_path = "/content/05-rapida-force.emt"
analyze_file(file_path)
