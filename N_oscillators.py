# Import everything i need

import numpy as np
import csv
import json
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import math

# Define my constants and read initial states and input parameters from json files
with open("input_initState.json", "r") as g:
    init_1 = json.load(g)
with open("input_parameters.json", "r") as f:
    param1 = json.load(f)
with open("frequencies.txt", "w") as f:
    pass

k = param1["k"]
l_rest = param1["l_rest"]
dt_list = []
m = param1["mass"]

# Main Part - Compute coordinates function
def compute_coordinates(setting):
    for frame in range(param1["frames_num"][setting]):
        # The goal in this  loop eventualy is to provide: 1) A list of lists of the positions of each oscillator for every frame. same goes for accelerations, and the half and full step velocities.
        # find initials
        if frame == 0:
            # define positions
            position_0 = position_list[frame]
            # I chose to create a new list of the effective length of the spring called "delta_Kfiz" with zeros on both ends so i can reduce for each osc the length of the former. this list will be re-written every run.
            delta_kfiz_0 = [0, 0]
            force_0 = []
            # find spring effective length for frame_0
            for i in range(len(position_0) - 1):
                delta_kfiz_0.insert(-1, ((position_0[i + 1] - position_0[i]) - l_rest))
            dx_list.append(delta_kfiz_0)
            # find force for spring's eff length
            for i in range(len(delta_kfiz_0) - 1):
                force_0.append(k * (-delta_kfiz_0[i] + delta_kfiz_0[i + 1]))
            # find accel of each mass
            a_0 = np.array(force_0) / param1["mass"]
            # The next 'if' statements are supposed to take care of the open/closed ends (so if the end is closed the acc is 0)
            if param1["first_is_open"] == False:
                a_0[0] = 0
            if param1["last_is_open"] == False:
                a_0[-1] = 0
            # find v_half and append to half-step velocity list
            velo_list.append(np.array(init_1["v"]) + (dt / 2) * a_0)
            acc_list.append(a_0)

        else:
            # find i position
            position = position_list[frame - 1] + dt * np.array(velo_list[frame - 1])
            # add to positions list
            position_list.append(position)
            # find spring eff length for the i frame
            delta_kfiz = [0, 0]
            force = []
            for i in range(len(position) - 1):
                delta_kfiz.insert(-1, ((position[i + 1] - position[i]) - l_rest))
            dx_list.append(delta_kfiz)
            for i in range(len(delta_kfiz) - 1):
                force.append(k * (-delta_kfiz[i] + delta_kfiz[i + 1]))
            # find accel of each mass
            acc = np.array(force) / param1["mass"]
            if param1["first_is_open"] == False:
                acc[0] = 0
            if param1["last_is_open"] == False:
                acc[-1] = 0
            # find velocity for i+1/2 frame
            v_i_half = velo_list[frame - 1] + acc * dt
            # add to velocity list
            velo_list.append(v_i_half)
            v_i_whole = v_i_half - acc * dt / 2
            velo_whole.append(v_i_whole)
            acc_list.append(acc)
    velo_whole.insert(0, init_1["v"])
    # Make a list of the position list (instead of a numpy array) and write to "frames" file
    fixed_position_list = [i.tolist() for i in position_list]
    with open("frames.txt", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(fixed_position_list)
    return [dx_list, velo_whole, position_list, acc]


# Define the function that computes energies
def calc_E(dx, v):
    # calculate E_k and E_p
    Ep_list = []
    Ek_list = []
    for i in dx:
        E_p = sum(0.5 * k * (np.array(i)) ** 2)
        Ep_list.append(E_p)
    for i in v:
        E_k = sum(0.5 * m * (np.array(i)) ** 2)
        Ek_list.append(E_k)
    E_tot = (np.array(Ek_list) + np.array(Ep_list)).tolist()
    return [Ep_list, Ek_list, E_tot]


# Define mean_frequency function
def mean_frequency(dx, t, dt):
    # find max positions and their indices in the position list and take only the index without the dict "[0]"
    maxima_indices = find_peaks(dx)[0]
    # find PeriodTime and append to a period time list
    T_list = []
    for i in range(len(maxima_indices) - 1):
        T = (maxima_indices[i + 1] - maxima_indices[i]) * dt
        T_list.append(T)
    T_avg = (sum(T_list)) / (len(T_list))
    if T_avg == 0:
        print("T_avg = 0 for dt = ", dt)
    omega_approx = (2 * np.pi) / T_avg
    omega_analytic = param1["omega"]
    rel_error = abs(omega_analytic - omega_approx) / omega_analytic
    # write these results to file
    with open("frequencies.txt", "a+", newline="") as g:
        writer = csv.writer(g, delimiter="\t")
        writer.writerow([omega_analytic, omega_approx, rel_error])


# Define function for calculation of cs__approx
def calc_wave_vel(v, t):
    for i in v:
        if i != 0:
            moving_frame = v.index(i)
            break
    t_fin = t[moving_frame]
    cs_approx = ((param1["osc_num"] - 1) * l_rest) / t_fin
    cs = l_rest * (np.sqrt(k / m))
    rel_cs_error = abs(cs - cs_approx) / cs
    with open("v wave.txt", "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([cs, cs_approx, rel_cs_error])
    return cs_approx


# Define function for kinematics plots
def coordinate_plots(dx, v, a, t):
    fig, ax = plt.subplots(3, sharey=True)
    ax[0].plot(t, dx)
    ax[0].set_title("dx(t)")
    ax[1].plot(t, v, "tab:green")
    ax[1].set_title("v(t)")
    ax[2].plot(t, a, "tab:red")
    ax[2].set_title("a(t)")
    fig.tight_layout(pad=2.0)
    plt.savefig("kinematics.png")
    plt.close()


# Define function for energy plots
def energy_plot(Ek, Ep, E_tot, t):
    fig, ax = plt.subplots(3, sharey=True)
    ax[0].plot(t, E_tot)
    ax[0].set_title("E_tot(t)")
    ax[1].plot(t, Ek, "tab:pink")
    ax[1].set_title("Ek(t)")
    ax[2].plot(t, Ep, "tab:orange")
    ax[2].set_title("Ep(t)")
    fig.tight_layout(pad=2.0)
    plt.savefig("Energy.png")
    plt.close()


# Define function that reads returns the error of the approximate frequency for each frame
def read_freq_error(freq_file):
    with open(freq_file, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        cols = []
        for row in reader:
            if row != []:
                cols.append(float(row[2]))
    return cols


# Define function for frequency error plot
def freq_error_plot(freq_error, dt_list):
    fig, ax = plt.subplots(1)
    ax.plot(dt_list, freq_error)
    ax.invert_xaxis()
    ax.set_title("freq_error(t)")
    plt.savefig("freq vs dt.png")
    plt.close()


# Big loop - runs on every setting (by setting i mean number of frames and dt)
for setting in range(len(param1["frames_num"])):
    dt = param1["dt"][setting]
    position_list = [np.array(init_1["x"])]
    # "velo_list" will be a list of lists of the half-step velocities for each frame
    velo_list = []
    # "dx_list" will be a list of the effective length of the spring for each frame
    dx_list = []
    # "velo_whole" will be a list of lists of the full-step velocities for each frame
    velo_whole = []
    acc_list = []
    coordinate_results = compute_coordinates(setting)
    dx_list = coordinate_results[0]
    velo_whole = coordinate_results[1]
    position_list = coordinate_results[2]
    acc = coordinate_results[3]
    energy_results = calc_E(dx_list, velo_whole)
    Ep = energy_results[0]
    Ek = energy_results[1]
    E_tot = energy_results[2]
    # compute for all cases
    # this will be some helpful variables:
    t = [i * dt for i in range(param1["frames_num"][setting])]
    # transpose so that the columns (position of each osc) become rows and easy to manipulate
    postions_transposed = np.transpose(position_list)
    position_list_second_osc = postions_transposed[1] - l_rest
    # In the next line i transposed the array of full-step velocities, then took the last row (which is after the trans' the velocity of the last osc') and converted to a list.
    Last_velo = np.array(velo_whole).T[-1].tolist()
    if param1["osc_num"] == 3 or param1["osc_num"] == 4:
        mean_frequency(position_list_second_osc, t, dt)
    if param1["osc_num"] > 100:
        calc_wave_vel(Last_velo, t)
    # Take dx,v_whole,acc of the middle osc
    revised_dx = np.array(dx_list).T[:-1].T.tolist()
    middleman = math.ceil(len(dx_list[0]) / 2) - 1
    middle_dx = np.array(revised_dx).T[middleman].tolist()
    middle_v_whole = np.array(velo_whole).T[middleman].tolist()
    middle_acc = np.array(acc_list).T[middleman].tolist()
    coordinate_plots(middle_dx, middle_v_whole, middle_acc, t)
    energy_plot(Ek, Ep, E_tot, t)
    dt_list.append(dt)

if param1["osc_num"] == 3 or param1["osc_num"] == 4:
    freq_error = read_freq_error("frequencies.txt")
    freq_error_plot(freq_error, dt_list)


# Bonus:
# In the energies graph, i would expect the total energy to be constant, because of energy conservation. also i would expect inverse ratio between the E_k and E_p oscillations, which also can be explained by energy conservation.
# Ultimatley, the graph for every configuration does satisfy our expectation - we see that the total energy is approximatley constant, and the potential and kinectic energy oscillates with a pi/2 phase (when one is at maximum the other is at minimum). but if we look closely, we see that for dt > 0.5 the approximate constant energy is not constant at all. the best explanation i could think of is that the energy is conserved only for the limit where dt approches 0, because only then can we apply noether's theorm and the lagrangian of the system will be invariant to inititesimal changes in t. meaning that the smaller dt is, the better the approximation in which dt is infinitesimal, and we can apply noether's theorm and get energy conservation.
