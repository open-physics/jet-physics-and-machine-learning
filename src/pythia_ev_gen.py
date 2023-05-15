import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import uproot
from chromo.constants import TeV
from chromo.kinematics import CenterOfMass
from chromo.models import Pythia8

# Create a new ROOT file and add a TTree
file = uproot.recreate("output_10k.root")


def run_conditions():
    evt_kin = CenterOfMass(7 * TeV, particle1="proton", particle2="proton")
    return evt_kin


def main():
    # x = [np.array([1,2,2], dtype=np.int32),np.array([2,1,1,1,1], dtype=np.int32)]
    # plt.hist(x,range=(1,3))
    # plt.show()

    # breakpoint()

    evt_kin = run_conditions()
    config = ["HardQCD:gg2ccbar = on"]
    pythia = Pythia8(evt_kin, seed=45, config=config)
    d0 = 421

    evt_list = []
    pt_avg_list = []
    eta_list = []
    pt_list = []
    phi_list = []
    chrg_list = []
    mass_list = []
    pid_list = []
    mult = []
    mult_plus = []
    mult_minus = []
    mult_zero = []
    for event in pythia(300000):
        # sanity check: skip if event has no tracks or has no D0 meson
        if len(event.pt) <= 0 or d0 not in event.pid:
            continue
        event_final = event.final_state()
        tr_eta = event_final.eta
        tr_pid = event_final.pid
        tr_phi = event_final.phi
        tr_pt = event_final.pt
        tr_mass = event_final.m
        tr_charge = event_final.charge
        # tr_pt = ev.pt[np.abs(ev.pid) == 211]

        # sanity check: does avg_pt make sense?
        avg_pt = np.mean(tr_pt)
        pt_avg_list.append(avg_pt)

        # keep a record of event numbers
        evt_list.append(event_final.nevent)
        # store event properties
        phi_list += list(tr_phi)
        pt_list += list(tr_pt)
        chrg_list += list(tr_charge)
        eta_list += list(tr_eta)
        mass_list += list(tr_mass)

        # if np.abs(ev.pid)==421:
        pid_list += list(tr_pid)
        # multiplicity
        plus = list(tr_charge).count(1)
        minus = list(tr_charge).count(-1)
        zero = list(tr_charge).count(0)
        mult_plus.append(plus)
        mult_minus.append(minus)
        mult_zero.append(zero)
        # multiplicity.append(len(event))
        mult.append(plus + minus + zero)

    # Multiplicity
    # ------------
    sns.color_palette("pastel")
    plots = [mult, mult_plus, mult_minus, mult_zero]
    labels = [
        "Multiplicity",
        "Multiplicity of +1",
        "Multiplicity of -1",
        "Multiplicity of 0",
    ]
    for i in range(len(plots)):
        fig, ax = plt.subplots()  # figsize=(7,5))
        sns.despine(fig)

        sns.histplot(
            plots[i],
            # x="price", hue="cut",
            edgecolor=".3",
            linewidth=0.5,
            binwidth=10,
            binrange=(0, 1200),
            legend=True,
        )
        # legend = ax.get_legend()
        # handles = legend.legendHandles
        # legend.remove()
        # ax.legend(handles, [labels[i]], title='Multiplicity')
        ax.set_xlabel(labels[i])
        plt.draw()
        plt.savefig(f"test_{i}.png")
        # ax.set_ylim(0,300)

    # Phi distribution
    # ----------------
    fig, ax = plt.subplots()  # figsize=(7,5))
    sns.despine(fig)

    sns.histplot(
        phi_list,
        # x="price", hue="cut",
        edgecolor=".3",
        linewidth=0.5,
        # binwidth=10,
        # binrange=(0,1200),
        # legend=True,
    )
    ax.set_xlabel(r"$\phi$")
    plt.draw()
    plt.savefig("phi.png")

    # pT distribution
    # ---------------
    fig, ax = plt.subplots()  # figsize=(7,5))
    # sns.despine(fig)

    # sns.histplot(
    #     pt_list,
    #     # x="price", hue="cut",
    #     edgecolor=".3",
    #     linewidth=.5,
    # )
    import boost_histogram as bh

    hist_pt = bh.Histogram(bh.axis.Regular(500, -2, 50))
    hist_pt.fill(np.array(pt_list))
    plt.stairs(hist_pt.values(), hist_pt.axes[0].edges)
    plt.yscale("log")
    ax.set_xlabel(r"$p_\mathrm{T}^{}$")
    ax.set_ylabel("Count")
    plt.draw()
    plt.savefig("pt.png")

    # Eta distribution
    # ---------------
    fig, ax = plt.subplots()  # figsize=(7,5))
    sns.despine(fig)

    sns.histplot(
        eta_list,
        # x="price", hue="cut",
        edgecolor=".3",
        linewidth=0.5,
    )
    ax.set_xlabel(r"$\eta$")
    plt.draw()
    plt.savefig("eta.png")
    # plt.hist(pt_avg_list, bins=10, edgecolor="blue", linewidth=0.5)
    # plt.xlabel("average pt")
    # plt.ylabel("# events")
    # plt.title("Mean pt distribution")
    # plt.savefig("mean_pt.pdf")
    # plt.show()

    # Fill the TTree with data
    # file["mytree"] = {'n_event':evt_list,
    #                   'pt': pt_avg_list,
    #                   'eta': eta_list,
    #                   'phi': phi_list}
    # 'charge':chrg_list,
    # 'mass':mass_list}
    # 'pid':pid_list}

    # Read the TTree back from the file to check the data was saved correctly
    # from pprint import pprint
    # with uproot.open("output_10k.root") as f:
    #     # pprint(f["mytree"].arrays())
    #     pprint(f["mytree"].keys())


if __name__ == "__main__":
    main()
