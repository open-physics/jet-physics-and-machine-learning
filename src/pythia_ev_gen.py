from cycler import mul
import uproot
from chromo.kinematics import CenterOfMass
from chromo.models import Pythia8
from chromo.constants import TeV
import numpy as np
import matplotlib.pyplot as plt


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
    phi_list = []
    chrg_list = []
    mass_list = []
    pid_list = []
    multiplicity = []
    multiplicity_plus = []
    multiplicity_minus = []
    multiplicity_zero = []
    for event in pythia(100):
        # sanity check: skip if event has no tracks or has no D0 meson
        if len(event.pt) <= 0 or d0 not in event.pid:
            continue
        tr_eta = event.eta
        tr_pid = event.pid
        tr_phi = event.phi
        tr_pt = event.pt
        tr_mass = event.m
        tr_charge = event.charge
        # tr_pt = ev.pt[np.abs(ev.pid) == 211]

        # sanity check: does avg_pt make sense?
        avg_pt = np.mean(tr_pt)
        pt_avg_list.append(avg_pt)

        # keep a record of event numbers
        evt_list.append(event.nevent)
        # store event properties
        phi_list.append(list(tr_phi))
        chrg_list.append(list(tr_charge))
        eta_list.append(list(tr_eta))
        mass_list.append(list(tr_mass))

        # if np.abs(ev.pid)==421:
        pid_list += list(tr_pid)
        # multiplicity
        multiplicity_plus.append(list(tr_charge).count(1))
        multiplicity_minus.append(list(tr_charge).count(-1))
        multiplicity_zero.append(list(tr_charge).count(0))
        # multiplicity.append(len(event))
        multiplicity.append(len(tr_charge))

    breakpoint()




    # plt.hist(pt_avg_list, bins=10, edgecolor="blue", linewidth=0.5)
    # plt.xlabel("average pt")
    # plt.ylabel("# events")
    # plt.title("Mean pt distribution")
    # plt.savefig("mean_pt.pdf")
    # plt.show()

    fig, (ax0, ax1)= plt.subplots(2)

    ax0.hist([multiplicity_zero, multiplicity_minus, multiplicity_plus], bins=10, linewidth=0.5)
    ax1.hist(multiplicity, bins=10, linewidth=0.5)
    
    plt.draw()
    plt.savefig("multiplicity.pdf")

    plt.hist(eta_list, bins=10, edgecolor="magenta", linewidth=0.5)
    plt.set_xlim(-15, 15)
    plt.xlabel("eta")
    plt.ylabel("# events")
    plt.title("Eta distribution")
    plt.savefig("eta.pdf")
    plt.show()


    # Fill the TTree with data
    file["mytree"] = {'n_event':evt_list,
                      'pt': pt_avg_list, 
                      'eta': eta_list, 
                      'phi': phi_list}
                      # 'charge':chrg_list,
                      # 'mass':mass_list}
                      # 'pid':pid_list}
    
    
    # Read the TTree back from the file to check the data was saved correctly
    from pprint import pprint
    with uproot.open("output_10k.root") as f:
        # pprint(f["mytree"].arrays())
        pprint(f["mytree"].keys())


if __name__ == "__main__":
    main()
