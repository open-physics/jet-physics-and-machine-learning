import uproot
from chromo.kinematics import CenterOfMass
from chromo.models import Pythia8
from chromo.constants import TeV
import numpy as np
import matplotlib.pyplot as plt


# Create a new ROOT file and add a TTree
file = uproot.recreate("output.root") 

def run_conditions():
    evt_kin = CenterOfMass(7 * TeV, particle1="proton", particle2="proton")
    return evt_kin


def main():
    
    evt_kin = run_conditions()
    pythia = Pythia8(evt_kin, seed=45)

    evt_list = []
    pt_list = []
    eta_list = []
    for event in pythia(5000):
        ev = event.final_state()
        tr_eta = ev.eta
        tr_pid = np.abs(ev.pid)
        tr_phi = ev.phi
        tr_pt = ev.pt[np.abs(ev.pid) == 211]
        tr_mass = ev.m
        if len(ev.pt)>0:
            avg_pt = np.mean(tr_pt)
            pt_list.append(avg_pt)
            evt_list.append(ev.nevent)

        if len(ev.eta)>0:
            avg_eta = np.mean(tr_eta)
            eta_list.append(avg_eta)

    breakpoint()

    plt.hist(pt_list, bins=10, edgecolor="blue", linewidth=0.5)
    plt.xlabel("average pt")
    plt.ylabel("# events")
    plt.title("Mean pt distribution")
    # fig.savefig("mean_pt.png")
    plt.show()


    # Fill the TTree with data
    file["mytree"] = {'pt': pt_list, 'eta': eta_list}
    
    plt.hist(eta_list, bins=10, edgecolor="red", linewidth=0.5)
    plt.show()
    # breakpoint()
    
    # Read the TTree back from the file to check the data was saved correctly
    from pprint import pprint
    with uproot.open("output.root") as f:
        pprint(f["mytree"].arrays())
        pprint(f["mytree"].keys())


if __name__ == "__main__":
    main()
