from chromo.kinematics import CenterOfMass
from chromo.models import Pythia8
from chromo.constants import TeV
import numpy as np
import matplotlib.pyplot as plt


def run_conditions():
    evt_kin = CenterOfMass(7 * TeV, particle1="proton", particle2="proton")
    return evt_kin


def main():
    
    evt_kin = run_conditions()
    pythia = Pythia8(evt_kin, seed=4)

    nevent = 0
    avg_pt = 0
    evt_list = []
    pt_list = []
    for event in pythia(10):
        ev = event.final_state()
        tr_eta = ev.eta
        tr_pid = np.abs(ev.pid)
        tr_phi = ev.phi
        tr_pt = ev.pt[np.abs(ev.pid) == 211]
        tr_mass = ev.m
        if len(ev.pt)>0:
            nevent +=1
            avg_pt += np.mean(ev.pt)

        pt_list.append(tr_pt)
        evt_list.append(nevent)

    plt.plot(pt_list, evt_list, color="blue", linewidth=7.0)
    plt.xlabel("average pt")
    plt.ylabel("# events")
    plt.title("Mean pt distribution")
    # fig.savefig("mean_pt.png")
    plt.show()

    breakpoint()
    

if __name__ == "__main__":
    main()
