# storing event using pickle
import pickle

import numpy as np
import uproot
from chromo.constants import TeV
from chromo.kinematics import CenterOfMass
from chromo.models import Pythia8
from particle import Particle


def run_conditions():
    evt_kin = CenterOfMass(7 * TeV, particle1="proton", particle2="proton")
    return evt_kin


def particle_info(event, index):
    pid = event.pid[index]
    name = Particle.from_pdgid(pid).name

    child_indices = event.children[index] - 1
    child_pids = [event.pid[x] for x in child_indices]
    child_names = (
        Particle.from_pdgid(child_pids[0]).name,
        Particle.from_pdgid(child_pids[1]).name,
    )

    return {
        "particle_info": [name, pid, index],
        "children_info": [child_names, child_pids, child_indices],
    }


def main():
    evt_kin = run_conditions()
    config = ["HardQCD:gg2ccbar = on"]
    pythia = Pythia8(evt_kin, seed=45, config=config)

    all_events = []
    mult = []
    mult_plus = []
    mult_minus = []
    mult_zero = []
    f = uproot.recreate("uproot_pythia.root")
    uproot.recreate("uproot_pythia2.root")
    for event in pythia(4):
        # sanity check: skip if event has no tracks or has no D0 meson
        # if np.abs(ev.pid)==421:
        if len(event.pt) <= 0:  # or d0 not in event.pid:
            continue
        event_final = event.final_state()
        tr_pid = event_final.pid
        tr_eta = np.array(event_final.eta)
        tr_phi = np.array(event_final.phi)
        tr_pt = event_final.pt
        tr_mass = event_final.m
        tr_charge = event_final.charge

        # multiplicity
        plus = list(tr_charge).count(1)
        minus = list(tr_charge).count(-1)
        zero = list(tr_charge).count(0)
        mult_plus.append(plus)
        mult_minus.append(minus)
        mult_zero.append(zero)
        mult.append(plus + minus + zero)

        # create dictionary to store event properties event-by-event
        single_event = {}
        single_event["tr_eta"] = tr_eta
        single_event["tr_phi"] = tr_phi
        single_event["tr_pid"] = tr_pid
        single_event["tr_pt"] = tr_pt
        single_event["tr_mass"] = tr_mass

        all_events.append(single_event)
        f[f"event_{event.nevent-1}"] = single_event

    # store/write event properties using pickle
    with open("array_store.pkl", "wb") as file:
        pickle.dump(all_events, file)

    # retrieve event properties using pickle
    with open("array_store.pkl", "rb") as file:
        pickle.load(file)

    # store/write event properties using uproot
    # with uproot.recreate("uproot_pythia.root") as f:
    #     f["all_event_data"] = {"all_events": all_events}
    # retrieve event properties using uproot
    with uproot.open("uproot_pythia.root") as f1:
        f1.keys()


if __name__ == "__main__":
    main()
