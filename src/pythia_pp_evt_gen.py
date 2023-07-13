# storing event using uproot and pickle file format
import os
import pickle

import numpy as np
import uproot
from chromo.constants import TeV
from chromo.kinematics import CenterOfMass
from chromo.models import Pythia8
from particle import Particle

workdir = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
file_name = f"{workdir}/data/uproot_pythia.root"


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
    # ReadString("HardQCD:qqbar2ccbar = on");
    # heavy quark mass
    # ReadString("ParticleData:mcRun = 1.5");

    pythia = Pythia8(evt_kin, seed=45, config=config)

    all_events = []
    f = uproot.recreate(file_name)
    # uproot.recreate("uproot_pythia2.root")
    for event in pythia(5):
        # sanity check: skip if event has no tracks or has no D0 meson
        # if np.abs(ev.pid)==421:
        if len(event.pt) <= 0:  # or d0 not in event.pid:
            continue
        event_final = event.final_state()
        pid = event_final.pid
        np.array(event_final.eta)
        np.array(event_final.phi)
        px = np.array(event_final.px)
        py = np.array(event_final.py)
        pz = np.array(event_final.pz)
        en = np.array(event_final.en)
        charge = event_final.charge

        # create dictionary to store event properties event-by-event
        single_event = {}
        single_event["px"] = px
        single_event["py"] = py
        single_event["pz"] = pz
        single_event["E"] = en
        single_event["ch"] = charge
        single_event["pid"] = pid

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
