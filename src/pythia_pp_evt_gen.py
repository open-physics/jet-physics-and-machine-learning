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
        "children_info": {
            "child_names": child_names,
            "child_pids": child_pids,
            "child_indices": child_indices,
        },
    }


def replace_children_with_parents(event):
    """
    We need charm hadrons, e.g. D0 meson in the particle container
    of an event before we cluster the particles into jets.
    So, we would like to replace the children of the charm hadron
    with the hadron themselves i.e., parent particle.
    """
    # 1. Find charm hadron, i.e. 421 or D0 meson and its index.
    # 2. Find its children, their indices, and their status values.
    # 3. If status == 2, find their children, indices, and status values.
    # 4. Repreat step 3, until status is 1.

    # Replace D0 children with D0
    D0meson = 421
    if D0meson not in event.pid:
        return
    for i in range(len(event)):
        # Step 1
        if event.pid[i] == D0meson:
            # Steps 2, 3, 4
            final_generation = find_final_generation(event, i)
            for particle_index in final_generation:
                event.status[particle_index] = -1
            event.status[i] = 1


def find_final_generation(event, i, final_generation=None):
    if final_generation is None:
        final_generation = []
    # Step 2
    child_indices = particle_info(event, i)["children_info"]["child_indices"]
    if child_indices[0] == child_indices[1]:
        child_indices = [child_indices[0]]
    for index in child_indices:
        # Step 3
        if event.status[index] == 2:
            # Step 4
            find_final_generation(event, index, final_generation)
        elif event.status[index] == 1:
            final_generation.append(index)
    return final_generation


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

        replace_children_with_parents(event)

        event_final = event[event.status == 1]
        pid = event_final.pid
        eta = np.array(event_final.eta)
        phi = np.array(event_final.phi)
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
        single_event["eta"] = eta
        single_event["phi"] = phi
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
