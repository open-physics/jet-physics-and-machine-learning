import numpy as np
from chromo.constants import TeV
from chromo.kinematics import CenterOfMass
from chromo.models import Pythia8
import pickle
from particle import Particle


def run_conditions():
    evt_kin = CenterOfMass(7 * TeV, particle1="proton", particle2="proton")
    return evt_kin


def particle_info(event, index):
    pid = event.pid[index]
    name = Particle.from_pdgid(pid).name

    child_indices = event.children[index] - 1
    child_pids = [event.pid[x] for x in child_indices]
    child_names = Particle.from_pdgid(child_pids[0]).name, Particle.from_pdgid(child_pids[1]).name

    return {
            'particle_info': [name, pid, index],
            'children_info': [child_names, child_pids, child_indices]
            }


def main():
    evt_kin = run_conditions()
    config = ["HardQCD:gg2ccbar = on"]
    pythia = Pythia8(evt_kin, seed=45, config=config)
    # d0 = 421

    all_events = []
    single_event = []
    pt_avg_list = []
    mult = []
    mult_plus = []
    mult_minus = []
    mult_zero = []
    for event in pythia(4):
        # sanity check: skip if event has no tracks or has no D0 meson
        # if np.abs(ev.pid)==421:
        if len(event.pt) <= 0: #or d0 not in event.pid:
            continue
        event_final = event.final_state()
        tr_eta = event_final.eta
        tr_pid = event_final.pid
        tr_phi = event_final.phi
        tr_pt = event_final.pt
        # tr_mass = event_final.m
        tr_charge = event_final.charge
        # tr_pt = ev.pt[np.abs(ev.pid) == 211]

        # sanity check: does avg_pt make sense?
        avg_pt = np.mean(tr_pt)
        pt_avg_list.append(avg_pt)

        # keep a record of event numbers
        # evt_list.append(event_final.nevent)

        
        # multiplicity
        plus = list(tr_charge).count(1)
        minus = list(tr_charge).count(-1)
        zero = list(tr_charge).count(0)
        mult_plus.append(plus)
        mult_minus.append(minus)
        mult_zero.append(zero)
        # multiplicity.append(len(event))
        mult.append(plus + minus + zero)

        # create dictionary to store event properties event-by-event 
        single_event = {}
        single_event["avg_pt"]  = avg_pt
        single_event["tr_eta"] = tr_eta
        single_event["tr_phi"]  = tr_phi
        single_event["tr_pid"]  = tr_pid

        # single_event = [event_final.nevent, avg_pt, tr_phi, tr_pid]
        all_events.append(single_event)
    
        breakpoint()

    # store/write event properties using pickle
    with open('array_store.pkl', 'wb') as file:
        pickle.dump(all_events, file)

    # retrieve event properties using pickle
    with open('array_store.pkl', 'rb') as file:
        loaded_array = pickle.load(file)
        
    print("event_info", loaded_array)
        


if __name__ == "__main__":
    main()
