import matplotlib.pyplot as plt
import numpy as np
from chromo.constants import TeV
from chromo.kinematics import CenterOfMass
from chromo.models import Pythia8
from particle import Particle  # Particle.from_pdgid(211)


def run_conditions():
    evt_kin = CenterOfMass(7 * TeV, particle1="proton", particle2="proton")
    return evt_kin


def pythia_config(which_process: str) -> str:
    config = {
        "pythia_mb": ["SoftQCD:inelastic = on"],
        "pythia_jets": ["HardQCD:all = on"],
        "pythia_charm": [
            "HardQCD:gg2ccbar = on",
            "HardQCD:qqbar2ccbar = on",
            "ParticleData:mcRun = 1.5",
        ],
        "pythia_beauty": [
            "HardQCD:gg2bbbar = on",
            "HardQCD:qqbar2bbbar = on",
            "ParticleData:mcRun = 4.75",
        ],
    }
    return config.get(which_process, config.get("pythia_mb"))


def status_meaning(status: int):
    statuses = {
        1: "is a stable final-state particle",
        2: "decays further",
        4: "is an intermediate-state particle",
    }
    return statuses.get(status)


def get_particle_basics(event, i):
    print(f"This is event #{event.nevent}.")
    print(f"The particle is {Particle.from_pdgid(event.pid[i])}.")
    print(f"It {status_meaning(event.status[i])}.")
    print("\n")


def get_parents(event, i):
    parent_indices = event.parents[i]
    parent_0_pid = event.pid[parent_indices[0]]
    parent_1_pid = event.pid[parent_indices[1]]
    print(
        f"Its parents are {parent_indices}: {Particle.from_pdgid(parent_0_pid)} and {Particle.from_pdgid(parent_1_pid)}."
    )
    if parent_indices[0] > 0 and parent_indices[1] == 0:
        print(
            f"Normal case: here it is meaningful to speak of one single parent to several products, in a shower or decay. "
            f"So, just {Particle.from_pdgid(parent_0_pid)}."
        )
        get_particle_basics(event, parent_indices[0])
        get_parents(event, parent_indices[0])
    if parent_indices[0] < parent_indices[1] and parent_indices[0] > 0:
        if 81 <= abs(event.status[i]) <= 86:
            print(
                "primary hadrons produced from the "
                "fragmentation of a string spanning the range from mother1 to mother2, so that all partons "
                "in this range should be considered mothers; and analogously"
            )
        if 101 <= abs(event.status[i]) <= 106:
            print("the formation of R-hadrons")
        else:
            print(
                r"particles with two truly different mothers, in particular the particles emerging from a hard $2 \rightarrow n$ interaction."
            )
        # get_particle_basics(event, i)
        # get_parents(event, parent_indices[0])
        # get_parents(event, parent_indices[1])

    if parent_indices[0] == parent_indices[1] > 0:
        print(
            "the particle is a ``carbon copy'' of its mother, but with changed momentum as a ``recoil'' effect, e.g. in a shower"
        )
        return parent_indices

    return parent_indices


def get_children(event, i):
    child_indices = event.children[i]
    child_0_pid = event.pid[child_indices[0]]
    child_1_pid = event.pid[child_indices[1]]
    print(
        f"It decays into {child_indices}: {Particle.from_pdgid(child_0_pid)} and {Particle.from_pdgid(child_1_pid)}."
    )
    return child_indices


def main():
    evt_kin = run_conditions()
    config = pythia_config("pythia_charm")
    pythia = Pythia8(evt_kin, seed=1, config=config)
    num_events = 10
    for event in pythia(num_events):
        x = np.cos(event.phi)
        y = np.sin(event.phi)
        plt.scatter(x, y)
        plt.draw()
        plt.savefig("phi.png")

        for i in range(len(event)):
            if abs(event.pid[i]) == 421:
                get_particle_basics(event, i)
                if event.status[i] == 2:
                    child_indices = get_children(event, i)
                    parent_indices = get_parents(event, i)
                    for index in child_indices:
                        print(f"Child status is {event.status[index]}.")
                    for index in parent_indices:
                        print(f"Parent status is {event.status[index]}.")


if __name__ == "__main__":
    main()
