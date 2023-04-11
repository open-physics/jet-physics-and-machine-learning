import numpy as np
import chromo
import matplotlib.pyplot as plt

def main():

    kinematics = chromo.kinematics.CenterOfMass(13 * chromo.constants.TeV, "proton", "proton")
    generator = chromo.models.Sibyll23d(kinematics)

    nevent = 0
    avg_pt = 0
    pt_list = []
    evt_list = []

    for event in generator(5):
        event = event.final_state_charged()
        pt = event.pt[np.abs(event.pid) == 211]
        if len(pt) > 0:
            nevent += 1
            avg_pt += np.mean(pt)

        # avg_pt = avg_pt / nevent
        average_pt = avg_pt
        pt_list.append(average_pt)
        evt_list.append(nevent)
        
        breakpoint()

        print(nevent, average_pt)
    plt.plot(pt_list, evt_list) #color="blue", linewidth=7.0)
    plt.xlabel("average pt")
    plt.ylabel("# events")
    plt.title("Mean pt distribution")
    # fig.savefig("mean_pt.png")
    plt.show()


if __name__ == "__main__":
    main()


