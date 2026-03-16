import matplotlib.pyplot as plt

from src.data_loader import filter_discharge_cycles, load_nasa_battery_directory


def main() -> None:
    df = load_nasa_battery_directory("data")
    discharge_df = filter_discharge_cycles(df)
    b0005 = discharge_df.loc[discharge_df["battery_id"] == "B0005"].copy()

    plt.plot(b0005["capacity_ah"].values)
    plt.title("Battery Capacity Fade Over Discharge Cycles (B0005)")
    plt.xlabel("Discharge Cycle Number")
    plt.ylabel("Capacity (Ah)")
    plt.show()


if __name__ == "__main__":
    main()
