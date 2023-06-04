import pandas as pd
import os
import matplotlib.pyplot as plt

dir = os.listdir("logs")
dir.sort()
latestfile = "logs/" + dir[-1]

data = pd.read_csv(latestfile,
                   names=["Time",
                          "Updates",
                          "Frames",
                          "Loss",
                          "Bert Loss",
                          "Reward",
                          "Epsilon"]
                   )

plt.subplot(6, 1, 1)
plt.title("Time (Sec)")
plt.plot(data["Time"])

plt.subplot(6, 1, 2)
plt.title("Updates")
plt.plot(data["Updates"])

plt.subplot(6, 1, 3)
plt.title("Frames")
plt.plot(data["Frames"])

plt.subplot(6, 1, 4)
plt.title("Loss (Log scale)")
plt.yscale("log")
plt.plot(data["Loss"])

plt.subplot(6, 1, 5)
plt.title("Bert Loss (Log scale)")
plt.yscale("log")
plt.plot(data["Bert Loss"])

plt.subplot(6, 1, 6)
plt.title("Normalized Reward")
plt.plot(data["Reward"])
plt.plot(data["Reward"].rolling(200).mean())

plt.show()
