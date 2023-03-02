import json
import numpy as np
import matplotlib.pyplot as plt

with open("../data/history.json", "r") as file:
    history = json.load(file)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 10))
points = len(history["val_value_acc"])
print(points)
print(history["val_value_acc"][-1])
print(history["val_policy_acc"][-1])
print(history["val_value_loss"][-1])
print(history["val_policy_loss"][-1])
x = np.linspace(0, 1024 * 1000 * points, num=points, endpoint=False)

ax[0].yaxis.grid()
ax[0].set_ylabel("Value Loss")
ax[0].set_xlabel("Number of Training Samples")
ax[0].plot(x, history["train_value_loss"], label='Training Loss')
ax[0].plot(x, history["val_value_loss"], label='Validation Loss')
ax[0].legend()


ax[1].yaxis.grid()
ax[1].set_ylabel("Value Accuracy")
ax[1].set_xlabel("Number of Training Samples")
ax[1].plot(x, history["train_value_acc"], label='Training Accuracy')
ax[1].plot(x, history["val_value_acc"], label='Validation Accuracy')
ax[1].legend()

'''ax[0].yaxis.grid()
ax[0].set_ylabel("Policy Loss")
ax[0].set_xlabel("Number of Training Samples")
ax[0].plot(x, history["train_policy_loss"], label='Training Loss')
ax[0].plot(x, history["val_policy_loss"], label='Validation Loss')
ax[0].legend()

ax[1].yaxis.grid()
ax[1].set_ylabel("Policy Accuracy")
ax[1].set_xlabel("Number of Training Samples")
ax[1].plot(x, history["train_policy_acc"], label='Training Accuracy')
ax[1].plot(x, history["val_policy_acc"], label='Validation Accuracy')
ax[1].legend()'''

#plt.show()
plt.savefig("value.png",bbox_inches='tight')
