import json
import numpy as np
import matplotlib.pyplot as plt
Dcm_patients = [str("00" + str(i)) if len(str(i)) == 1 else str("0"+str(i)) for i in range(1, 21)]
Hcm_patients = [str("0"+str(i)) for i in range(21, 41)]
Minf_patients = [str("0"+str(i)) for i in range(41, 61)]
Nor_patients = [str("0"+str(i)) for i in range(61, 81)]
Rv_patients = [str("0"+str(i)) for i in range(81, 100)]
#Rv_patients.append("100")

with open('rayons.json', "r") as file:
    data = json.load(file)

Dcm_x = []
Dcm_y =[]
for id in Dcm_patients:
    Dcm_x+=list(data[id].keys())
    Dcm_y+=list(data[id].values())
sorted_pairs = sorted(zip(Dcm_x, Dcm_y))  # Trie selon Dcm_x
Dcm_x_sorted, Dcm_y_sorted = zip(*sorted(zip(Dcm_x, Dcm_y)))
Dcm_x_sorted_float = [float(x) for x in Dcm_x_sorted]
rounded_x = np.round(Dcm_x_sorted_float, 2)
plt.figure(1)
plt.plot(Dcm_x_sorted_float, Dcm_y_sorted, marker='o')
plt.xticks(Dcm_x_sorted_float, labels=rounded_x, rotation=45)
plt.xlabel("Dcm_x (arrondi à 2 décimales)")
plt.ylabel("Dcm_y")
plt.title("Graphique avec Dcm_x arrondi")
plt.grid(True)

Hcm_x = []
Hcm_y =[]
for id in Hcm_patients:
    Hcm_x+=list(data[id].keys())
    Hcm_y+=list(data[id].values())
sorted_pairs = sorted(zip(Hcm_x, Hcm_y)) 
Hcm_x_sorted, Hcm_y_sorted = zip(*sorted(zip(Hcm_x, Hcm_y)))
Hcm_x_sorted_float = [float(x) for x in Hcm_x_sorted]
rounded_x = np.round(Hcm_x_sorted_float, 2)
plt.plot(Hcm_x_sorted_float, Hcm_y_sorted, marker='o')
plt.xticks(Hcm_x_sorted_float, labels=rounded_x, rotation=45)
plt.xlabel("Hcm_x (arrondi à 2 décimales)")
plt.ylabel("Hcm_y")
plt.title("Graphique avec Hcm_x arrondi")
plt.grid(True)

Minf_x = []
Minf_y =[]
for id in Minf_patients:
    Minf_x+=list(data[id].keys())
    Minf_y+=list(data[id].values())
sorted_pairs = sorted(zip(Minf_x, Minf_y))
Minf_x_sorted, Minf_y_sorted = zip(*sorted(zip(Minf_x, Minf_y)))
Minf_x_sorted_float = [float(x) for x in Minf_x_sorted]
rounded_x = np.round(Minf_x_sorted_float, 2)
plt.plot(Minf_x_sorted_float, Minf_y_sorted, marker='o')
plt.xticks(Minf_x_sorted_float, labels=rounded_x, rotation=45)
plt.xlabel("Minf_x (arrondi à 2 décimales)")
plt.ylabel("Minf_y")
plt.title("Graphique avec Minf_x arrondi")
plt.grid(True)

Nor_x = []
Nor_y =[]
for id in Nor_patients:
    Nor_x+=list(data[id].keys())
    Nor_y+=list(data[id].values())
sorted_pairs = sorted(zip(Nor_x, Nor_y))
Nor_x_sorted, Nor_y_sorted = zip(*sorted(zip(Nor_x, Nor_y)))
Nor_x_sorted_float = [float(x) for x in Nor_x_sorted]
rounded_x = np.round(Nor_x_sorted_float, 2)
plt.plot(Nor_x_sorted_float, Nor_y_sorted, marker='o')
plt.xticks(Nor_x_sorted_float, labels=rounded_x, rotation=45)
plt.xlabel("Nor_x (arrondi à 2 décimales)")
plt.ylabel("Nor_y")
plt.title("Graphique avec Nor_x arrondi")
plt.grid(True)

Rv_x = []
Rv_y =[]
for id in Rv_patients:
    Rv_x+=list(data[id].keys())
    Rv_y+=list(data[id].values())
sorted_pairs = sorted(zip(Rv_x, Rv_y))
Rv_x_sorted, Rv_y_sorted = zip(*sorted(zip(Rv_x, Rv_y)))
Rv_x_sorted_float = [float(x) for x in Rv_x_sorted]
rounded_x = np.round(Rv_x_sorted_float, 2)
plt.plot(Rv_x_sorted_float, Rv_y_sorted, marker='o')
plt.xticks(Rv_x_sorted_float, labels=rounded_x, rotation=45)
plt.xlabel("Rv_x (arrondi à 2 décimales)")
plt.ylabel("Rv_y")
plt.title("Graphique avec Rv_x arrondi")
plt.grid(True)

all_x = Dcm_x + Hcm_x + Minf_x + Nor_x + Rv_x
all_y = Dcm_y + Hcm_y + Minf_y + Nor_y + Rv_y
sorted_pairs = sorted(zip(all_x, all_y))
all_x_sorted, all_y_sorted = zip(*sorted(zip(all_x, all_y)))
all_x_sorted_float = [float(x) for x in all_x_sorted]
rounded_x = np.round(all_x_sorted_float, 2)
plt.figure(2)
plt.plot(all_x_sorted, all_y_sorted, marker='o')
plt.xticks(all_x_sorted_float, labels=rounded_x, rotation=45)

print(all_x_sorted_float)
dico={"x":all_x_sorted_float, "y":all_y_sorted}
with open("rayons_opti.json", "w") as f:
    json.dump(dico, f)
#plt.show()