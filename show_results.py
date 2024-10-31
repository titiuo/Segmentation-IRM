import json
import matplotlib.pyplot as plt
import numpy as np

with open(f'test_logs.json', 'r') as file:
    data = json.load(file)

all_data = {}
all_errors = []

for k in range(1,101):
    k = str(k)
    if len(k)==1:
        k = '00'+str(k)
    elif len(k)==2:
        k = '0'+str(k)
    else:
        pass
    try:
        new_data = data[str(k)]
        #print(new_data)
        val = float(new_data['Mean dice coefficient:'])
        all_data[k] = [val, new_data['Group']]
    except:
        all_errors.append(k)


import statistics

#mean = statistics.mean(all_data.values())



# Plotting the bar chart
def plot_bar(all_data,type="all"):
    values = list(all_data.values())

    plt.figure(figsize=(10, 5))
    if type=="all":   
        plt.bar(all_data.keys(), [val[0] for val in values], color='blue')
        plt.title('Mean Dice Coefficient for Each Data Index')
    elif type=="NOR":
        filtered_data = {key: val for key, val in all_data.items() if val[1] == 'NOR'}
        values = list(filtered_data.values())
        plt.bar(filtered_data.keys(), [val[0] for val in filtered_data.values()], color='blue')
        plt.title('Mean Dice Coefficient for Each Data Index - NOR')
    elif type=="RV":
        filtered_data = {key: val for key, val in all_data.items() if val[1] == 'RV'}
        values = list(filtered_data.values())
        plt.bar(filtered_data.keys(), [val[0] for val in filtered_data.values()], color='blue')
        plt.title('Mean Dice Coefficient for Each Data Index - RV')
    elif type == "MINF" :
        filtered_data = {key: val for key, val in all_data.items() if val[1] == 'MINF'}
        values = list(filtered_data.values())
        plt.bar(filtered_data.keys(), [val[0] for val in filtered_data.values()], color='blue')
        plt.title('Mean Dice Coefficient for Each Data Index - MINF')
    elif type == "HCM":
        filtered_data = {key: val for key, val in all_data.items() if val[1] == 'HCM'}
        values = list(filtered_data.values())
        plt.bar(filtered_data.keys(), [val[0] for val in filtered_data.values()], color='blue')
        plt.title('Mean Dice Coefficient for Each Data Index - HCM')
    elif type == "DCM":
        filtered_data = {key: val for key, val in all_data.items() if val[1] == 'DCM'}
        values = list(filtered_data.values())
        plt.bar(filtered_data.keys(), [val[0] for val in filtered_data.values()], color='blue')
        plt.title('Mean Dice Coefficient for Each Data Index - DCM')
    plt.xlabel('Data Index')
    plt.ylabel('Mean Dice Coefficient')
    plt.xticks(rotation=90)
    plt.tight_layout()

    mean = np.mean([val[0] for val in values])
    print(f'Mean dice for {type}: ', mean)
    count = 0
    for val in values:
        if val[0] < 0.4:
            count += 1
    print('Number of means<0.4 in values: ', count)
    print('Nb of errors: ', len(all_errors))
    
plot_bar(all_data, type="all")
plot_bar(all_data, type="NOR")
plot_bar(all_data, type="RV")
plot_bar(all_data, type="MINF")
plot_bar(all_data, type="HCM")
plot_bar(all_data, type="DCM")
plt.show()




