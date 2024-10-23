import json
import matplotlib.pyplot as plt

with open(f'logs.json', 'r') as file:
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
        print(new_data)
        val = new_data['Mean dice coefficient:']
        all_data[k] = val
    except:
        all_errors.append(k)


import statistics

#mean = statistics.mean(all_data.values())

#print('Mean dice: mean')

print('Nb of errors: ', len(all_errors))

# Plotting the bar chart
plt.figure(figsize=(10, 5))
plt.bar(all_data.keys(), all_data.values(), color='blue')
plt.xlabel('Data Index')
plt.ylabel('Mean Dice Coefficient')
plt.title('Mean Dice Coefficient for Each Data Index')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()




