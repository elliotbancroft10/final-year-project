import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import seaborn as sns
import matplotlib as mpl
import numpy as np
import pandas as pd
import csv

x = []
y1 = []
y2 = []
xdetail = []
y1detail = []
y2detail = []

with open("C:/Users/Elliot/Documents/Y4 Uni/stresstrain.csv", 'r') as ugn:
    plots = csv.reader(ugn, delimiter = ',')
    for column in plots:
        x.append(column[0])
        y1.append(float(column[6]))
        y2.append(float(column[11]))
        #xdetail.append(column[0])
        #y1detail.append(float(column[6]))
        #y2detail.append(float(column[10]))
'''
with open("C:/Users/Elliot/Documents/Y4 Uni/RNstrainTrain.csv", 'r') as ugn:
    plots = csv.reader(ugn, delimiter = ',')
    for column in plots:
       #y2.append(float(column[3])) 
'''
del xdetail[0:83:1]        
del y1detail[0:83:1]
del y2detail[0:83:1]    

#print(xdetail)
#print(y1detail)
#print(y2detail)
fig, ax = plt.subplots(tight_layout = True, figsize = (11,8))

sns.set_style()

ax.set_xlabel('Epoch Number', fontsize = 20)
ax.set_ylabel('Error', fontsize = 20)
sns.set_theme(style="ticks")
plt.plot(x, y1, color = 'orange', label = 'Training Error')
plt.plot(x, y2, color = 'blue', label = 'Validation Error')
plt.xticks(range(-1,60, 5))
plt.grid(axis='both')
plt.legend(fontsize = 20)
plt.tick_params(axis = 'both',labelsize = 20)
plt.yscale('log')
#plt.ylim(7.2, 9.88)




#sns.relplot(data=, x="Date", y="Price", kind="line"
#plt.draw()
plt.show()