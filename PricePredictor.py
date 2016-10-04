import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from dateutil import parser
import csv

with open ('pp-2015-part1.csv', 'rb') as csvfile:
    #pp2015reader=csv.DictReader(csvfile,fieldnames=["TUI","Price","DateOfTransfer","Postcode","PropertyType","Old/New", "Duration","PAON", "SAON","Street","Locality","Town/City","District","County","PPDCategoryType"],  delimiter=",", quotechar="\"")
    pp2015reader=list(csv.reader(csvfile,delimiter=",",quotechar="\""))

Price=[i[1] for i in pp2015reader]
DateOfTransfer=[parser.parse(i[2]) for i in pp2015reader[0:10000]]

#print DateOfTransfer[0:1000]
plt.plot(DateOfTransfer[0:10], Price[0:10], marker="o", linestyle=" ")
plt.show()
