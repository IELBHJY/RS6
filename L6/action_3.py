import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('powerbi/city.csv')
data['province'] = ['Not-HuBei' if value !='湖北' else 'HuBei' for value in data['province']]
confirm_num = dict()
dead_num = dict()
heal_num = dict()
for index in data.index:
    if data.ix[index,'province'] in confirm_num:
        confirm_num[data.ix[index,'province']] +=data.ix[index,'confirm']
    else:
        confirm_num[data.ix[index,'province']] = 0
    if data.ix[index,'province'] in dead_num:
        dead_num[data.ix[index,'province']] +=data.ix[index,'dead']
    else:
        dead_num[data.ix[index,'province']] = 0
    if data.ix[index,'province'] in heal_num:
        heal_num[data.ix[index,'province']] +=data.ix[index,'heal']
    else:
        heal_num[data.ix[index,'province']] = 0

c = sorted(confirm_num.items(), key=lambda d: d[1],reverse=True)
sns.barplot([value[0] for value in c],[value[1] for value in c])
plt.title("Confirm Num")
plt.show()

d = sorted(dead_num.items(), key=lambda d: d[1],reverse=True)
sns.barplot([value[0] for value in d],[value[1] for value in d])
plt.title("Dead Num")
plt.show()

h = sorted(heal_num.items(), key=lambda d: d[1],reverse=True)
sns.barplot([value[0] for value in h],[value[1] for value in h])
plt.title("Heal Num")
plt.show()