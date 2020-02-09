'''
Action1：针对MarketBasket数据集进行购物篮分析（频繁项集及关联规则挖掘）
'''
import pandas as pd
from efficient_apriori import apriori


def rule_one():
    data = pd.read_csv('./MarketBasket/Market_Basket_Optimisation.csv', header=None)
    transactions=[]
    data = data.fillna(0)
    for index in data.index:
        temp=set()
        for column in data.columns:
            if data.loc[index,column] == 0:
                continue
            else:
                temp.add(data.loc[index,column])
        transactions.append(temp)
    items, rules = apriori(transactions,min_support=0.02,min_confidence=0.2)
    print(items)
    print(rules)


if __name__ == '__main__':
    rule_one()






