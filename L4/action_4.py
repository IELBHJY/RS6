'''
Action1：使用Python模拟下面的PageRank计算过程，求每个节点的影响力（迭代100次）
简化模型 随机模型
Action2：使用TextRank对新闻进行关键词提取，及文章摘要输出
'''
import networkx as nx

'''
基于networkx库实现pagerank
'''
def action_one():
    edges=[['A','B'],['B','C'],['A','F'],['A','D'],['A','E'],
           ['C','E'],['D','A'],['D','C'],['D','E'],
           ['E','C'],['E','B'],['F','D']]
    graph=nx.DiGraph()
    for edge in edges:
        graph.add_edge(edge[0],edge[1])
    print("node num:",graph.number_of_nodes())
    print("edge num:",graph.number_of_edges())
    rank_list1 = nx.pagerank(graph,alpha=1,max_iter=100)
    print("简化模型结果：",rank_list1)
    print(sorted(rank_list1.items(),key= lambda item:item[1],reverse=True))

    rank_list2 = nx.pagerank(graph,alpha=0.85,max_iter=100)
    print("随机模型结果：",rank_list2)
    print(sorted(rank_list2.items(),key= lambda item:item[1],reverse=True))

'''
python实现pagerank计算过程
通过参数d来控制哪种模式d=1为简单模式 d<1 为随机模式
'''
def action_one_pagerank(d):
    import numpy as np
    M = np.array([[0,0,0,1/3,0,0],
                  [1/4,0,0,0,1/2,0],
                  [0,1,0,1/3,1/2,0],
                  [1/4,0,0,0,0,1],
                  [1/4,0,1,1/3,0,0],
                  [1/4,0,0,0,0,0]],dtype=np.float)
    print(M)
    b=np.array([1/6,1/6,1/6,1/6,1/6,1/6],dtype=np.float)
    w = b
    for i in range(100):
        w = (1-d) / M.shape[0] + d * np.dot(M, w)
        print(w)


def action_two():
    import pandas as pd
    from textrank4zh import TextRank4Keyword, TextRank4Sentence
    news = pd.read_table('textrank/news.txt',encoding='GB18030',header=None)
    strings=''
    for index in range(news.shape[0]):
        strings += news.loc[index, 0]
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=strings, lower=True, window=3)
    print('关键词：')
    for item in tr4w.get_keywords(20, word_min_len=2):
        print(item.word, item.weight)

    tr4s = TextRank4Sentence()
    tr4s.analyze(text=strings, lower=True, source='all_filters')
    print('摘要：')
    # 重要性较高的三个句子
    for item in tr4s.get_key_sentences(num=3):
        print(item.weight, item.sentence)

if __name__ == '__main__':
    action_one()
    action_one_pagerank(1)
    action_one_pagerank(0.85)
    action_two()


