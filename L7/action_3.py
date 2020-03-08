from gensim.models import word2vec

segment_folder = 'word2vec/three_kingdoms/segment'
sentences = word2vec.PathLineSentences(segment_folder)

model = word2vec.Word2Vec(sentences, size=100, window=3, min_count=3)
#model.wv.save_word2vec_format('file1.txt', binary=False)
#model.wv.similarity('刘备', '关羽')
print(model.wv.most_similar(positive=['曹操']))
print(model.wv.most_similar(positive=['曹操','刘备'],negative=['张飞']))
#[('孙权', 0.986218273639679), ('荆州', 0.9801917672157288), ('夫人', 0.9764574766159058), ('周瑜', 0.9756923913955688), ('今反', 0.9745445847511292), ('孔明', 0.9739490747451782), ('已', 0.9734069108963013), ('拜', 0.9730291366577148), ('拜谢', 0.9727320671081543), ('袁绍', 0.9722797870635986)]
#[('今', 0.9847639799118042), ('臣', 0.9846991300582886), ('吾', 0.9833989143371582), ('主公', 0.9833654165267944), ('丞相', 0.9818264842033386), ('某', 0.9800719022750854), ('问', 0.9799109697341919), ('此', 0.9775131940841675), ('告', 0.9753938317298889), ('卿', 0.9734485149383545)]
