from elasticsearch import Elasticsearch


es = Elasticsearch()


def get_es_relevant_entity(q, size=5):
    dsl = {
        'query': {
            'match': {
                'entity': q
            }
        },
        'size': size
    }
    result = es.search(index='huawei', body=dsl)  # 查询
    if result['hits']['total']['value'] == 0:
        return []
    hits = result['hits']['hits'][:size]  # 取前size个结果
    docs = []
    for hit in hits:  # 依次取出查询结果
        entity = hit['_source']['entity'].strip()
        score = hit['_score']
        docs.append({'entity': entity, 'score': score})
    return docs


def get_es_relevant_zuofa(q, size=1):
    dsl = {
        'query': {
            'match': {
                'detail': q
            }
        },
        'size': size
    }
    result = es.search(index='zuofa', body=dsl)  # 查询
    if result['hits']['total']['value'] == 0:
        return []
    hits = result['hits']['hits'][:size]  # 取前size个结果
    docs = []
    for hit in hits:  # 依次取出查询结果
        zuofa = hit['_source']['detail'].strip()
        score = hit['_score']
        docs.append({'zuofa': zuofa, 'score': score})
    return docs


def get_es_relevant_chengyu(q, size=1):
    dsl = {
        'query': {
            'match': {
                'detail': q
            }
        },
        'size': size
    }
    result = es.search(index='chengyu', body=dsl)  # 查询
    if result['hits']['total']['value'] == 0:
        return []
    hits = result['hits']['hits'][:size]  # 取前size个结果
    docs = []
    for hit in hits:  # 依次取出查询结果
        chengyu = hit['_source']['detail'].strip()
        score = hit['_score']
        docs.append({'chengyu': chengyu, 'score': score})
    return docs


if __name__ == "__main__":
    print(get_es_relevant_entity('''安地列斯地区有有什么鱼？''', size=5))
    # print(get_es_relevant_chengyu('''是谁写的子夜四时歌:球格呢？''', size=5))
