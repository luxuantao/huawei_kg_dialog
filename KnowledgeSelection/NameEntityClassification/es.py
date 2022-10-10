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


if __name__ == "__main__":
    print(get_es_relevant_entity('''是的，这是一部在韩国拍摄的影片，记不住是哪年的了。''', size=5))
