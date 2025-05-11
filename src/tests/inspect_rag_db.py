from src.rag_db import get_vector_store

if __name__ == '__main__':
    vector_store = get_vector_store('faiss', embedding_dim=1536)
    all_records = vector_store.get_all()
    print(f"Total records: {len(all_records)}")
    found = False
    for sample in all_records:
        features = sample['features']
        if all(k in features for k in ['open_price', 'close_price', 'high_price', 'low_price']):
            print('--- Sample Record with Price Features ---')
            print("Week:", sample['week_start'])
            print("Ticker:", sample['ticker'])
            print("Features:", sample['features'])
            print("Texts:", sample['texts'])
            print("Outcome:", sample['outcome'])
            for k in ['open_price', 'close_price', 'high_price', 'low_price']:
                print(f"{k}: {features[k]}")
            print("Num embeddings:", len(sample['embeddings']))
            print("Embedding[0] (first 5 values):", sample['embeddings'][0][:5] if sample['embeddings'] else None)
            found = True
            break
    if not found:
        print("No record with all price features found.") 