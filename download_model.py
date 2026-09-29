from sentence_transformers import SentenceTransformer

print("Saving model to local project folder...", flush=True)
model = SentenceTransformer("all-MiniLM-L6-v2")
model.save("./model_cache")
print("Done, model in ./model_cache", flush=True)