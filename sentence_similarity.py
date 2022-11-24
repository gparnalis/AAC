from sentence_transformers import SentenceTransformer

model = SentenceTransformer('{MODEL_NAME}')

sentences1 = ['Το κινητό έπεσε και έσπασε.',
              'Το κινητό έπεσε και έσπασε.',
              'Το κινητό έπεσε και έσπασε.']

sentences2 = ["H πτώση κατέστρεψε τη συσκευή.",
              "Το αυτοκίνητο έσπασε στα δυο.",
              "Ο υπουργός έπεσε και έσπασε το πόδι του."]

embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

# Compute cosine-similarities (clone repo for util functions)
from sentence_transformers import util

cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

# Output the pairs with their score
for i in range(len(sentences1)):
    print("{} 		 {} 		 Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))

# Outputs:
# Το κινητό έπεσε και έσπασε. 		 H πτώση κατέστρεψε τη συσκευή. 		 Score: 0.6741
# Το κινητό έπεσε και έσπασε. 		 Το αυτοκίνητο έσπασε στα δυο. 		 Score: 0.5067
# Το κινητό έπεσε και έσπασε. 		 Ο υπουργός έπεσε και έσπασε το πόδι του. 		 Score: 0.4548
