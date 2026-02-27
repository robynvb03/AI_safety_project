import transformer_lens
from transformer_lens import HookedTransformer
import joblib

## load the LLM2 model
llm2 = HookedTransformer.from_pretrained("gpt2-small")

## load the svm model
svm_model = joblib.load('model.pkl')

def get_classification(response, llm2, svm_model):
    ## grab the LLM's response and pass it into LLM2. Classify the response based on LLM2's activations.
    ## first pass the response into llm2
    logits, cache = llm2.run_with_cache(llm2.to_tokens(response), stop_at_layer=1)
    activations = cache['blocks.0.hook_mlp_out'][0][-1]

    ## pass the activations into the svm to get a classification
    features = activations.detach().cpu().numpy().reshape(1, -1)
    label = svm_model.predict(features)


    return label[0]

def classify(out_file):
    with open(out_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        scores = []
        for line in lines:
            score = get_classification(line.strip(), llm2, svm_model)
            scores.append(score)
        avg_score = sum(scores) / len(scores) if scores else 0
        """#print scores for debugging
        print(f"Classification scores for each line: {scores}")
        print(f"Average classification score: {avg_score}")"""
        return avg_score    

