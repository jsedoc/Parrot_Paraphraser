import warnings
warnings.filterwarnings("ignore")
from parrot import Parrot
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")
phrases = [["Can you recommed some upscale restaurants in Rome?", "This is great."]]
for phrase in phrases:
  print("-"*100)
  print(phrase)
  print("-"*100)
  para_phrases = parrot.augment(input_phrase_lst=phrase,
                                use_gpu=True,
                                diversity_ranker="levenshtein",
                                do_diverse=False, 
                                max_return_phrases = 10, 
                                max_length=32, 
                                adequacy_threshold = 0.99, 
                                fluency_threshold = 0.90)
  print(para_phrases)
  for orig, paraphrase in para_phrases.items():                                  
      print(orig)
      print(paraphrase)                               

 
  
