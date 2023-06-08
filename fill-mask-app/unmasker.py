#!/usr/bin/env python3

from transformers import pipeline

def unmasker_fun(input_sentence: str) -> list:
    '''
    Write up here
    
    '''

    model_id = "bert-large-uncased-whole-word-masking"
    unmasker = pipeline('fill-mask', model='distilroberta-base')
    return unmasker(input_sentence, top_k = 5)


def main():
    input_sentence = input('\nPlease input a sentance with a word to be masked with the format "<mask>":\n(Example: "The <mask> barked at me")\n')
    print('\n')
    print(*unmasker_fun(input_sentence), sep="\n")
    print('\n')

if __name__ == "__main__":
    main()