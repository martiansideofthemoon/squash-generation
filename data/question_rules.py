import pickle
import spacy
from spacy.tokenizer import Tokenizer
from spacy.matcher import PhraseMatcher


nlp = spacy.load('en_core_web_sm', disable=['ner'])


def specific_concept_completion(question):
    first = question[0]
    return (
        first.text.lower() in ['when', 'where', 'who'] or
        question[0:2].text.lower() == 'how many' or
        question[0:2].text.lower() == 'how long'
    )


def judgemental(question):
    matcher = PhraseMatcher(nlp.vocab)
    matcher.add('JUDGEMENT', None, nlp('your'), nlp('you'), nlp('Your'), nlp('You'))
    return len(matcher(question)) > 0


def instrumental(question):
    ancestors = [x for x in question[0].ancestors]
    return question[0].text.lower() == 'how' and ancestors[0].pos_ == 'VERB'


def causal(question):
    return (
        question[0].text.lower() == 'why' or
        question[0:3].text.lower() == 'what happened after' or
        question[0:3].text.lower() == 'what happened before' or
        question[0:3].text.lower() == 'what led to' or
        question[0:4].text.lower() == 'what was the reason' or
        question[0:2].text.lower() == 'what causes' or
        question[0:4].text.lower() == 'what was the purpose' or
        question[0:4].text.lower() == 'what was their purpose' or
        question[0:4].text.lower() == 'what was the cause' or
        question[0:3].text.lower() == 'what happened next' or
        question[0:3].text.lower() == 'what happens next'
    )


def verification(question):
    first = question[0]
    return first.pos_ == 'VERB'


def general_concept_completion(question):
    if question[0].text.lower() == 'who' and \
       question[1].text.lower() in ['is', 'was'] and \
       (question[2].pos_ == 'PROPN' or question[2].text.lower() in ['he', 'she']) and \
       (question[-2].pos_ == 'PROPN' or question[-2].text.lower() in ['he', 'she']):
        return True
    elif question[0:3].text.lower() == 'what happened in':
        return True
    elif question[0:3].text.lower() == 'what happened during':
        return True
    else:
        return False


def labeller(question):
    question_nlp = nlp(question)
    try:
        if verification(question_nlp):
            return 'verification'
        elif judgemental(question_nlp):
            return 'judgemental'
        elif instrumental(question_nlp):
            return 'instrumental'
        elif general_concept_completion(question_nlp):
            return 'general_concept_completion'
        elif causal(question_nlp):
            return 'causal'
        elif specific_concept_completion(question_nlp):
            return 'specific_concept_completion'
        else:
            return 'none'
    except:
        if question == '':
            return 'none'
        elif len(question.split()) < 4:
            return 'none'
        else:
            import pdb; pdb.set_trace()
    return 'none'
