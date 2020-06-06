import stanza


if __name__ == '__main__':
    nlp = stanza.Pipeline('ru', processors='tokenize,ner', tokenize_pretokenized=True,
                          ner_model_path="../saved_models/ner/ru_pampers_nertagger.pt")
    print(nlp.loaded_processors[1].pipeline.processors["ner"].config)
    print(nlp.loaded_processors[1].pipeline.processors["tokenize"].config)
    doc = nlp("Подгузники-трусики Bella baby Happy")
    print(*[f'entity: {ent.text}\ttype: {ent.type}' for sent in doc.sentences for ent in sent.ents], sep='\n')
