import random
import typer
from pathlib import Path
import spacy
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
from scripts.rel_pipe import make_relation_extractor, score_relations
from scripts.rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors
# We load the relation extraction (REL) model
nlp2 = spacy.load("en_core_web_trf")
text = ['''2+ years of non-internship professional software development experience
Programming experience with at least one modern language such as Java, C++, or C# including object-oriented design.
1+ years of experience contributing to the architecture and design (architecture, design patterns, reliability and scaling) of new and current systems.
Bachelor / MS Degree in Computer Science. Preferably a PhD in data science.
8+ years of professional experience in software development. 2+ years of experience in project management.
Experience in mentoring junior software engineers to improve their skills, and make them more effective, product software engineers.
Experience in data structures, algorithm design, complexity analysis, object-oriented design.
3+ years experience in at least one modern programming language such as Java, Scala, Python, C++, C#
Experience in professional software engineering practices & best practices for the full software development life cycle, including coding standards, code reviews, source control management, build processes, testing, and operations
Experience in communicating with users, other technical teams, and management to collect requirements, describe software product features, and technical designs.
Experience with building complex software systems that have been successfully delivered to customers
Proven ability to take a project from scoping requirements through actual launch of the project, with experience in the subsequent operation of the system in production''']
# We take the entities generated from the NER pipeline and input them to the REL pipeline
for doc in nlp2.pipe(text, disable=["tagger"]):
    #for name, proc in nlp2.pipe(doc):
    #          doc = proc(doc)
# Here, we split the paragraph into sentences and apply the relation extraction for each pair of entities found in each sentence.
for value, rel_dict in doc._.rel.items():
    print(value, rel_dict)
    for sent in doc.sents:
      for e in sent.ents:
        for b in sent.ents:
          if e.start == value[0] and b.start == value[1]:
            if rel_dict['EXPERIENCE_IN'] >=0.9 :
              print(f" entities: {e.text, b.text} --> predicted relation: {rel_dict}")
