import spacy
import random
import typer
from pathlib import Path
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
from scripts.rel_pipe import make_relation_extractor, score_relations
from scripts.rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors
import pandas as pd

from os import listdir
from os.path import isfile, join
nlp = spacy.load('en_core_web_trf')
docs = []

mypath = "../processed_text/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
csvs = dict()
for filename in onlyfiles:
    doc = []
    with open(mypath + filename, "r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            doc.append(line)
    docs += doc
for doc in nlp.pipe(docs, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]):
    print(doc)
    print(f"spans: {[(e.start, e.text, e.label_) for e in doc.ents]}")
    for sent in [doc]:
      for e in sent.ents:
        for b in sent.ents:
            print(e, b)
    print("-" * 100)
    break


# text=['''2+ years of non-internship professional software development experience
# Programming experience with at least one modern language such as Java, C++, or C# including object-oriented design.
# 1+ years of experience contributing to the architecture and design (architecture, design patterns, reliability and scaling) of new and current systems.
# Bachelor / MS Degree in Computer Science. Preferably a PhD in data science.
# 8+ years of professional experience in software development. 2+ years of experience in project management.
# Experience in mentoring junior software engineers to improve their skills, and make them more effective, product software engineers.
# Experience in data structures, algorithm design, complexity analysis, object-oriented design.
# 3+ years experience in at least one modern programming language such as Java, Scala, Python, C++, C#
# Experience in professional software engineering practices & best practices for the full software development life cycle, including coding standards, code reviews, source control management, build processes, testing, and operations
# Experience in communicating with users, other technical teams, and management to collect requirements, describe software product features, and technical designs.
# Experience with building complex software systems that have been successfully delivered to customers
# Proven ability to take a project from scoping requirements through actual launch of the project, with experience in the subsequent operation of the system in production''']
# docs+=text
#
# text = ['''Marylanders deserve less rhetoric and more results As your next Governor I ll make sure that state government once again gets the basics right for our working families so that we can restore faith in its ability to achieve bold policy ideas Learn more at
# Featuring King Von FBG Duck Lil Durk and more Link to full documentary in bio
# The House could vote on the Build Back Better package as early as this week This is what s at stake Send a message to Rep Golden and tell him to pass this historic investment in our communities No half measures
# What Employers Need to Know About Vaccination Mandates Federal agencies issued several mandates recently requiring some employers to develop and put in place mandatory vaccination programs by early December At the same time the State of Florida has joined lawsuits seeking to invalidate such mandates and Florida s legislature is considering several bills that if passed would conflict with the federal mandates A panel from Williams Parker s Labor and Employment and Healthcare teams will present a webinar on November at pm providing a bird s eye view for employers on what this could mean Register here
# Thought leaders in higher education gathered on Nov as part of The Hunt Institute virtual panel to discuss how post secondary institutions are thinking about remedying college readiness
# Do you have friends or family who are pregnant or breastfeeding Tag them below to share the key facts
# Accounting for a substantial proportion of the agricultural labour force rural women play a crucial role in agriculture food security and nutrition yet they face daily struggles They are less likely to have access to quality health services essential medicines and other basic services Furthermore a lot of rural women suffer from isolation as well as the spread of misinformation and a lack of access to critical technologies that might improve their work and personal life Peace Parks Foundation Women Rural PeaceParksFoundation Communitydevelopment Healthcare Africa Pandemic EarthshotPrize
# Those of us with deep roots in the community remember well the problems we had with anti social behaviour in the Morrisons car park area when the store first opened This time it is not the boy racer drivers that are the problem but people using the skate park after dark to create their own sculptures from shopping carts We don t want this to be a tipping point for the old behaviours to return We know Morrisions have deactivated all the tokens on the carts as a Covid reduction measure Understandable Less understandable is the car park left unlocked after trading hours and no real attempt to secure the shopping carts to prevent this type of behaviour Local Labour activists such as the one who took these photos are in the process of contacting Morrisons to see what can be done to address what could develop into more serious issues Local Labour Local Activism
# ''']
