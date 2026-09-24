## Currently there's only a single target for the probes/supervised training. But could add more and include a dictionary so it can be selected with an argument...

LABEL_GROUPS = ("particle_truth", "particle_visible")

MULTIPLICITY_TARGETS = {
    'nproton':   {'weight': 1.0, 'cap': 3},
    'npipm':     {'weight': 1.0, 'cap': 2},
    'npi0':      {'weight': 1.0, 'cap': 2},
    'nem':       {'weight': 1.0, 'cap': 2},
    'ncluster':  {'weight': 1.0, 'cap': 3},
    'nlambda0':  {'weight': 5.0, 'cap': 1},  # upweight rare events
    'nkapm':     {'weight': 5.0, 'cap': 1},  # upweight rare events    
    'nka0':      {'weight': 5.0, 'cap': 1},  # upweight rare events    
    'ncharged':  {'weight': 1.0, 'cap': 5},
}


for cfg in MULTIPLICITY_TARGETS.values():
    cfg["n_classes"] = cfg["cap"] + 1

def label_clamp(targets):
    return {name: cfg["cap"] for name, cfg in targets.items()}
