"""Six scalar vocabularies; native relations and state machines live in world3_structure.
Values are shared by relevant and distracting facts.
"""
DOMAINS = {
    "software": dict(entity="software release", team="maintainers", location="development workspace",
        places=["build room", "review booth", "integration lab", "release desk"],
        skill="programming language", skills=["Python", "Rust", "Go", "Java"],
        requirement="required language", availability="on-call coverage", available="on duty", unavailable="off duty",
        clearance="repository permission", cleared="write access", uncleared="read access",
        context_key="release track", routine="stable", emergency="hotfix",
        attributes={"test suite":["unit", "integration", "acceptance", "regression"],
            "build runner":["Linux", "Windows", "macOS", "container"],
            "review policy":["pair review", "maintainer review", "security review", "team review"],
            "artifact store":["registry", "mirror", "archive", "package index"],
            "deployment region":["east", "west", "central", "south"],
            "release channel":["canary", "beta", "stable", "nightly"],
            "issue queue":["bugs", "features", "maintenance", "security"],
            "documentation format":["Markdown", "HTML", "PDF", "plain text"]}),
    "hospital": dict(entity="hospital roster", team="ward staff", location="assigned station",
        places=["north ward", "south ward", "intake desk", "staff room"],
        skill="clinical qualification", skills=["triage", "imaging", "phlebotomy", "critical care"],
        requirement="required qualification", availability="shift coverage", available="scheduled", unavailable="off shift",
        clearance="ward authorization", cleared="authorized", uncleared="pending",
        context_key="shift type", routine="day", emergency="night",
        attributes={"shift start":["06:00", "08:00", "14:00", "22:00"],
            "ward allocation":["medical", "surgical", "pediatric", "recovery"],
            "handover format":["bedside", "written", "telephone", "team briefing"],
            "break room":["lounge A", "lounge B", "lounge C", "lounge D"],
            "rotation length":["four hours", "six hours", "eight hours", "ten hours"],
            "pager channel":["alpha", "bravo", "charlie", "delta"],
            "rounds slot":["early", "midday", "afternoon", "evening"],
            "roster format":["wall chart", "spreadsheet", "calendar", "printed sheet"]}),
    "travel": dict(entity="family itinerary", team="travel party", location="meeting location",
        places=["hotel lobby", "train platform", "museum entrance", "ferry terminal"],
        skill="travel skill", skills=["navigation", "translation", "driving", "first aid"],
        requirement="required travel skill", availability="trip availability", available="joining", unavailable="staying home",
        clearance="booking permission", cleared="confirmed", uncleared="waitlisted",
        context_key="weather", routine="dry", emergency="rainy",
        attributes={"transport":["train", "bus", "ferry", "rental car"],
            "lodging":["apartment", "hotel", "guesthouse", "cabin"],
            "meal preference":["vegetarian", "seafood", "noodles", "sandwiches"],
            "activity":["museum", "hiking", "beach", "walking tour"],
            "luggage":["backpack", "carry-on", "duffel", "suitcase"],
            "departure slot":["dawn", "morning", "afternoon", "evening"],
            "ticket format":["mobile", "paper", "smart card", "voucher"],
            "room arrangement":["single", "double", "family", "connecting"]}),
    "school": dict(entity="school club event", team="club committee", location="activity room",
        places=["music room", "art room", "assembly hall", "sports field"],
        skill="club skill", skills=["stage lighting", "public speaking", "first aid", "audio mixing"],
        requirement="required club skill", availability="event availability", available="attending", unavailable="absent",
        clearance="faculty permission", cleared="signed", uncleared="unsigned",
        context_key="event setting", routine="indoors", emergency="outdoors",
        attributes={"meeting day":["Monday", "Tuesday", "Thursday", "Friday"],
            "activity format":["rehearsal", "discussion", "demonstration", "competition"],
            "refreshment":["juice", "water", "fruit", "biscuits"],
            "publicity":["poster", "newsletter", "assembly", "class notice"],
            "equipment":["projector", "microphone", "speakers", "whiteboard"],
            "meeting time":["lunch", "after school", "early morning", "study period"],
            "entry policy":["members", "guests", "families", "all students"],
            "record format":["minutes", "photo journal", "audio notes", "bulletin"]}),
    "restaurant": dict(entity="restaurant service", team="service crew", location="service station",
        places=["hot line", "cold kitchen", "bar counter", "front desk"],
        skill="service skill", skills=["grilling", "pastry", "barista", "table service"],
        requirement="required service skill", availability="service availability", available="rostered", unavailable="on leave",
        clearance="kitchen certification", cleared="certified", uncleared="in training",
        context_key="service period", routine="lunch", emergency="dinner",
        attributes={"menu":["seasonal", "tasting", "a la carte", "set menu"],
            "table layout":["banquet", "pairs", "communal", "terrace"],
            "reservation method":["telephone", "website", "walk-in", "concierge"],
            "supplier":["local farm", "wholesaler", "market", "cooperative"],
            "special dish":["risotto", "stew", "salad", "roast"],
            "uniform":["apron", "waistcoat", "chef coat", "polo shirt"],
            "closing task":["inventory", "sanitation", "cash count", "linen check"],
            "service style":["plated", "buffet", "family style", "counter service"]}),
    "laboratory": dict(entity="research experiment", team="lab group", location="bench location",
        places=["wet lab", "dry lab", "microscopy room", "sample archive"],
        skill="laboratory technique", skills=["PCR", "microscopy", "chromatography", "spectrometry"],
        requirement="required technique", availability="experiment availability", available="assigned", unavailable="away",
        clearance="instrument authorization", cleared="trained", uncleared="supervised only",
        context_key="sample class", routine="reference", emergency="field",
        attributes={"assay":["fluorescence", "absorbance", "luminescence", "colorimetric"],
            "sample storage":["ambient", "refrigerated", "frozen", "cryogenic"],
            "buffer":["phosphate", "acetate", "citrate", "Tris"],
            "replicate plan":["duplicate", "triplicate", "quadruplicate", "single"],
            "analysis method":["regression", "ANOVA", "bootstrap", "rank test"],
            "notebook format":["paper", "electronic", "spreadsheet", "database"],
            "calibration":["daily", "weekly", "per batch", "per sample"],
            "label format":["barcode", "QR code", "printed text", "handwritten"]}),
}


def vocabulary(ontology, attr):
    if attr in (ontology["skill"], ontology["requirement"]):
        return ontology["skills"]
    if attr == ontology["location"]:
        return ontology["places"]
    if attr == ontology["availability"]:
        return [ontology["available"], ontology["unavailable"]]
    if attr.startswith(ontology["clearance"]):
        return [ontology["cleared"], ontology["uncleared"]]
    if attr == "coordinator":
        # Identity-valued relations use the very same people in both classes.
        return ["Ari", "Bela", "Cleo", "Davi", "Esme", "Farah"]
    return ontology["attributes"][attr]


def all_values(ontology):
    attrs = list(ontology["attributes"]) + [ontology[k] for k in
            ("skill", "requirement", "location", "availability", "clearance")] + ["coordinator"]
    return sorted({v for attr in attrs for v in vocabulary(ontology, attr)})


# Shared by every candidate role. Names are sampled without looking at queries,
# values, status or labels; no special name bank is reserved for correct facts.
NAME_WORDS = {
    'software': 'Cedar Juniper Willow Maple Aspen Birch Rowan Linden Laurel Hazel Alder Elm Pine Spruce Cypress Oak Iris Aster Clover Fern Moss Sage Reed Heather',
    'hospital': 'Brookside Riverside Hillcrest Fairview Lakeside Meadowview Oakwood Woodside Sunnyvale Rosewood Glenwood Northgate Westhaven Eastbrook Southbank Clearview Springvale Greenfield Parkside Bayside Stonebridge Waterford Ashford Elmwood',
    'travel': 'Lisbon Verona Bergen Kyoto Siena Porto Lucerne Oxford Durham York Bath Brighton Dover Exeter Galway Cork Oslo Tallinn Riga Turin Parma Cadiz Nice Arles',
    'laboratory': 'Orchid Dahlia Iris Aster Clover Fern Moss Sage Reed Heather Violet Poppy Lily Lotus Olive Laurel Willow Cedar Maple Aspen Rowan Birch Hazel Alder',
    'restaurant': 'Basil Thyme Fennel Saffron Ginger Nutmeg Pepper Clove Parsley Rosemary Sorrel Mint Dill Chervil Tarragon Cinnamon Cardamom Sesame Juniper Lemon Olive Sage Bay Coriander',
    'school': 'Orion Lyra Vega Nova Aurora Comet Horizon Discovery Compass Lantern Beacon Mosaic Studio Meadow Summit Harbor Orchard Garden Willow Cedar Maple Aspen Rowan Birch',
}
# Each domain has nouns that can be spoken naturally without looking like an address.
ENTITY_PARTS = {
    'software': ('billing auth search checkout inventory payments analytics notifications catalog orders shipping identity sessions reports storage gateway scheduling accounts pricing audit sync profiles export import',
                 'api service worker library client adapter gateway cache queue scheduler proxy connector parser runtime agent toolkit module backend frontend indexer'),
    'hospital': ('cardiology neurology oncology pediatric surgical medical renal respiratory recovery maternity emergency outpatient imaging intake trauma dialysis orthopedics geriatric neonatal ophthalmology urology transplant radiology endocrine',
                 'ward unit clinic roster team service station rotation patient consultant nurse doctor registrar resident attendant coordinator fellow specialist therapist porter'),
    'travel': (NAME_WORDS['travel'], 'itinerary booking hotel flight train ferry tour transfer ticket reservation excursion museum visit connection journey pass permit visa document stop'),
    'laboratory': ('serum plasma tissue saliva water soil sediment culture enzyme protein RNA DNA buffer reagent control reference field river coastal thermal optical isotope fluorescent microbial',
                 'sample assay protocol instrument spectrometer microscope centrifuge freezer incubator detector batch plate column chamber scanner workflow calibration analysis extraction culture'),
    'restaurant': (NAME_WORDS['restaurant'], 'risotto soup stew salad roast tart pasta sauce stock delivery supplier farm kitchen menu station bakery grill pantry order batch'),
    'school': ('debate drama robotics chess music arts science coding astronomy history ecology film photography dance choir orchestra literature language math theater design makers volunteering sports',
               'club committee workshop rehearsal showcase festival meeting event session project organizer team coordinator society program practice activity assembly exhibition competition'),
}
ENTITY_PREFIXES = ('main regional central north south east west evening morning weekend '
                   'junior senior advanced general specialist community district campus local '
                   'primary secondary research training').split()
UPDATE_ACTIONS = 'refresh migration rollout revision adjustment cleanup upgrade review handover patch rebuild correction'.split()


def natural_names(keys, domain, seed, world, purpose='entity', occupied=()):
    """Sample a domain-specific bijection after ledger structure is frozen."""
    import itertools
    import random
    nouns, roles = (part.split() for part in ENTITY_PARTS[domain])
    if purpose == 'update':
        bank = [f'{noun} {action} {number}' for noun,action,number in
                itertools.product(nouns,UPDATE_ACTIONS,range(1,65))]
    elif domain == 'software':
        bank = [f'{prefix} {noun}-{role}' if role=='api' else f'{prefix} {noun} {role}' for prefix,noun,role in
                itertools.product(ENTITY_PREFIXES,nouns,roles)]
    else:
        bank = [f'{prefix} {noun} {role}' for prefix,noun,role in
                itertools.product(ENTITY_PREFIXES,nouns,roles)]
    random.Random(f'names-v7:{seed}:{world}:{domain}:{purpose}').shuffle(bank)
    occupied = set(occupied)
    bank = [name for name in bank if name not in occupied]
    keys = sorted(set(keys))
    if len(keys) > len(bank):
        raise ValueError('natural name bank exhausted; use smaller worlds')
    return dict(zip(keys,bank))
