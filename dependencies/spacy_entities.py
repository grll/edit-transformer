"""Define in this file the spacy entities and the word they are mapping to a list of common instance or representation
    of that word.

"""
en_core_web_lg = {
    '<person>': ['person', 'people', 'man', 'woman'], # People, including fictional.
    '<norp>': ['english', 'french', 'italian', 'american', 'catholic', 'protestant', 'democrats', 'conservatives',
               'nationality', 'group'],  # Nationalities or religious or political groups.
    '<fac>': ['building', 'airport', 'highway', 'bridge'], # Buildings, airports, highways, bridges, etc.
    '<org>': ['Apple', 'Google', 'Facebook', 'Microsoft', 'company', 'agency',
              'institution'], # Companies, agencies, institutions, etc.
    '<gpe>': ['country', 'city', 'state', 'france', 'england', 'italy'], #	Countries, cities, states.
    '<loc>': ['location', 'mountain', 'alps', 'river', 'sea'], # Non-GPE locations, mountain ranges, bodies of water.
    '<product>': ['product', 'object', 'vehicle', 'food'], # Objects, vehicles, foods, etc. (Not services.)
    '<event>': ['event', 'hurricane', 'battle', 'war', 'sport'], # Named hurricanes, battles, wars, sports events, etc.
    '<work_of_art>': ['art', 'book', 'title', 'song', 'painting'], # Titles of books, songs, etc.
    '<law>': ['law'], # Named documents made into laws.
    '<language>': ['language', 'french', 'english', 'italian', 'spanish', 'german'], # Any named language.
    '<date>': ['date', 'period', 'time'], # Absolute or relative dates or periods.
    '<time>': ['time', 'hour', 'minute', 'second'], # Times smaller than a day.
    '<percent>': ['percentage', 'percent'], # Percentage, including ”%“.
    '<money>': ['dollar', 'euro'], # Monetary values, including unit.
    '<quantity>': ['measure', 'weight', 'distance', "meter", "kg"], # Measurements, as of weight or distance.
    '<ordinal>': ['ordinal', 'first', 'second', 'third', 'fourth'], # “first”, “second”, etc.
    '<cardinal>': ['number'] # Numerals that do not fall under another type.
}
en_core_wed_md = en_core_web_lg
en_core_web_sm = en_core_web_lg
