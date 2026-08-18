import re

ALIASES = {
    "chickpeas": "chickpea",
    "garbanzo beans": "chickpea",
    "garbanzo": "chickpea",
    "aubergine": "eggplant",
    "courgette": "zucchini",
    "capsicum": "bell pepper",
    "scallions": "green onion",
    "spring onions": "green onion",
    "coriander": "cilantro",
    "chilli": "chili",
    "tomatoes": "tomato",
    "potatoes": "potato",
}
PANTRY = {"water", "salt", "pepper", "black pepper", "oil", "olive oil", "vegetable oil"}
UNITS = r"cups?|tbsp|tablespoons?|tsp|teaspoons?|ounces?|oz|grams?|g|kg|ml|liters?|lbs?|pounds?"


def normalize_ingredient(value: str) -> str:
    text = value.lower().replace("½", " 1/2 ").replace("¼", " 1/4 ").replace("¾", " 3/4 ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(rf"^\s*[\d\s./¼½¾-]+\s*(?:{UNITS})?\b", "", text)
    text = re.sub(r"\b(?:fresh|frozen|dried|chopped|diced|minced|sliced|optional|to taste)\b", " ", text)
    text = re.sub(r"[^a-z\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -")
    return ALIASES.get(text, text)


def split_ingredients(raw: str | list[str]) -> list[str]:
    if isinstance(raw, list):
        values = raw
    else:
        cleaned = raw.strip().strip("[]")
        values = re.split(r"\n|\s*[;|]\s*|',\s*'|\",\s*\"", cleaned)
    return [v.strip(" '\"•-") for v in values if v.strip(" '\"•-")]


def deterministic_parse(text: str) -> tuple[list[str], list[str]]:
    exclusions = re.findall(r"(?:no|without|exclude|allergic to)\s+([a-z][a-z -]{1,40})", text.lower())
    positive = re.sub(r"(?:no|without|exclude|allergic to)\s+[a-z][a-z -]{1,40}", "", text.lower())
    parts = re.split(r",|;|\band\b|\bwith\b|\bi have\b|\busing\b", positive)
    ingredients = [normalize_ingredient(x) for x in parts]
    return [x for x in ingredients if x], [normalize_ingredient(x) for x in exclusions]


def ingredient_matches(query: str, recipe: str) -> bool:
    return query == recipe or query in recipe or recipe in query
