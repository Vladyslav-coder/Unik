import numpy as np

alternatives = [
    {"name": "Легковий автомобіль", "cost": 10000, "eco": 6, "speed": 120},
    {"name": "Електромобіль", "cost": 15000, "eco": 9, "speed": 100},
    {"name": "Гібридний автомобіль", "cost": 13000, "eco": 8, "speed": 110},
    {"name": "Мотоцикл", "cost": 5000, "eco": 5, "speed": 90},
    {"name": "Велосипед", "cost": 1000, "eco": 10, "speed": 30}
]

def is_dominated(alt1, alt2, criteria):
    better = False
    for crit in criteria:
        if alt1[crit] < alt2[crit]:  
            return False
        if alt1[crit] > alt2[crit]:  
            better = True
    return better

criteria = ["cost", "eco", "speed"]
pareto_set = []

for alt1 in alternatives:
    dominated = False
    for alt2 in alternatives:
        if alt1 != alt2 and is_dominated(alt1, alt2, criteria):
            dominated = True
            break
    if not dominated:
        pareto_set.append(alt1)

print("Множина Парето:")
for alt in pareto_set:
    print(alt["name"])

weights = {"cost": 0.4, "eco": 0.4, "speed": 0.2}

def calculate_score(alt, weights):
    score = 0
    for crit, weight in weights.items():
        score += alt[crit] * weight
    return score

print("\nОцінки альтернатив:")
for alt in pareto_set:
    alt_score = calculate_score(alt, weights)
    print(f"{alt['name']}: {alt_score:.2f}")

def calculate_score(alt, weights):
    score = 0
    for crit, weight in weights.items():
        score += alt[crit] * weight
    return score

scores = {alt['name']: calculate_score(alt, weights) for alt in pareto_set}
best_alternative = max(scores, key=scores.get)

print("Найкраща альтернатива на множині Парето:", best_alternative)
