import networkx as nx
import matplotlib.pyplot as plt

info_center_graph = nx.DiGraph()

nodes = ["Вхід (Файли/Запити)", "Обробка (Обробка даних)", "Вихід (Результати/Доступ)",
         "Зберігання", "Архів", "Модуль пошуку"]
info_center_graph.add_nodes_from(nodes)

edges = [
    ("Вхід (Файли/Запити)", "Обробка (Обробка даних)"),
    ("Обробка (Обробка даних)", "Зберігання"),
    ("Обробка (Обробка даних)", "Архів"),
    ("Зберігання", "Модуль пошуку"),
    ("Модуль пошуку", "Вихід (Результати/Доступ)"),
    ("Архів", "Вихід (Результати/Доступ)")
]
info_center_graph.add_edges_from(edges)

plt.figure(figsize=(10, 8))
nx.draw(info_center_graph, with_labels=True, node_color='skyblue', node_size=2000, 
        font_size=10, font_weight='bold', arrows=True, arrowstyle='-|>')  
plt.title("Structural Graph of the Information Center")
plt.show()
