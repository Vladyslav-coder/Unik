def search_information(database, query):
    results = [item for item in database if query.lower() in item.lower()]
    return results

database = ["User Guide.pdf", "Video Tutorial.mp4", "Data Analysis Report.xlsx", "Project Plan.docx"]
query = "data"

search_results = search_information(database, query)
print("Результати пошуку:", search_results)
