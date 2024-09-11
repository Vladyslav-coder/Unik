import pandas as pd

data = {
    'Product': ['Apple', 'Banana', 'Milk', 'Chiken', 'Rise', 'Spinach', 'Cheese', 'Nuts', 'Bread', 'Egg', 'Potato', 'Fish', 'Avocado', 'Oil', 'Shugar', 'Tomato', 'Carrot', 'Pineapple', 'Beef', 'Brocolli', 'Yogurt'],
    'Calories': [52, 89, 42, 165, 130, 23, 402, 576, 265, 155, 77, 206, 160, 884, 387, 18, 41, 50, 250, 55, 59],
    'Proteins': [0.3, 1.1, 3.4, 31, 2.7, 2.9, 25, 20, 9, 13, 2, 22, 2, 0, 0, 0.9, 0.9, 0.5, 26, 3.7, 10],
    'Fats': [0.2, 0.3, 1, 3.6, 0.3, 0.4, 33, 49, 3.2, 11, 0.1, 12, 15, 100, 0, 0.2, 0.1, 0.1, 15, 0.6, 0.4],
    'Carbohydrates': [14, 23, 5, 0, 28, 3.6, 1.3, 22, 49, 1.1, 17, 0, 8.5, 0, 100, 3.9, 9.6, 13, 0, 11.2, 7]
}

df = pd.DataFrame(data)

df['Price'] = [10, 15, 8, 50, 30, 5, 60, 100, 20, 12, 3, 40, 35, 150, 5, 2, 1, 25, 80, 4, 10]

df.loc[df['Product'] == 'Milk', 'Calories'] = 50

df.rename(columns={'Calories': 'Calorie content'}, inplace=True)

df['Protein from Calories'] = df.apply(lambda row: (row['Proteins'] * 4 / row['Calorie content']) * 100, axis=1)

df['Low in calories'] = df['Calorie content'].apply(lambda x: 'Yes' if x < 50 else 'No')

df['Energy value'] = df.apply(lambda row: row['Calorie content'] + row['Proteins']*4 + row['Fats']*9 + row['Carbohydrates']*4, axis=1)

print(df)

df.to_csv('products.csv', index=False)