import requests
import json

#entry = {'age': '50', 'workclass': 'Self-emp-not-inc', 'fnlgt': '83311', 'education': 'Bachelors', 'education_num': '13','marital_status': 'Married_civ_spouse','occupation': 'Exec_managerial', 'relationship': 'Husband','race': 'White','sex': 'Male','capital_gain': '0','capital_loss':'0','hours_per_week': '13', 'native_country': 'United-States'}
entry = {'age': '47', 'workclass': 'Private-gov', 'fnlgt': '51835', 'education': 'Prof-school', 'education_num': '15','marital_status': 'Married-civ-spouse','occupation': 'Prof-specialty', 'relationship': 'Wife','race': 'White','sex': 'Female','capital_gain': '0','capital_loss':'1902','hours_per_week': '60', 'native_country': 'Honduras'}


#response = requests.post('https://census-prosperity-app.herokuapp.com:5000/census/', data=json.dumps(entry))



response = requests.get('https://census-prosperity-app.herokuapp.com/', data=json.dumps(entry))

#response = requests.get('https://census-prosperity-app.herokuapp.com:5000/', data=json.dumps(entry))

#print(response.status_code)
#print(response.json())
