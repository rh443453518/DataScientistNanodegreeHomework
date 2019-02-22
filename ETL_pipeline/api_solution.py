# possible solution for the API exercise
payload = {'format': 'json', 'per_page': '500', 'date':'2013:2016'}
r = requests.get('http://api.worldbank.org/v2/countries/in;cn/indicators/SP.POP.GROW', params=payload)

# clean the data and put it in a dictionary
data = defaultdict(list)
for entry in r.json()[1]:
    # check if country is already in dictionary. If so, append the new x and y values to the lists
    if data[entry['country']['value']]:
        data[entry['country']['value']][0].append(int(entry['date']))
        data[entry['country']['value']][1].append(float(entry['value']))       
    else: # if country not in dictionary, then initialize the lists that will hold the x and y values
        data[entry['country']['value']] = [[],[]] 